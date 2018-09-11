from argparse import ArgumentParser
import os
import numpy as np

from google.protobuf import json_format

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr import Evaluator
from calamari_ocr.ocr.datasets import create_dataset, DataSetType
from calamari_ocr.proto import CheckpointParams
from calamari_ocr.ocr.text_processing import text_processor_from_proto


def print_confusions(r, n_confusions):
    # sort descending
    if n_confusions != 0 and r["total_sync_errs"] > 0:
        total_percent = 0
        keys = sorted(r['confusion'].items(), key=lambda item: -item[1])
        print("{:8s} {:8s} {:8s} {:10s}".format("GT", "PRED", "COUNT", "PERCENT"))

        for i, ((gt, pred), count) in enumerate(keys):
            gt_fmt = "{" + gt + "}"
            pred_fmt = "{" + pred + "}"
            if i == n_confusions:
                break

            percent = count * max(len(gt), len(pred)) / r["total_sync_errs"]
            print("{:8s} {:8s} {:8d} {:10.2%}".format(gt_fmt, pred_fmt, count, percent))
            total_percent += percent

        print("The remaining but hidden errors make up {:.2%}".format(1.0 - total_percent))


def print_worst_lines(r, gt_samples, preds, n_worst_lines):
    if len(r["single"]) != len(gt_samples) or len(gt_samples) != len(preds):
        raise Exception("Mismatch in number of predictions and gt files")

    sorted_lines = sorted(zip(r["single"], gt_samples, preds), key=lambda a: -a[0][1])

    if n_worst_lines < 0:
        n_worst_lines = len(gt_samples)

    if n_worst_lines > 0:
        print("{:60s} {:4s} {:3s} {:3s} {}".format("GT FILE", "LEN", "ERR", "SER", "CONFUSIONS"))
        for (len_gt, errs, sync_errs, confusion), gt, pred in sorted_lines[:n_worst_lines]:
            print("{:60s} {:4d} {:3d} {:3d} {}".format(gt['id'][-60:], len_gt, errs, sync_errs, confusion))


def write_xlsx(xlsx_file, eval_datas):
    print("Writing xlsx file to {}".format(xlsx_file))
    import xlsxwriter
    workbook = xlsxwriter.Workbook(xlsx_file)

    for eval_data in eval_datas:
        prefix = eval_data["prefix"]
        r = eval_data["results"]
        gt_files = eval_data["gt_files"]
        gts = eval_data["gts"]
        preds = eval_data["preds"]

        # all files
        ws = workbook.add_worksheet("{} - per line".format(prefix))

        for i, heading in enumerate(["GT FILE", "GT", "PRED", "LEN", "ERR", "CER", "REL. ERR", "SYNC ERR", "CONFUSIONS"]):
            ws.write(0, i, heading)

        sorted_lines = sorted(zip(r["single"], gt_files, gts, preds), key=lambda a: -a[0][1])

        all_cs = []
        for i, ((len_gt, errs, sync_errs, confusion), gt_file, gt, pred) in enumerate(sorted_lines):
            ws.write(i + 1, 0, gt_file)
            ws.write(i + 1, 1, gt.strip())
            ws.write(i + 1, 2, pred.strip())
            ws.write(i + 1, 3, len_gt)
            ws.write(i + 1, 4, errs)
            ws.write(i + 1, 5, errs / max(len(gt), len(pred)))
            ws.write(i + 1, 6, errs / r["total_char_errs"])
            ws.write(i + 1, 7, sync_errs)
            ws.write(i + 1, 8, "{}".format(confusion))
            all_cs.append(errs / max(len(gt), len(pred)))

        # total confusions
        ws = workbook.add_worksheet("{} - global".format(prefix))
        for i, heading in enumerate(["GT", "PRED", "COUNT", "PERCENT"]):
            ws.write(0, i, heading)

        keys = sorted(r['confusion'].items(), key=lambda item: -item[1])

        for i, ((gt, pred), count) in enumerate(keys):
            gt_fmt = "{" + gt + "}"
            pred_fmt = "{" + pred + "}"

            percent = count * max(len(gt), len(pred)) / r["total_sync_errs"]
            ws.write(i + 1, 0, gt_fmt)
            ws.write(i + 1, 1, pred_fmt)
            ws.write(i + 1, 2, count)
            ws.write(i + 1, 3, percent)

        # histogram of cers
        hsl = "{} - histogram".format(prefix)
        ws = workbook.add_worksheet(hsl)
        ws.write_row("A1", ["Class", "Count"])
        hist, bin_edges = np.histogram(all_cs, bins="auto")
        ws.write_column("A2", bin_edges)
        ws.write_column("B2", hist)

        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({'name': "CER hist",
                          'categories': "='{}'!$A$2:$A${}".format(hsl, 2 + len(bin_edges)),
                          'values': "='{}'!$B$2:$B${}".format(hsl, 2 + len(bin_edges))
                          })
        chart.set_title({'name': 'CER distribution'})
        chart.set_x_axis({'name': 'CER'})
        chart.set_y_axis({'name': 'Amount'})

        ws.insert_chart("D2", chart, {"x_offset": 25, 'y_offset': 10})

    workbook.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--gt", nargs="+", required=True,
                        help="Ground truth files (.gt.txt extension)")
    parser.add_argument("--pred", nargs="+", default=None,
                        help="Prediction files if provided. Else files with .pred.txt are expected at the same "
                             "location as the gt.")
    parser.add_argument("--pred_ext", type=str, default=".pred.txt",
                        help="Extension of the predicted text files")
    parser.add_argument("--n_confusions", type=int, default=10,
                        help="Only print n most common confusions. Defaults to 10, use -1 for all.")
    parser.add_argument("--n_worst_lines", type=int, default=0,
                        help="Print the n worst recognized text lines with its error")
    parser.add_argument("--xlsx_output", type=str,
                        help="Optionally write a xlsx file with the evaluation results")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads to use for evaluation")
    parser.add_argument("--non_existing_file_handling_mode", type=str, default="error",
                        help="How to handle non existing .pred.txt files. Possible modes: skip, empty, error. "
                             "'Skip' will simply skip the evaluation of that file (not counting it to errors). "
                             "'Empty' will handle this file as would it be empty (fully checking for errors)."
                             "'Error' will throw an exception if a file is not existing. This is the default behaviour.")
    parser.add_argument("--no_progress_bars", action="store_true",
                        help="Do not show any progress bars")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specify an optional checkpoint to parse the text preprocessor (for the gt txt files)")

    args = parser.parse_args()

    print("Resolving files")
    gt_files = sorted(glob_all(args.gt))

    if args.pred:
        pred_files = sorted(glob_all(args.pred))
        if len(pred_files) != len(gt_files):
            raise Exception("Mismatch in the number of gt and pred files: {} vs {}".format(
                len(gt_files), len(pred_files)))
    else:
        pred_files = [split_all_ext(gt)[0] + args.pred_ext for gt in gt_files]

    if args.non_existing_file_handling_mode.lower() == "skip":
        non_existing_pred = [p for p in pred_files if not os.path.exists(p)]
        for f in non_existing_pred:
            idx = pred_files.index(f)
            del pred_files[idx]
            del gt_files[idx]

    text_preproc = None
    if args.checkpoint:
        with open(args.checkpoint if args.checkpoint.endswith(".json") else args.checkpoint + '.json', 'r') as f:
            checkpoint_params = json_format.Parse(f.read(), CheckpointParams())
            text_preproc = text_processor_from_proto(checkpoint_params.model.text_preprocessor)

    non_existing_as_empty = args.non_existing_file_handling_mode.lower() == "empty"
    gt_data_set = create_dataset(
        args.dataset,
        texts=gt_files,
        non_existing_as_empty=non_existing_as_empty
    )
    pred_data_set = create_dataset(
        args.dataset,
        texts=pred_files,
        non_existing_as_empty=non_existing_as_empty
    )

    evaluator = Evaluator(text_preprocessor=text_preproc)
    r = evaluator.run(gt_dataset=gt_data_set, pred_dataset=pred_data_set, processes=args.num_threads,
                      progress_bar=not args.no_progress_bars)

    # TODO: More output
    print("Evaluation result")
    print("=================")
    print("")
    print("Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
        r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]))

    # sort descending
    print_confusions(r, args.n_confusions)

    print_worst_lines(r, gt_data_set.samples(), pred_data_set.text_samples(), args.n_worst_lines)

    if args.xlsx_output:
        write_xlsx(args.xlsx_output,
                   [{
                       "prefix": "evaluation",
                       "results": r,
                       "gt_files": gt_files,
                       "gts": gt_data_set.text_samples(),
                       "preds": pred_data_set.text_samples()
                   }])

if __name__ == '__main__':
    main()
