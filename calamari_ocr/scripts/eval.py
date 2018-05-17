from argparse import ArgumentParser

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr import FileDataSet, Evaluator

def main():
    parser = ArgumentParser()
    parser.add_argument("--gt", nargs="+", required=True,
                        help="Ground truth files (.gt.txt extension)")
    parser.add_argument("--pred", nargs="+", default=None,
                        help="Prediction files if provided. Else files with .pred.txt are expected at the same "
                             "location as the gt.")
    parser.add_argument("--pred_ext", type=str, default=".pred.txt",
                        help="Extension of the predicted text files")
    parser.add_argument("--n_confusions", type=int, default=-1,
                        help="Only print n most common confusions")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads to use for evaluation")

    args = parser.parse_args()

    gt_files = sorted(glob_all(args.gt))

    if args.pred:
        pred_files = sorted(glob_all(args.pred))
        if len(pred_files) != len(gt_files):
            raise Exception("Mismatch in the number of gt and pred files: {} vs {}".format(
                len(gt_files), len(pred_files)))
    else:
        pred_files = [split_all_ext(gt)[0] + args.pred_ext for gt in gt_files]

    gt_data_set = FileDataSet(texts=gt_files)
    pred_data_set = FileDataSet(texts=pred_files)

    evaluator = Evaluator()
    r = evaluator.run(gt_dataset=gt_data_set, pred_dataset=pred_data_set, processes=args.num_threads, progress_bar=True)

    # TODO: More output
    print("Evaluation result")
    print("=================")
    print("")
    print("Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
        r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]))

    # sort descending
    if args.n_confusions != 0 and r["total_sync_errs"] > 0:
        total_percent = 0
        keys = sorted(r['confusion'].items(), key=lambda item: -item[1])
        print("{:8s} {:8s} {:8s} {:10s}".format("GT", "PRED", "COUNT", "PERCENT"))

        for i, ((gt, pred), count) in enumerate(keys):
            gt_fmt = "{" + gt + "}"
            pred_fmt = "{" + pred + "}"
            if i == args.n_confusions:
                break

            percent = count * max(len(gt), len(pred)) / r["total_sync_errs"]
            print("{:8s} {:8s} {:8d} {:10.2%}".format(gt_fmt, pred_fmt, count, percent))
            total_percent += percent

        print("The remaining but hidden errors make up {:.2%}".format(1.0 - total_percent))



if __name__ == '__main__':
    main()
