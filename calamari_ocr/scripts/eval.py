from dataclasses import dataclass, field
from typing import Optional

import tfaip.util.logging
import numpy as np
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES
from paiargparse import PAIArgumentParser, pai_dataclass, pai_meta

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.evaluator import EvaluatorParams

logger = tfaip.util.logging.logger(__name__)


def print_confusions(r, n_confusions):
    # sort descending
    if n_confusions != 0 and r["total_sync_errs"] > 0:
        total_percent = 0
        keys = sorted(r["confusion"].items(), key=lambda item: -item[1])
        print("{:8s} {:8s} {:8s} {:10s}".format("GT", "PRED", "COUNT", "PERC (CER)"))

        for i, ((gt, pred), count) in enumerate(keys):
            gt_fmt = "{" + gt + "}"
            pred_fmt = "{" + pred + "}"
            if i == n_confusions:
                break

            percent = count * max(len(gt), len(pred)) / r["total_sync_errs"]
            print("{:8s} {:8s} {:8d} {:10.2%}".format(gt_fmt, pred_fmt, count, percent))
            total_percent += percent

        print("The remaining but hidden errors make up {:.2%}".format(1.0 - total_percent))


def print_worst_lines(r, n_worst_lines):
    if len(r["single"]) != len(r["ids"]):
        raise Exception("Mismatch in number of predictions and gt files")

    sorted_lines = sorted(zip(r["single"], r["ids"]), key=lambda a: -a[0][1])

    if n_worst_lines < 0:
        n_worst_lines = len(r["ids"])

    if n_worst_lines > 0:
        print("{:60s} {:4s} {:3s} {:3s} {}".format("GT FILE", "LEN", "ERR", "SER", "CONFUSIONS"))
        for (len_gt, errs, sync_errs, confusion, gt_pred), gtid in sorted_lines[:n_worst_lines]:
            print("{:60s} {:4d} {:3d} {:3d} {}".format(gtid[-60:], len_gt, errs, sync_errs, confusion))


def write_xlsx(xlsx_file, eval_datas):
    logger.info("Writing xlsx file to {}".format(xlsx_file))
    import xlsxwriter

    workbook = xlsxwriter.Workbook(xlsx_file)

    for eval_data in eval_datas:
        prefix = eval_data["prefix"]
        r = eval_data["results"]
        gt_files = eval_data["gt_files"]

        # all files
        ws = workbook.add_worksheet("{} - per line".format(prefix))

        for i, heading in enumerate(
            [
                "GT FILE",
                "GT",
                "PRED",
                "LEN",
                "ERR",
                "CER",
                "REL. ERR",
                "SYNC ERR",
                "CONFUSIONS",
            ]
        ):
            ws.write(0, i, heading)

        sorted_lines = sorted(zip(r["single"], gt_files), key=lambda a: -a[0][1])

        all_cs = []
        for i, ((len_gt, errs, sync_errs, confusion, (gt, pred)), gt_file) in enumerate(sorted_lines):
            ws.write(i + 1, 0, gt_file)
            ws.write(i + 1, 1, gt.strip())
            ws.write(i + 1, 2, pred.strip())
            ws.write(i + 1, 3, len_gt)
            ws.write(i + 1, 4, errs)
            ws.write(i + 1, 5, errs / max(len(gt), len(pred), 1))
            ws.write(i + 1, 6, errs / r["total_char_errs"] if r["total_char_errs"] > 0 else 0)
            ws.write(i + 1, 7, sync_errs)
            ws.write(i + 1, 8, "{}".format(confusion))
            all_cs.append(errs / max(len(gt), len(pred), 1))

        # total confusions
        ws = workbook.add_worksheet("{} - global".format(prefix))
        for i, heading in enumerate(["GT", "PRED", "COUNT", "PERC (CER)"]):
            ws.write(0, i, heading)

        keys = sorted(r["confusion"].items(), key=lambda item: -item[1])

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

        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": "CER hist",
                "categories": "='{}'!$A$2:$A${}".format(hsl, 2 + len(bin_edges)),
                "values": "='{}'!$B$2:$B${}".format(hsl, 2 + len(bin_edges)),
            }
        )
        chart.set_title({"name": "CER distribution"})
        chart.set_x_axis({"name": "CER"})
        chart.set_y_axis({"name": "Amount"})

        ws.insert_chart("D2", chart, {"x_offset": 25, "y_offset": 10})

    workbook.close()


@pai_dataclass
@dataclass
class EvalArgs:
    gt: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(help="GT", mode="flat", choices=DATA_GENERATOR_CHOICES),
    )
    pred: Optional[CalamariDataGeneratorParams] = field(
        default=None,
        metadata=pai_meta(
            help="Optional prediction dataset",
            mode="flat",
            choices=DATA_GENERATOR_CHOICES,
        ),
    )
    n_confusions: int = field(
        default=10,
        metadata=pai_meta(
            help="Only print n most common confusions. Defaults to 10, use -1 for all.",
            mode="flat",
        ),
    )
    n_worst_lines: int = field(
        default=0,
        metadata=pai_meta(help="Print the n worst recognized text lines with its error", mode="flat"),
    )
    xlsx_output: Optional[str] = field(
        default=None,
        metadata=pai_meta(help="Optionally write a xlsx file with the evaluation results", mode="flat"),
    )
    skip_empty_gt: bool = field(
        default=False,
        metadata=pai_meta(help="Ignore lines of the gt that are empty.", mode="flat"),
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            help="Specify an optional checkpoint to parse the text preprocessor (for the gt txt files)",
            mode="flat",
        ),
    )
    evaluator: EvaluatorParams = field(
        default_factory=EvaluatorParams,
        metadata=pai_meta(
            mode="flat",
            fix_dc=True,
        ),
    )


def run():
    main(parse_args())


def parse_args(args=None):
    parser = PAIArgumentParser()
    parser.add_root_argument("root", EvalArgs, ignore=["gt.images", "pred.images"])
    return parser.parse_args(args=args).root


def main(args: EvalArgs):
    # Local imports (imports that require tensorflow)
    from calamari_ocr.ocr.scenario import CalamariScenario
    from calamari_ocr.ocr.dataset.data import Data
    from calamari_ocr.ocr.evaluator import Evaluator

    if args.checkpoint:
        saved_model = SavedCalamariModel(args.checkpoint, auto_update=True)
        trainer_params = CalamariScenario.trainer_cls().params_cls().from_dict(saved_model.dict)
        data_params = trainer_params.scenario.data
    else:
        data_params = Data.default_params()

    data = Data(data_params)

    pred_data = args.pred if args.pred is not None else args.gt.to_prediction()
    evaluator = Evaluator(args.evaluator, data=data)
    evaluator.preload_gt(gt_dataset=args.gt)
    r = evaluator.run(gt_dataset=args.gt, pred_dataset=pred_data)

    # TODO: More output
    print("Evaluation result")
    print("=================")
    print("")
    print(
        "Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
            r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]
        )
    )

    # sort descending
    print_confusions(r, args.n_confusions)
    print_worst_lines(r, args.n_worst_lines)

    if args.xlsx_output:
        write_xlsx(
            args.xlsx_output,
            [
                {
                    "prefix": "evaluation",
                    "results": r,
                    "gt_files": r["ids"],
                }
            ],
        )

    return r


if __name__ == "__main__":
    run()
