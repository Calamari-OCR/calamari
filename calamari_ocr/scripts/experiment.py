import argparse
import os
import copy
import numpy as np
import random
import inspect

from tfaip.util.multiprocessing.parallelmap import parallel_map

from calamari_ocr.ocr import DataSetType
from calamari_ocr.scripts.train import setup_train_args
import calamari_ocr.scripts.train as train_script
from calamari_ocr.scripts.eval import print_confusions, write_xlsx
from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.utils.multiprocessing import run, prefix_run_command

# path to the dir of this script to automatically detect the training script
this_absdir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


# create necessary directories
def run_for_single_line(args):
    # lines/network/pretraining as base dir
    args.base_dir = os.path.join(args.base_dir, "all" if args.n_lines < 0 else str(args.n_lines))
    pretrain_prefix = "scratch"
    if args.weights and len(args.weights) > 0:
        pretrain_prefix = ",".join([split_all_ext(os.path.basename(path))[0] for path in args.weights])

    args.base_dir = os.path.join(args.base_dir, args.network, pretrain_prefix)

    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)

    tmp_dir = os.path.join(args.base_dir, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    best_models_dir = os.path.join(args.base_dir, "models")
    if not os.path.exists(best_models_dir):
        os.makedirs(best_models_dir)

    prediction_dir = os.path.join(args.base_dir, "predictions")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # select number of files
    files = args.files
    if args.n_lines > 0:
        all_files = glob_all(args.files)
        files = random.sample(all_files, args.n_lines)

    # run the cross-fold-training
    setattr(args, "early_stopping_best_model_output_dir", best_models_dir)
    setattr(args, "output_dir", tmp_dir)
    setattr(args, "keep_temporary_files", False)
    setattr(args, "files", files)
    setattr(args, "text_files", None)
    setattr(args, "gt_extension", None)
    setattr(args, "dataset", DataSetType.FILE)
    setattr(args, "best_model_label", "{id}")
    if not args.skip_train:
        train_script.main(args)

    dump_file = os.path.join(tmp_dir, "prediction.pkl")

    # run the prediction
    if not args.skip_eval:
        # locate the eval script (must be in the same dir as "this")
        predict_script_path = os.path.join(this_absdir, "predict_and_eval.py")

        model = os.path.join(best_models_dir, "best.ckpt.json")
        if not os.path.exists(model):
            raise Exception(f"Expected model at '{model}', but file does not exist")

        for line in run(
            prefix_run_command(
                [
                    "python3",
                    "-u",
                    predict_script_path,
                    "-j",
                    str(args.num_threads),
                    "--batch_size",
                    str(args.batch_size),
                    "--dump",
                    dump_file,
                    "--files",
                ]
                + args.eval_files
                + []
                + (["--verbose"] if args.verbose else [])
                + ["--checkpoint"]
                + [model]
                + [],
                args.run,
                {"threads": args.num_threads},
            ),
            verbose=args.verbose,
        ):
            # Print the output of the thread
            if args.verbose:
                print(line)

    import pickle

    with open(dump_file, "rb") as f:
        prediction = pickle.load(f)

    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="The base directory where to store all working files",
    )
    parser.add_argument(
        "--eval_files",
        type=str,
        nargs="+",
        required=True,
        help="All files that shall be used for evaluation",
    )
    parser.add_argument(
        "--n_lines",
        type=int,
        default=[-1],
        nargs="+",
        help="Optional argument to specify the number of lines (images) used for training. "
        "On default, all available lines will be used.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="An optional command that will receive the train calls. Useful e.g. when using a resource "
        "manager such as slurm.",
    )

    parser.add_argument("--skip_train", action="store_true", help="Skip the cross fold training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip the cross fold evaluation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--n_confusions",
        type=int,
        default=0,
        help="Only print n most common confusions. Defaults to 0, use -1 for all.",
    )
    parser.add_argument(
        "--xlsx_output",
        type=str,
        help="Optionally write a xlsx file with the evaluation results",
    )

    setup_train_args(parser, omit=["early_stopping_best_model_output_dir", "output_dir"])

    args = parser.parse_args()

    args.base_dir = os.path.abspath(os.path.expanduser(args.base_dir))

    np.random.seed(args.seed)
    random.seed(args.seed)

    # run for all lines
    single_args = [copy.copy(args) for _ in args.n_lines]
    for s_args, n_lines in zip(single_args, args.n_lines):
        s_args.n_lines = n_lines

    predictions = parallel_map(
        run_for_single_line,
        single_args,
        progress_bar=False,
        processes=len(single_args),
        use_thread_pool=True,
    )
    predictions = list(predictions)

    # output predictions as csv:
    header = "lines," + ",".join([str(fold) for fold in range(len(predictions[0]["full"]) - 1)]) + ",avg,std,voted"

    print(header)

    for prediction_map, n_lines in zip(predictions, args.n_lines):
        prediction = prediction_map["full"]
        data = "{}".format(n_lines)
        folds_lers = []
        for fold, pred in prediction.items():
            if fold == "voted":
                continue

            eval = pred["eval"]
            data += ",{}".format(eval["avg_ler"])
            folds_lers.append(eval["avg_ler"])

        data += ",{},{}".format(np.mean(folds_lers), np.std(folds_lers))
        eval = prediction["voted"]["eval"]
        data += ",{}".format(eval["avg_ler"])

        print(data)

    if args.n_confusions != 0:
        for prediction_map, n_lines in zip(predictions, args.n_lines):
            prediction = prediction_map["full"]
            print("")
            print("CONFUSIONS (lines = {})".format(n_lines))
            print("==========")
            print()

            for fold, pred in prediction.items():
                print("FOLD {}".format(fold))
                print_confusions(pred["eval"], args.n_confusions)

    if args.xlsx_output:
        data_list = []
        for prediction_map, n_lines in zip(predictions, args.n_lines):
            prediction = prediction_map["full"]
            for fold, pred in prediction.items():
                data_list.append(
                    {
                        "prefix": "L{} - Fold{}".format(n_lines, fold),
                        "results": pred["eval"],
                        "gt_files": prediction_map["gt_txts"],
                        "gts": prediction_map["gt"],
                        "preds": pred["data"],
                    }
                )

            for voter in ["sequence_voter", "confidence_voter_default_ctc"]:
                pred = prediction[voter]
                data_list.append(
                    {
                        "prefix": "L{} - {}".format(n_lines, voter[:3]),
                        "results": pred["eval"],
                        "gt_files": prediction_map["gt_txts"],
                        "gts": prediction_map["gt"],
                        "preds": pred["data"],
                    }
                )

        write_xlsx(args.xlsx_output, data_list)


if __name__ == "__main__":
    main()
