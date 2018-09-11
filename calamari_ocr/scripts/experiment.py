import argparse
import os
import copy
import numpy as np
import random
import inspect
from calamari_ocr.scripts.train import setup_train_args
import calamari_ocr.scripts.cross_fold_train as cross_fold_train
from calamari_ocr.scripts.eval import print_confusions, write_xlsx
from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.utils.multiprocessing import parallel_map, run, prefix_run_command

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
    files = args.train_files
    if args.n_lines > 0:
        all_files = glob_all(args.train_files)
        file = random.sample(all_files, args.n_lines)

    # run the cross-fold-training
    setattr(args, "max_parallel_models", args.max_parallel_models)
    setattr(args, "best_models_dir", best_models_dir)
    setattr(args, "temporary_dir", tmp_dir)
    setattr(args, "keep_temporary_files", False)
    setattr(args, "files", files)
    setattr(args, "best_model_label", "{id}")
    if not args.skip_train:
        cross_fold_train.main(args)

    dump_file = os.path.join(tmp_dir, "prediction.pkl")

    # run the prediction
    if not args.skip_eval:
        # locate the eval script (must be in the same dir as "this")
        predict_script_path = os.path.join(this_absdir, "experiment_eval.py")

        if len(args.single_fold) > 0:
            models = [os.path.join(best_models_dir, "{}.ckpt.json".format(sf)) for sf in args.single_fold]
            for m in models:
                if not os.path.exists(m):
                    raise Exception("Expected model at '{}', but file does not exist".format(m))
        else:
            models = [os.path.join(best_models_dir, d) for d in sorted(os.listdir(best_models_dir)) if d.endswith("json")]
            if len(models) != args.n_folds:
                raise Exception("Expected {} models, one for each fold respectively, but only {} models were found".format(
                    args.n_folds, len(models)
                ))

        for line in run(prefix_run_command([
                "python3", "-u",
                predict_script_path,
                "-j", str(args.num_threads),
                "--batch_size", str(args.batch_size),
                "--dump", dump_file,
                "--eval_imgs"] + args.eval_files + [
                ] + (["--verbose"] if args.verbose else []) + [
                "--checkpoint"] + models + [
                ], args.run, {"threads": args.num_threads}), verbose=args.verbose):
            # Print the output of the thread
            if args.verbose:
                print(line)

    import pickle
    with open(dump_file, 'rb') as f:
        prediction = pickle.load(f)

    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="The base directory where to store all working files")
    parser.add_argument("--eval_files", type=str, nargs="+", required=True,
                        help="All files that shall be used for evaluation")
    parser.add_argument("--train_files", type=str, nargs="+", required=True,
                        help="All files that shall be used for (cross-fold) training")
    parser.add_argument("--n_lines", type=int, default=[-1], nargs="+",
                        help="Optional argument to specify the number of lines (images) used for training. "
                             "On default, all available lines will be used.")
    parser.add_argument("--run", type=str, default=None,
                        help="An optional command that will receive the train calls. Useful e.g. when using a resource "
                             "manager such as slurm.")

    parser.add_argument("--n_folds", type=int, default=5,
                        help="The number of fold, that is the number of models to train")
    parser.add_argument("--max_parallel_models", type=int, default=-1,
                        help="Number of models to train in parallel per fold. Defaults to all.")
    parser.add_argument("--weights", type=str, nargs="+", default=[],
                        help="Load network weights from the given file. If more than one file is provided the number "
                             "models must match the number of folds. Each fold is then initialized with the weights "
                             "of each model, respectively")
    parser.add_argument("--single_fold", type=int, nargs="+", default=[],
                        help="Only train a single (list of single) specific fold(s).")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip the cross fold training")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip the cross fold evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--n_confusions", type=int, default=0,
                        help="Only print n most common confusions. Defaults to 0, use -1 for all.")
    parser.add_argument("--xlsx_output", type=str,
                        help="Optionally write a xlsx file with the evaluation results")

    setup_train_args(parser, omit=["files", "validation", "weights",
                                   "early_stopping_best_model_output_dir", "early_stopping_best_model_prefix",
                                   "output_dir"])

    args = parser.parse_args()

    args.base_dir = os.path.abspath(os.path.expanduser(args.base_dir))

    np.random.seed(args.seed)
    random.seed(args.seed)

    # argument checks
    if len(args.weights) > 1 and len(args.weights) != args.n_folds:
        raise Exception("Either no, one or n_folds (={}) models are required for pretraining but got {}.".format(
            args.n_folds, len(args.weights)
        ))

    if len(args.single_fold) > 0:
        if len(set(args.single_fold)) != len(args.single_fold):
            raise Exception("Repeated fold id's found.")
        for fold_id in args.single_fold:
            if fold_id < 0 or fold_id >= args.n_folds:
                raise Exception("Invalid fold id found: 0 <= id <= {}, but id == {}".format(args.n_folds, fold_id))

        actual_folds = args.single_fold
    else:
        actual_folds = list(range(args.n_folds))

    # run for all lines
    single_args = [copy.copy(args) for _ in args.n_lines]
    for s_args, n_lines in zip(single_args, args.n_lines):
        s_args.n_lines = n_lines

    predictions = parallel_map(run_for_single_line, single_args, progress_bar=False, processes=len(single_args), use_thread_pool=True)

    # output predictions as csv:
    header = "lines," + ",".join([str(fold) for fold in range(args.n_folds)])\
             + ",avg,std,seq. vot., def. conf. vot., fuz. conf. vot."

    print(header)

    for prediction_map, n_lines in zip(predictions, args.n_lines):
        prediction = prediction_map["full"]
        data = "{}".format(n_lines)
        folds_lers = []
        for fold in range(len(actual_folds)):
            eval = prediction[str(fold)]["eval"]
            data += ",{}".format(eval['avg_ler'])
            folds_lers.append(eval['avg_ler'])

        data += ",{},{}".format(np.mean(folds_lers), np.std(folds_lers))
        for voter in ['sequence_voter', 'confidence_voter_default_ctc', 'confidence_voter_fuzzy_ctc']:
            eval = prediction[voter]["eval"]
            data += ",{}".format(eval['avg_ler'])

        print(data)

    if args.n_confusions != 0:
        for prediction_map, n_lines in zip(predictions, args.n_lines):
            prediction = prediction_map["full"]
            print("")
            print("CONFUSIONS (lines = {})".format(n_lines))
            print("==========")
            print()

            for fold in range(len(actual_folds)):
                print("FOLD {}".format(fold))
                print_confusions(prediction[str(fold)]['eval'], args.n_confusions)

            for voter in ['sequence_voter', 'confidence_voter_default_ctc', 'confidence_voter_fuzzy_ctc']:
                print("VOTER {}".format(voter))
                print_confusions(prediction[voter]['eval'], args.n_confusions)

    if args.xlsx_output:
        data_list = []
        for prediction_map, n_lines in zip(predictions, args.n_lines):
            prediction = prediction_map["full"]
            for fold in actual_folds:
                pred = prediction[str(fold)]
                data_list.append({
                    "prefix": "L{} - Fold{}".format(n_lines, fold),
                    "results": pred['eval'],
                    "gt_files": prediction_map['gt_txts'],
                    "gts": prediction_map['gt'],
                    "preds": pred['data']
                })

            for voter in ['sequence_voter', 'confidence_voter_default_ctc']:
                pred = prediction[voter]
                data_list.append({
                    "prefix": "L{} - {}".format(n_lines, voter[:3]),
                    "results": pred['eval'],
                    "gt_files": prediction_map['gt_txts'],
                    "gts": prediction_map['gt'],
                    "preds": pred['data']
                })

        write_xlsx(args.xlsx_output, data_list)


if __name__=="__main__":
    main()
