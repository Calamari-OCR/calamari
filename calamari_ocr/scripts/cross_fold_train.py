import argparse
import tempfile
import os
import inspect
import multiprocessing
import json

from calamari_ocr.ocr.cross_fold import CrossFold
from calamari_ocr.utils.multiprocessing import prefix_run_command, run
from calamari_ocr.scripts.train import setup_train_args

# path to the dir of this script to automatically detect the training script
this_absdir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def train_individual_model(run_args):
    # Call the training script with the json file as args
    # The json file contains all training parameters, including the files for training and validation
    # Note: It is necessary to launch a new thread because the command might be prefixed (e. g. use slurm as job
    #       skeduler to train all folds on different machines
    args = run_args["args"]
    train_args_json = run_args["json"]
    for line in run(prefix_run_command([
        "python3", "-u",
        args["train_script"],
        "--files", train_args_json,

    ], args["run"], {"threads": args['num_threads']}), verbose=args["verbose"]):
        # Print the output of the thread
        if args["verbose"]:
            print("FOLD {} | {}".format(args["id"], line), end="")

    return args


def main(args=None):
    if args is None:
        # parse args from command line
        parser = argparse.ArgumentParser()

        # fold parameters
        parser.add_argument("--files", nargs="+",
                            help="List all image files that shall be processed. Ground truth fils with the same "
                                 "base name but with '.gt.txt' as extension are required at the same location")
        parser.add_argument("--n_folds", type=int, default=5,
                            help="The number of fold, that is the number of models to train")
        parser.add_argument("--keep_temporary_files", action="store_true",
                            help="By default all temporary files (e.g. intermediate checkpoints) will be erased. Set this "
                                 "flag if you want to keep those files.")
        parser.add_argument("--best_models_dir", type=str, required=True,
                            help="path where to store the best models of each fold")
        parser.add_argument("--best_model_label", type=str, default="{id}",
                            help="The label of the best model in best model dirs. This will be string formatted. "
                                 "The default '{id}' will label the models 0, 1, 2, 3, ...")
        parser.add_argument("--temporary_dir", type=str, default=None,
                            help="A path to a temporary dir, where the intermediate model training data will be stored"
                                 "for each fold. Use --keep_temporary_files flag to keep the files. By default a system"
                                 "temporary dir will be used")
        parser.add_argument("--run", type=str, default=None,
                            help="An optional command that will receive the train calls. Useful e.g. when using a resource "
                                 "manager such as slurm.")
        parser.add_argument("--max_parallel_models", type=int, default=-1,
                            help="Number of models to train in parallel. Defaults to all.")
        parser.add_argument("--weights", type=str, nargs="+", default=[],
                            help="Load network weights from the given file. If more than one file is provided the number "
                                 "models must match the number of folds. Each fold is then initialized with the weights "
                                 "of each model, respectively. If a model path is set to 'None', this model will start "
                                 "from scratch")
        parser.add_argument("--single_fold", type=int, nargs="+", default=[],
                            help="Only train a single (list of single) specific fold(s).")

        # add the training args (omit those params, that are set by the cross fold training)
        setup_train_args(parser, omit=["files", "validation", "weights",
                                       "early_stopping_best_model_output_dir", "early_stopping_best_model_prefix",
                                       "output_dir"])

        args = parser.parse_args()

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

    # automatically set the number of models that shall be run in parallel
    if args.max_parallel_models <= 0:
        args.max_parallel_models = args.n_folds

    # by default, the temporary files will be deleted after a successful training
    # if you specify a temporary dir, you can easily resume to train if an error occurred
    if args.keep_temporary_files and not args.temporary_dir:
        raise Exception("If you want to keep the temporary model files you have to specify a temporary dir")

    if not args.temporary_dir:
        args.temporary_dir = tempfile.mkdtemp(prefix="calamari")
    else:
        args.temporary_dir = os.path.abspath(args.temporary_dir)
        if not os.path.exists(args.temporary_dir):
            os.makedirs(args.temporary_dir)

    # location of best models output
    if not os.path.exists(args.best_models_dir):
        os.makedirs(args.best_models_dir)

    # locate the training script (must be in the same dir as "this")
    train_script_path = os.path.join(this_absdir, "train.py")

    if not os.path.exists(train_script_path):
        raise Exception("Missing train script path. Expected 'train.py' at {}".format(this_absdir))

    # Compute the files in the cross fold (create a CrossFold)
    fold_file = os.path.join(args.temporary_dir, "folds.json")
    cross_fold = CrossFold(n_folds=args.n_folds, source_files=args.files, output_dir=args.best_models_dir)
    cross_fold.write_folds_to_json(fold_file)

    # Create the json argument file for each individual training
    run_args = []
    folds_to_run = args.single_fold if len(args.single_fold) > 0 else range(len(cross_fold.folds))
    for fold in folds_to_run:
        train_files = cross_fold.train_files(fold)
        test_files = cross_fold.test_files(fold)
        path = os.path.join(args.temporary_dir, "fold_{}.json".format(fold))
        with open(path, 'w') as f:
            fold_args = vars(args).copy()
            fold_args["id"] = fold
            fold_args["files"] = train_files
            fold_args["validation"] = test_files
            fold_args["train_script"] = train_script_path
            fold_args["verbose"] = True
            fold_args["output_dir"] = os.path.join(args.temporary_dir, "fold_{}".format(fold))
            fold_args["early_stopping_best_model_output_dir"] = args.best_models_dir
            fold_args["early_stopping_best_model_prefix"] = args.best_model_label.format(id=fold)

            if args.seed >= 0:
                fold_args["seed"] = args.seed + fold

            if len(args.weights) == 1:
                fold_args["weights"] = args.weights[0]
            elif len(args.weights) > 1:
                fold_args["weights"] = args.weights[fold]
            else:
                fold_args["weights"] = None

            # start from scratch via None
            if fold_args["weights"]:
                if len(fold_args["weights"].strip()) == 0 or fold_args["weights"].upper() == "NONE":
                    fold_args["weights"] = None

            json.dump(
                fold_args,
                f,
                indent=4,
            )

        run_args.append({"json": path, "args": fold_args})

    # Launch the individual processes for each training
    with multiprocessing.Pool(processes=args.max_parallel_models) as pool:
        # workaround to forward keyboard interrupt
        pool.map_async(train_individual_model, run_args).get(999999999)


if __name__ == "__main__":
    main()
