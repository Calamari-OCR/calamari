import multiprocessing
import os
import inspect
import json
import tempfile
import sys

from calamari_ocr.ocr import CrossFold
from calamari_ocr.utils.multiprocessing import prefix_run_command, run

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
        sys.executable, "-u",
        args["train_script"],
        "--files", train_args_json,

    ], args.get("run", None), {"threads": args.get('num_threads', -1)}), verbose=args.get("verbose", False)):
        # Print the output of the thread
        if args.get("verbose", False):
            print("FOLD {} | {}".format(args["id"], line), end="")

    return args


class CrossFoldTrainer:
    def __init__(self, n_folds, dataset,
                 best_models_dir, best_model_label,
                 train_args,
                 progress_bars=False,
                 ):
        self.n_folds = n_folds
        self.dataset = dataset
        self.best_models_dir = best_models_dir
        self.best_model_label = best_model_label
        self.progress_bars = progress_bars
        self.train_args = train_args
        # locate the training script (must be in the same dir as "this")
        self.train_script_path = os.path.abspath(os.path.join(this_absdir, "..", "scripts", "train.py"))

        # location of best models output
        if not os.path.exists(self.best_models_dir):
            os.makedirs(self.best_models_dir)

        if not os.path.exists(self.train_script_path):
            raise FileNotFoundError("Missing train script path. Expected 'train.py' at {}".format(self.train_script_path))

        if not isinstance(train_args, dict):
            raise TypeError("Train args must be type of dict")

    def run(self, single_fold=None, seed=-1, weights=None, max_parallel_models=-1,
            temporary_dir=None, keep_temporary_files=False,
            ):
        # Default params
        single_fold = single_fold if single_fold else []
        weights = weights if weights else []
        if max_parallel_models <= 0:
            max_parallel_models = self.n_folds

        # argument checks
        if len(weights) > 1 and len(weights) != self.n_folds:
            raise Exception("Either no, one or n_folds (={}) models are required for pretraining but got {}.".format(
                self.n_folds, len(weights)
            ))

        if len(single_fold) > 0:
            if len(set(single_fold)) != len(single_fold):
                raise Exception("Repeated fold id's found.")
            for fold_id in single_fold:
                if fold_id < 0 or fold_id >= self.n_folds:
                    raise Exception("Invalid fold id found: 0 <= id <= {}, but id == {}".format(self.n_folds, fold_id))

        # create temporary dir
        # by default, the temporary files will be deleted after a successful training
        # if you specify a temporary dir, you can easily resume to train if an error occurred
        if keep_temporary_files and not temporary_dir:
            raise Exception("If you want to keep the temporary model files you have to specify a temporary dir")

        # temporary dir
        if temporary_dir is None:
            temporary_dir = tempfile.mkdtemp(prefix="calamari")
        else:
            temporary_dir = os.path.abspath(temporary_dir)

        if not os.path.exists(temporary_dir):
            os.makedirs(temporary_dir)

        # Compute the files in the cross fold (create a CrossFold)
        fold_file = os.path.join(temporary_dir, "folds.json")
        cross_fold = CrossFold(n_folds=self.n_folds, dataset=self.dataset, output_dir=temporary_dir,
                               progress_bar=self.progress_bars
                               )
        cross_fold.write_folds_to_json(fold_file)

        # Create the json argument file for each individual training
        run_args = []
        folds_to_run = single_fold if len(single_fold) > 0 else range(len(cross_fold.folds))
        for fold in folds_to_run:
            train_files = cross_fold.train_files(fold)
            test_files = cross_fold.test_files(fold)
            path = os.path.join(temporary_dir, "fold_{}.json".format(fold))
            with open(path, 'w') as f:
                fold_args = self.train_args.copy()
                fold_args["dataset"] = cross_fold.dataset_type.name
                fold_args["validation_dataset"] = cross_fold.dataset_type.name
                fold_args["validation_extension"] = self.train_args['gt_extension']
                fold_args["id"] = fold
                fold_args["files"] = train_files
                fold_args["validation"] = test_files
                fold_args["train_script"] = self.train_script_path
                fold_args["verbose"] = True
                fold_args["output_dir"] = os.path.join(temporary_dir, "fold_{}".format(fold))
                fold_args["early_stopping_best_model_output_dir"] = self.best_models_dir
                fold_args["early_stopping_best_model_prefix"] = self.best_model_label.format(id=fold)

                if seed >= 0:
                    fold_args["seed"] = seed + fold

                if len(weights) == 1:
                    fold_args["weights"] = weights[0]
                elif len(weights) > 1:
                    fold_args["weights"] = weights[fold]
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
        with multiprocessing.Pool(processes=max_parallel_models) as pool:
            # workaround to forward keyboard interrupt
            pool.map_async(train_individual_model, run_args).get()

        if not keep_temporary_files:
            import shutil
            shutil.rmtree(temporary_dir)
