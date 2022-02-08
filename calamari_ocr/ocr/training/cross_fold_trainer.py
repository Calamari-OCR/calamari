import inspect
import json
import logging
import multiprocessing
import os
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from subprocess import check_call
from typing import Optional, List

from paiargparse import pai_dataclass, pai_meta
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr import CrossFold, SavedCalamariModel
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import (
    TrainerParams,
    CalamariDefaultTrainerPipelineParams,
)
from calamari_ocr.utils.multiprocessing import prefix_run_command, run

logger = logging.getLogger(__name__)

# path to the dir of this script to automatically detect the training script
this_absdir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))


def train_individual_model(run_args):
    # Call the training script with the json file as args
    # The json file contains all training parameters, including the files for training and validation
    # Note: It is necessary to launch a new thread because the command might be prefixed (e. g. use slurm as job
    #       scheduler to train all folds on different machines
    args = run_args["args"]
    train_args_json = run_args["json"]
    fold_logger = logging.getLogger(f"FOLD {run_args['id']}")
    for handler in fold_logger.handlers:
        handler.terminator = ""

    for out, err in run(
        prefix_run_command(
            [
                sys.executable,
                "-u",
                run_args["train_script"],
                train_args_json,
            ],
            run_args.get("run", None),
            {"threads": run_args.get("num_threads", -1)},
        ),
        verbose=run_args.get("verbose", False),
    ):
        # Print the output of the thread
        if run_args.get("verbose", False):
            if out:
                fold_logger.info(out.rstrip("\n"))
            if err:
                fold_logger.info(err.rstrip("\n"))

    return args


@pai_dataclass
@dataclass
class CrossFoldTrainerParams:
    trainer: TrainerParams = field(
        default_factory=CalamariScenario.default_trainer_params,
        metadata=pai_meta(mode="flat", help="The actual trainer params"),
    )
    n_folds: int = field(
        default=5,
        metadata=pai_meta(
            mode="flat",
            help="The number of fold, that is the number of models to train",
        ),
    )
    keep_temporary_files: bool = field(
        default=False,
        metadata=pai_meta(
            mode="flat",
            help="By default all temporary files (e.g. intermediate checkpoints) will be erased. Set this "
            "flag if you want to keep those files.",
        ),
    )
    best_models_dir: str = field(
        default="",
        metadata=pai_meta(
            mode="flat",
            required=True,
            help="path where to store the best models of each fold",
        ),
    )
    best_model_label: str = field(
        default="{id}",
        metadata=pai_meta(
            mode="flat",
            help="The label of the best model in best model dirs. This will be string formatted. "
            "The default '{id}' will label the models 0, 1, 2, 3, ...",
        ),
    )
    temporary_dir: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            mode="flat",
            help="A path to a temporary dir, where the intermediate model training data will be stored"
            "for each fold. Use --keep_temporary_files flag to keep the files. By default a system"
            "temporary dir will be used",
        ),
    )
    run_train: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            mode="flat",
            help="An optional command that will receive the train calls. Useful e.g. when using a resource "
            "manager such as slurm.",
        ),
    )
    run_split: Optional[str] = field(
        default=None,
        metadata=pai_meta(
            mode="flat",
            help="An optional command that will receive the split the data calls. Useful e.g. when using a resource "
            "manager such as slurm.",
        ),
    )
    max_parallel_models: int = field(
        default=-1,
        metadata=pai_meta(mode="flat", help="Number of models to train in parallel. Defaults to all."),
    )
    weights: List[str] = field(
        default_factory=list,
        metadata=pai_meta(
            mode="flat",
            help="Load network weights from the given file. If more than one file is provided the number "
            "models must match the number of folds. Each fold is then initialized with the weights "
            "of each model, respectively. If a model path is set to 'None', this model will start "
            "from scratch",
        ),
    )
    single_fold: List[int] = field(
        default_factory=list,
        metadata=pai_meta(mode="flat", help="Only train a single (list of single) specific fold(s)."),
    )
    visible_gpus: Optional[List[int]] = field(
        default=None,
        metadata=pai_meta(
            mode="flat",
            help="GPUs to use for scheduling the individual trainig."
            "Use e.g. 0, 1, 2 to schedule five trainings on 0, 1, 2, 0, 1 by setting CUDA_VISIBLE_DEVICES. "
            "This option should not be used with `--device.gpus`",
        ),
    )
    no_train: bool = field(default=False, metadata=pai_meta(mode="flat", help="Only create the folds. Do not train."))

    def __post_init__(self):
        if self.max_parallel_models <= 0:
            self.max_parallel_models = self.n_folds

        # argument checks
        if len(self.weights) > 1 and len(self.weights) != self.n_folds:
            raise Exception(
                "Either no, one or n_folds (={}) models are required for pretraining but got {}.".format(
                    self.n_folds, len(self.weights)
                )
            )

        if len(self.single_fold) > 0:
            if len(set(self.single_fold)) != len(self.single_fold):
                raise Exception("Repeated fold id's found.")
            for fold_id in self.single_fold:
                if fold_id < 0 or fold_id >= self.n_folds:
                    raise Exception("Invalid fold id found: 0 <= id <= {}, but id == {}".format(self.n_folds, fold_id))

        # by default, the temporary files will be deleted after a successful training
        # if you specify a temporary dir, you can easily resume to train if an error occurred
        if self.keep_temporary_files and not self.temporary_dir:
            raise ValueError("If you want to keep the temporary model files you have to specify a temporary dir")


class CrossFoldTrainer:
    def __init__(self, params: CrossFoldTrainerParams):
        self.params = params
        # locate the training script (must be in the same dir as "this")
        self.train_script_path = os.path.abspath(os.path.join(this_absdir, "../..", "scripts", "train_from_params.py"))
        self.cf_split_script_path = os.path.abspath(os.path.join(this_absdir, "standalone_cross_fold_split_script.py"))
        # location of best models output
        if not os.path.exists(self.params.best_models_dir):
            os.makedirs(self.params.best_models_dir)

        if not os.path.exists(self.train_script_path):
            raise FileNotFoundError(
                "Missing train script path. Expected 'train.py' at {}".format(self.train_script_path)
            )

        if not isinstance(self.params.trainer, TrainerParams):
            raise TypeError("Train args must be type of TrainerParams")

    def run(self):
        # temporary dir
        temporary_dir = self.params.temporary_dir
        if temporary_dir is None:
            temporary_dir = tempfile.mkdtemp(prefix="calamari")
        else:
            temporary_dir = os.path.abspath(temporary_dir)

        if not os.path.exists(temporary_dir):
            os.makedirs(temporary_dir)

        # Compute the files in the cross fold (create a CrossFold)
        fold_file = os.path.join(temporary_dir, "folds.json")
        cross_fold = CrossFold(
            n_folds=self.params.n_folds,
            data_generator_params=self.params.trainer.gen.train,
            output_dir=temporary_dir,
        )
        if self.params.run_split is not None:
            cfg_path = os.path.join(temporary_dir, "split.cfg.json")
            with open(cfg_path, "w") as f:
                json.dump(cross_fold.to_dict(), f)
            check_call(
                self.params.run_split.split()
                + [sys.executable, "-u", self.cf_split_script_path, cfg_path, "--dir", fold_file]
                + (["--progress_bar"] if self.params.trainer.progress_bar else []),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            with open(cfg_path) as f:
                cross_fold = CrossFold.from_dict(json.load(f))
        else:
            # do not run as separate call
            cross_fold.create_folds(self.params.trainer.progress_bar)
            cross_fold.write_folds_to_json(fold_file)

        # Create the json argument file for each individual training
        run_args = []
        seed = self.params.trainer.random_seed or -1
        folds_to_run = self.params.single_fold if len(self.params.single_fold) > 0 else range(len(cross_fold.folds))
        for fold in folds_to_run:
            train_files = cross_fold.train_files(fold)
            test_files = cross_fold.test_files(fold)
            path = os.path.join(temporary_dir, "fold_{}.json".format(fold))
            with open(path, "w") as f:
                trainer_params = deepcopy(self.params.trainer)
                trainer_params.gen = CalamariDefaultTrainerPipelineParams(
                    train=trainer_params.gen.train,
                    val=deepcopy(trainer_params.gen.train),
                    setup=trainer_params.gen.setup,
                )
                if cross_fold.is_h5_dataset:
                    tp = trainer_params.gen.train.to_dict()
                    del tp["__cls__"]
                    tp["files"] = train_files
                    trainer_params.gen.train = Hdf5.from_dict(tp)
                    vp = trainer_params.gen.val.to_dict()
                    del vp["__cls__"]
                    vp["files"] = test_files
                    trainer_params.gen.val = Hdf5.from_dict(vp)
                else:
                    trainer_params.gen.train.images = train_files
                    trainer_params.gen.val.images = test_files
                    trainer_params.gen.val.gt_extension = trainer_params.gen.train.gt_extension

                trainer_params.scenario.id = fold
                trainer_params.progress_bar_mode = 2
                trainer_params.output_dir = os.path.join(temporary_dir, "fold_{}".format(fold))
                trainer_params.early_stopping.best_model_output_dir = self.params.best_models_dir
                trainer_params.early_stopping.best_model_name = ""
                best_model_prefix = self.params.best_model_label.format(id=fold)
                trainer_params.best_model_prefix = best_model_prefix

                if self.params.visible_gpus:
                    assert trainer_params.device.gpus is None, "Using visible_gpus with device.gpus is not supported"
                    trainer_params.device.gpus = [self.params.visible_gpus[fold % len(self.params.visible_gpus)]]

                if seed >= 0:
                    trainer_params.random_seed = seed + fold

                if len(self.params.weights) == 1:
                    trainer_params.warmstart.model = self.params.weights[0]
                elif len(self.params.weights) > 1:
                    trainer_params.warmstart.model = self.params.weights[fold]

                # start from scratch via None
                if trainer_params.warmstart.model:
                    if (
                        len(trainer_params.warmstart.model.strip()) == 0
                        or trainer_params.warmstart.model.upper() == "NONE"
                    ):
                        trainer_params.warmstart.model = None
                    else:
                        # access model once to upgrade the model if necessary
                        # (can not be performed in parallel if multiple folds use the same model)
                        SavedCalamariModel(trainer_params.warmstart.model)

                post_init(trainer_params)

                json.dump(
                    trainer_params.to_dict(),
                    f,
                    indent=4,
                )

            run_args.append(
                {
                    "json": path,
                    "args": trainer_params,
                    "id": fold,
                    "train_script": self.train_script_path,
                    "run": self.params.run_train,
                    "verbose": True,
                }
            )

        if not self.params.no_train:
            logger.info("Starting the training.")
            # Launch the individual processes for each training
            with multiprocessing.pool.ThreadPool(processes=self.params.max_parallel_models) as pool:
                # workaround to forward keyboard interrupt
                pool.map_async(train_individual_model, run_args).get()
        else:
            logger.info("Not training since the `no_train` flag is set.")

        if not self.params.keep_temporary_files:
            import shutil

            shutil.rmtree(temporary_dir)
