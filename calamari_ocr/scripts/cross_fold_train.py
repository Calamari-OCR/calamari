import argparse
import json

from calamari_ocr.ocr import CrossFoldTrainer
from calamari_ocr.scripts.train import setup_train_args, DataSetType, create_train_dataset


def main(args=None):
    if args is None:
        # parse args from command line
        parser = argparse.ArgumentParser()

        # fold parameters
        parser.add_argument("--files", nargs="+",
                            help="List all image files that shall be processed. Ground truth fils with the same "
                                 "base name but with '.gt.txt' as extension are required at the same location. "
                                 "Optionally you can pass a single json file defining all arguments")
        parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
        parser.add_argument("--text_files", nargs="+", default=None,
                            help="Optional list of GT files if they are in other directory")
        parser.add_argument("--gt_extension", default=None,
                            help="Default extension of the gt files (expected to exist in same dir)")

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

    # check if loading a json file
    if len(args.files) == 1 and args.files[0].endswith("json"):
        with open(args.files[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                if key == 'dataset' or key == 'validation_dataset':
                    setattr(args, key, DataSetType.from_string(value))
                else:
                    setattr(args, key, value)

    dataset = create_train_dataset(args)

    trainer = CrossFoldTrainer(
        n_folds=args.n_folds,
        dataset=dataset,
        best_models_dir=args.best_models_dir,
        best_model_label=args.best_model_label,
        train_args=vars(args),
        progress_bars=not args.no_progress_bars,
    )
    trainer.run(
        args.single_fold, seed=args.seed, weights=args.weights, max_parallel_models=args.max_parallel_models,
        temporary_dir=args.temporary_dir, keep_temporary_files=args.keep_temporary_files,
    )


if __name__ == "__main__":
    main()
