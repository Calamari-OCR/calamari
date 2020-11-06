import argparse
import os
import json
from tfaip.util.logging import setup_log

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount

from calamari_ocr import __version__
from calamari_ocr.ocr.backends.dataset.data_types import CalamariDataParams
from calamari_ocr.ocr.backends.dataset.datareader.factory import FileDataReaderFactory, FileDataReaderArgs
from calamari_ocr.ocr.backends.scenario import CalamariScenario
from calamari_ocr.ocr.data_processing import DefaultDataPreprocessor
from calamari_ocr.ocr.text_processing.text_regularizer import default_text_regularizer_replacements
from calamari_ocr.proto.converters import params_from_definition_string
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
from calamari_ocr.ocr.datasets import create_data_reader, DataSetType, DataSetMode
from calamari_ocr.ocr.augmentation.data_augmenter import SimpleDataAugmenter
from calamari_ocr.ocr.text_processing import \
    MultiTextProcessor, TextNormalizer, \
    TextRegularizer, StripTextProcessor, BidiTextProcessor


from calamari_ocr.proto.params import *


def setup_train_args(parser, omit=None):
    if omit is None:
        omit = []

    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    if "files" not in omit:
        parser.add_argument("--files", nargs="+", default=[],
                            help="List all image files that shall be processed. Ground truth fils with the same "
                                 "base name but with '.gt.txt' as extension are required at the same location")
        parser.add_argument("--text_files", nargs="+", default=None,
                            help="Optional list of GT files if they are in other directory")
        parser.add_argument("--gt_extension", default=None,
                            help="Default extension of the gt files (expected to exist in same dir)")
        parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)

    parser.add_argument("--train_data_on_the_fly", action='store_true', default=False,
                        help='Instead of preloading all data during the training, load the data on the fly. '
                             'This is slower, but might be required for limited RAM or large datasets')

    parser.add_argument("--seed", type=int, default="0",
                        help="Seed for random operations. If negative or zero a 'random' seed is used")
    parser.add_argument("--network", type=str, default="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
                        help="The network structure")
    parser.add_argument("--line_height", type=int, default=48,
                        help="The line height")
    parser.add_argument("--pad", type=int, default=16,
                        help="Padding (left right) of the line")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="The number of threads to use for all operations")
    parser.add_argument("--display", type=int, default=1,
                        help="Frequency of how often an output shall occur during training [epochs]")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size to use for training")
    parser.add_argument("--checkpoint_frequency", type=int, default=-1,
                        help="The frequency how often to write checkpoints during training [epochs]"
                             "If -1, the early_stopping_frequency will be used.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="The number of iterations for training. "
                             "If using early stopping, this is the maximum number of iterations")
    # TODO: allow negative values to give fraction of sample size e.g. -1 = complete dataset
    parser.add_argument("--samples_per_epoch", type=int, default=100,
                        help="The number of samples to process per epoch"
                        )
    parser.add_argument("--early_stopping_at_accuracy", type=float, default=1.0,
                        help="Stop training if the early stopping accuracy reaches this value")
    parser.add_argument("--stats_size", type=int, default=100,
                        help="Average this many iterations for computing an average loss, label error rate and "
                             "training time")
    parser.add_argument("--no_skip_invalid_gt", action="store_true",
                        help="Do no skip invalid gt, instead raise an exception.")
    parser.add_argument("--no_progress_bars", action="store_true",
                        help="Do not show any progress bars")

    if "output_dir" not in omit:
        parser.add_argument("--output_dir", type=str, default="",
                            help="Default directory where to store checkpoints and models")
    if "output_model_prefix" not in omit:
        parser.add_argument("--output_model_prefix", type=str, default="model_",
                            help="Prefix for storing checkpoints and models")

    parser.add_argument("--bidi_dir", type=str, default=None, choices=["ltr", "rtl", "auto"],
                        help="The default text direction when preprocessing bidirectional text. Supported values "
                             "are 'auto' to automatically detect the direction, 'ltr' and 'rtl' for left-to-right and "
                             "right-to-left, respectively")

    if "weights" not in omit:
        parser.add_argument("--weights", type=str, default=None,
                            help="Load network weights from the given file.")

    parser.add_argument("--no_auto_compute_codec", action='store_true', default=False,
                        help="Do not compute the codec automatically. See also whitelist")
    parser.add_argument("--whitelist_files", type=str, nargs="+", default=[],
                        help="Whitelist of txt files that may not be removed on restoring a model")
    parser.add_argument("--whitelist", type=str, nargs="+", default=[],
                        help="Whitelist of characters that may not be removed on restoring a model. "
                             "For large datasets you can use this to skip the automatic codec computation "
                             "(see --no_auto_compute_codec)")
    parser.add_argument("--keep_loaded_codec", action='store_true', default=False,
                        help="Fully include the codec of the loaded model to the new codec")

    # clipping
    parser.add_argument("--gradient_clipping_norm", type=float, default=5,
                        help="Clipping constant of the norm of the gradients.")

    # early stopping
    if "validation" not in omit:
        parser.add_argument("--validation", type=str, nargs="+",
                            help="Validation line files used for early stopping")
        parser.add_argument("--validation_text_files", nargs="+", default=None,
                            help="Optional list of validation GT files if they are in other directory")
        parser.add_argument("--validation_extension", default=None,
                            help="Default extension of the gt files (expected to exist in same dir)")
        parser.add_argument("--validation_dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)

    parser.add_argument("--validation_data_on_the_fly", action='store_true', default=False,
                        help='Instead of preloading all data during the training, load the data on the fly. '
                             'This is slower, but might be required for limited RAM or large datasets')

    parser.add_argument("--early_stopping_frequency", type=int, default=1,
                        help="The frequency of early stopping [epochs].")
    parser.add_argument("--early_stopping_nbest", type=int, default=5,
                        help="The number of models that must be worse than the current best model to stop")
    if "early_stopping_best_model_prefix" not in omit:
        parser.add_argument("--early_stopping_best_model_prefix", type=str, default="best",
                            help="The prefix of the best model using early stopping")
    if "early_stopping_best_model_output_dir" not in omit:
        parser.add_argument("--early_stopping_best_model_output_dir", type=str, default=None,
                            help="Path where to store the best model. Default is output_dir")
    parser.add_argument("--n_augmentations", type=float, default=0,
                        help="Amount of data augmentation per line (done before training). If this number is < 1 "
                             "the amount is relative.")
    parser.add_argument("--only_train_on_augmented", action="store_true", default=False,
                        help="When training with augmentations usually the model is retrained in a second run with "
                             "only the non augmented data. This will take longer. Use this flag to disable this "
                             "behavior.")

    # backend specific params
    parser.add_argument("--num_inter_threads", type=int, default=0,
                        help="Tensorflow's session inter threads param")
    parser.add_argument("--num_intra_threads", type=int, default=0,
                        help="Tensorflow's session intra threads param")

    # text normalization/regularization
    parser.add_argument("--text_regularization", type=str, nargs="+", default=["extended"],
                        help="Text regularization to apply.")
    parser.add_argument("--text_normalization", type=str, default="NFC",
                        help="Unicode text normalization to apply. Defaults to NFC")
    # TODO: reactivate
    # parser.add_argument("--data_preprocessing", nargs="+", type=DataPreprocessors,
    #                    choices=list(DataPreprocessors), default=[DataPreprocessors.DefaultDataPreprocessor])

    # text/line generation params (loaded from json files)
    parser.add_argument("--text_generator_params", type=str, default=None)
    parser.add_argument("--line_generator_params", type=str, default=None)

    # additional dataset args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=0)

    parser.add_argument("--debug", action='store_true')


def create_train_dataset(args, dataset_args=None):
    gt_extension = args.gt_extension if args.gt_extension is not None else DataSetType.gt_extension(args.dataset)

    # Training dataset
    print("Resolving input files")
    input_image_files = sorted(glob_all(args.files))
    if not args.text_files:
        if gt_extension:
            gt_txt_files = [split_all_ext(f)[0] + gt_extension for f in input_image_files]
        else:
            gt_txt_files = [None] * len(input_image_files)
    else:
        gt_txt_files = sorted(glob_all(args.text_files))
        input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
        for img, gt in zip(input_image_files, gt_txt_files):
            if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = create_data_reader(
        args.dataset,
        DataSetMode.TRAIN,
        images=input_image_files,
        texts=gt_txt_files,
        skip_invalid=not args.no_skip_invalid_gt,
        args=dataset_args if dataset_args else {},
    )
    print("Found {} files in the dataset".format(len(dataset)))
    return dataset


def run(args):

    # check if loading a json file
    if len(args.files) == 1 and args.files[0].endswith("json"):
        with open(args.files[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                if key == 'dataset' or key == 'validation_dataset':
                    setattr(args, key, DataSetType.from_string(value))
                else:
                    setattr(args, key, value)

    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        setup_log(args.output_dir, append=False)

    # parse whitelist
    whitelist = args.whitelist
    if len(whitelist) == 1:
        whitelist = list(whitelist[0])

    whitelist_files = glob_all(args.whitelist_files)
    for f in whitelist_files:
        with open(f) as txt:
            whitelist += list(txt.read())

    if args.gt_extension is None:
        args.gt_extension = DataSetType.gt_extension(args.dataset)

    if args.validation_extension is None:
        args.validation_extension = DataSetType.gt_extension(args.validation_dataset)

    if args.text_generator_params is not None:
        with open(args.text_generator_params, 'r') as f:
            args.text_generator_params = TextGeneratorParams.from_json(f.read())
    else:
        args.text_generator_params = TextGeneratorParams()

    if args.line_generator_params is not None:
        with open(args.line_generator_params, 'r') as f:
            args.line_generator_params = LineGeneratorParams.from_json(f.read())
    else:
        args.line_generator_params = LineGeneratorParams()

    dataset_args = FileDataReaderArgs(
        line_generator_params=args.line_generator_params,
        text_generator_params=args.text_generator_params,
        pad=args.dataset_pad,
        text_index=args.pagexml_text_index,
    )

    params: TrainerParams = CalamariScenario.trainer_cls().get_params_cls()()
    params.scenario_params = CalamariScenario.default_params()

    # =================================================================================================================
    # Data Params

    data_params: CalamariDataParams = params.scenario_params.data_params
    data_params.train_reader = FileDataReaderFactory(args.dataset, DataSetMode.TRAIN,
                                                     args.files, args.text_files,
                                                     not args.no_skip_invalid_gt, args.gt_extension, dataset_args)
    if args.validation:
        data_params.val_reader = FileDataReaderFactory(args.validation_dataset, DataSetMode.PRED_AND_EVAL,
                                                       args.validation, args.validation_text_files,
                                                       not args.no_skip_invalid_gt, args.validation_extension, dataset_args
                                                       )
    else:
        data_params.val_reader = data_params.train_reader
    data_params.data_processor = DefaultDataPreprocessor(args.line_height, args.pad)

    # Text pre processing (reading)
    data_params.text_processor = MultiTextProcessor([
        TextNormalizer(args.text_normalization),
        TextRegularizer(default_text_regularizer_replacements(args.text_regularization)),
        StripTextProcessor(),
    ])

    # Text post processing (prediction)
    if False:
        # TODO: text post processing
        data_params.text_post_processor = MultiTextProcessor([
            TextNormalizer(args.text_normalization),
            TextRegularizer(default_text_regularizer_replacements(args.text_regularization)),
            StripTextProcessor(),
        ])

    # =================================================================================================================
    # TODO: ORDER
    params.device_params.gpus = list(map(int, filter(lambda x: len(x) > 0, os.environ.get("CUDA_VISIBLE_DEVICES", '').split(','))))
    params.force_eager = args.debug
    params.scenario_params.debug_graph_construction = args.debug
    params.epochs = args.epochs
    params.samples_per_epoch = args.samples_per_epoch
    params.stats_size = args.stats_size
    params.skip_load_model_test = True
    params.scenario_params.export_frozen = False
    params.scenario_params.data_params.train_batch_size = args.batch_size
    params.scenario_params.data_params.val_batch_size = args.batch_size
    # TODO: params.checkpoint_frequency = args.checkpoint_frequency if args.checkpoint_frequency >= 0 else args.early_stopping_frequency
    params.checkpoint_dir = args.output_dir
    params.test_every_n = args.display
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.scenario_params.data_params.train_num_processes = args.num_threads
    params.scenario_params.data_params.val_num_processes = args.num_threads
    params.data_aug_retrain_on_original = not args.only_train_on_augmented

    params.early_stopping_params.frequency = args.early_stopping_frequency
    params.early_stopping_params.upper_threshold = 0.9
    params.early_stopping_params.lower_threshold = 1.0 - args.early_stopping_at_accuracy
    params.early_stopping_params.n_to_go = args.early_stopping_nbest
    params.early_stopping_params.best_model_name = args.early_stopping_best_model_prefix
    params.early_stopping_params.best_model_output_dir = args.early_stopping_best_model_output_dir
    if args.seed > 0:
        params.random_seed = args.seed

    if args.bidi_dir:
        params.scenario_params.data_params.text_processor.sub_processors.append(BidiTextProcessor(args.bidi_dir))
        params.scenario_params.data_params.text_post_processor.sub_processors.append(BidiTextProcessor(args.bidi_dir))

    params.scenario_params.data_params.line_height_ = args.line_height

    params_from_definition_string(args.network, params)
    params.optimizer_params.clip_grad = args.gradient_clipping_norm
    # params.model.network.backend.num_inter_threads = args.num_inter_threads
    # params.model.network.backend.num_intra_threads = args.num_intra_threads
    params.codec_whitelist = whitelist
    params.keep_loaded_codec = args.keep_loaded_codec
    params.preload_training = not args.train_data_on_the_fly
    params.preload_validation = not args.validation_data_on_the_fly
    params.warmstart_params.model = args.weights

    params.scenario_params.data_params.data_augmenter = SimpleDataAugmenter()
    params.scenario_params.data_params.data_aug_params = DataAugmentationAmount.from_factor(args.n_augmentations)
    params.auto_compute_codec = not args.no_auto_compute_codec
    params.progress_bar = not args.no_progress_bars

    scenario = CalamariScenario(params.scenario_params)
    trainer = scenario.create_trainer(params)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    setup_train_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
