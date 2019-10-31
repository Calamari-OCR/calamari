import argparse
import os

from calamari_ocr import __version__
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
from calamari_ocr.ocr.datasets import create_dataset, DataSetType, DataSetMode
from calamari_ocr.ocr.augmentation.data_augmenter import SimpleDataAugmenter
from calamari_ocr.ocr import Trainer
from calamari_ocr.ocr.text_processing import \
    default_text_normalizer_params, default_text_regularizer_params

from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, \
    network_params_from_definition_string, NetworkParams, TextGeneratorParameters, LineGeneratorParameters

from google.protobuf import json_format


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
    parser.add_argument("--display", type=float, default=100,
                        help="Frequency of how often an output shall occur during training. If 0 < display <= 1 "
                             "the display is in units of epochs.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size to use for training")
    parser.add_argument("--checkpoint_frequency", type=float, default=-1,
                        help="The frequency how often to write checkpoints during training. If 0 < value <= 1 the "
                             "unit is in epochs, thus relative to the number of training examples."
                             "If -1, the early_stopping_frequency will be used.")
    parser.add_argument("--max_iters", type=int, default=1000000,
                        help="The number of iterations for training. "
                             "If using early stopping, this is the maximum number of iterations")
    parser.add_argument("--early_stopping_at_accuracy", type=float, default=0,
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

    parser.add_argument("--early_stopping_frequency", type=float, default=-1,
                        help="The frequency of early stopping. By default the checkpoint frequency uses the early "
                             "stopping frequency. By default (negative value) the early stopping frequency is "
                             "approximated as a half epoch time (if the batch size is 1). "
                             "If 0 < value <= 1 the frequency has the unit of an epoch (relative to the "
                             "number of training data).")
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
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000,
                        help="Number of examples in the shuffle buffer for training (default 1000). A higher number "
                             "required more memory. If set to 0, the buffer size equates an epoch i.e. the full dataset")

    # text normalization/regularization
    parser.add_argument("--text_regularization", type=str, nargs="+", default=["extended"],
                        help="Text regularization to apply.")
    parser.add_argument("--text_normalization", type=str, default="NFC",
                        help="Unicode text normalization to apply. Defaults to NFC")
    parser.add_argument("--data_preprocessing", nargs="+", type=DataPreprocessorParams.Type.Value,
                        choices=DataPreprocessorParams.Type.values(), default=[DataPreprocessorParams.DEFAULT_NORMALIZER])

    # text/line generation params (loaded from json files)
    parser.add_argument("--text_generator_params", type=str, default=None)
    parser.add_argument("--line_generator_params", type=str, default=None)


    # additional dataset args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=0)



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

    dataset = create_dataset(
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
        import json
        with open(args.files[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                if key == 'dataset' or key == 'validation_dataset':
                    setattr(args, key, DataSetType.from_string(value))
                else:
                    setattr(args, key, value)

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
            args.text_generator_params = json_format.Parse(f.read(), TextGeneratorParameters())
    else:
        args.text_generator_params = TextGeneratorParameters()

    if args.line_generator_params is not None:
        with open(args.line_generator_params, 'r') as f:
            args.line_generator_params = json_format.Parse(f.read(), LineGeneratorParameters())
    else:
        args.line_generator_params = LineGeneratorParameters()

    dataset_args = {
        'line_generator_params': args.line_generator_params,
        'text_generator_params': args.text_generator_params,
        'pad': args.dataset_pad,
        'text_index': args.pagexml_text_index,
    }

    # Training dataset
    dataset = create_train_dataset(args, dataset_args)

    # Validation dataset
    if args.validation:
        print("Resolving validation files")
        validation_image_files = glob_all(args.validation)
        if not args.validation_text_files:
            val_txt_files = [split_all_ext(f)[0] + args.validation_extension for f in validation_image_files]
        else:
            val_txt_files = sorted(glob_all(args.validation_text_files))
            validation_image_files, val_txt_files = keep_files_with_same_file_name(validation_image_files, val_txt_files)
            for img, gt in zip(validation_image_files, val_txt_files):
                if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                    raise Exception("Expected identical basenames of validation file: {} and {}".format(img, gt))

        if len(set(val_txt_files)) != len(val_txt_files):
            raise Exception("Some validation images are occurring more than once in the data set.")

        validation_dataset = create_dataset(
            args.validation_dataset,
            DataSetMode.TRAIN,
            images=validation_image_files,
            texts=val_txt_files,
            skip_invalid=not args.no_skip_invalid_gt,
            args=dataset_args,
        )
        print("Found {} files in the validation dataset".format(len(validation_dataset)))
    else:
        validation_dataset = None

    params = CheckpointParams()

    params.max_iters = args.max_iters
    params.stats_size = args.stats_size
    params.batch_size = args.batch_size
    params.checkpoint_frequency = args.checkpoint_frequency if args.checkpoint_frequency >= 0 else args.early_stopping_frequency
    params.output_dir = args.output_dir
    params.output_model_prefix = args.output_model_prefix
    params.display = args.display
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.processes = args.num_threads
    params.data_aug_retrain_on_original = not args.only_train_on_augmented

    params.early_stopping_at_acc = args.early_stopping_at_accuracy
    params.early_stopping_frequency = args.early_stopping_frequency
    params.early_stopping_nbest = args.early_stopping_nbest
    params.early_stopping_best_model_prefix = args.early_stopping_best_model_prefix
    params.early_stopping_best_model_output_dir = \
        args.early_stopping_best_model_output_dir if args.early_stopping_best_model_output_dir else args.output_dir

    if args.data_preprocessing is None or len(args.data_preprocessing) == 0:
        args.data_preprocessing = [DataPreprocessorParams.DEFAULT_NORMALIZER]

    params.model.data_preprocessor.type = DataPreprocessorParams.MULTI_NORMALIZER
    for preproc in args.data_preprocessing:
        pp = params.model.data_preprocessor.children.add()
        pp.type = DataPreprocessorParams.Type.Value(preproc) if isinstance(preproc, str) else preproc
        pp.line_height = args.line_height
        pp.pad = args.pad

    # Text pre processing (reading)
    params.model.text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_preprocessor.children.add(), default=args.text_normalization)
    default_text_regularizer_params(params.model.text_preprocessor.children.add(), groups=args.text_regularization)
    strip_processor_params = params.model.text_preprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    # Text post processing (prediction)
    params.model.text_postprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_postprocessor.children.add(), default=args.text_normalization)
    default_text_regularizer_params(params.model.text_postprocessor.children.add(), groups=args.text_regularization)
    strip_processor_params = params.model.text_postprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    if args.seed > 0:
        params.model.network.backend.random_seed = args.seed

    if args.bidi_dir:
        # change bidirectional text direction if desired
        bidi_dir_to_enum = {"rtl": TextProcessorParams.BIDI_RTL, "ltr": TextProcessorParams.BIDI_LTR,
                            "auto": TextProcessorParams.BIDI_AUTO}

        bidi_processor_params = params.model.text_preprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = bidi_dir_to_enum[args.bidi_dir]

        bidi_processor_params = params.model.text_postprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = TextProcessorParams.BIDI_AUTO

    params.model.line_height = args.line_height

    network_params_from_definition_string(args.network, params.model.network)
    params.model.network.clipping_norm = args.gradient_clipping_norm
    params.model.network.backend.num_inter_threads = args.num_inter_threads
    params.model.network.backend.num_intra_threads = args.num_intra_threads
    params.model.network.backend.shuffle_buffer_size = args.shuffle_buffer_size

    # create the actual trainer
    trainer = Trainer(params,
                      dataset,
                      validation_dataset=validation_dataset,
                      data_augmenter=SimpleDataAugmenter(),
                      n_augmentations=args.n_augmentations,
                      weights=args.weights,
                      codec_whitelist=whitelist,
                      keep_loaded_codec=args.keep_loaded_codec,
                      preload_training=not args.train_data_on_the_fly,
                      preload_validation=not args.validation_data_on_the_fly,
                      )
    trainer.train(
        auto_compute_codec=not args.no_auto_compute_codec,
        progress_bar=not args.no_progress_bars
    )


def main():
    parser = argparse.ArgumentParser()
    setup_train_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
