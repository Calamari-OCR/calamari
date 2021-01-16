import tfaip.util.logging
import argparse
import os
import json

from tfaip.base.data.pipeline.datapipeline import SamplePipelineParams
from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, INPUT_PROCESSOR, \
    TARGETS_PROCESSOR, PipelineMode
from tfaip.util.logging import setup_log

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount

from calamari_ocr import __version__
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset import TextGeneratorParams, LineGeneratorParams
from calamari_ocr.ocr.dataset.params import DataParams, PipelineParams
from calamari_ocr.ocr.dataset.datareader.factory import FileDataReaderArgs
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSampleProcessor
from calamari_ocr.ocr.dataset.postprocessors.ctcdecoder import CTCDecoderProcessor
from calamari_ocr.ocr.dataset.postprocessors.reshape import ReshapeOutputsProcessor
from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import default_image_processors
from calamari_ocr.ocr.dataset.textprocessors import TextNormalizer, TextRegularizer, StripTextProcessor, \
    BidiTextProcessor
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import default_text_regularizer_replacements
from calamari_ocr.ocr.training.params import params_from_definition_string, TrainerParams
from calamari_ocr.utils import glob_all
from calamari_ocr.ocr.dataset import DataSetType


logger = tfaip.util.logging.logger(__name__)


def setup_train_args(parser, omit=None):
    # required params for args
    from calamari_ocr.ocr.dataset.data import Data

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
                             'This is slower, but might be required for limited RAM or large dataset')

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
    parser.add_argument("--checkpoint_frequency", type=int, default=0,
                        help="The frequency how often to write checkpoints during training [epochs]"
                             "If -1, the early_stopping_frequency will be used. default (0) no checkpoints are written")
    parser.add_argument("--epochs", type=int, default=100,
                        help="The number of iterations for training. "
                             "If using early stopping, this is the maximum number of iterations")
    parser.add_argument("--samples_per_epoch", type=float, default=-1,
                        help="The number of samples to process per epoch. By default the size of the training dataset."
                             "If in (0,1) it is relative to the dataset size"
                        )
    parser.add_argument("--early_stopping_at_accuracy", type=float, default=1.0,
                        help="Stop training if the early stopping accuracy reaches this value")
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
                             "For large dataset you can use this to skip the automatic codec computation "
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
                             'This is slower, but might be required for limited RAM or large dataset')

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

    # text normalization/regularization
    parser.add_argument("--text_regularization", type=str, nargs="+", default=["extended"],
                        help="Text regularization to apply.")
    parser.add_argument("--text_normalization", type=str, default="NFC",
                        help="Unicode text normalization to apply. Defaults to NFC")
    parser.add_argument("--data_preprocessing", nargs="+", type=str,
                        choices=[k for k, p in Data.data_processor_factory().processors.items() if issubclass(p, ImageProcessor)],
                        default=[p.name for p in default_image_processors()])

    # text/line generation params (loaded from json files)
    parser.add_argument("--text_generator_params", type=str, default=None)
    parser.add_argument("--line_generator_params", type=str, default=None)

    # additional dataset args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=0)

    parser.add_argument("--debug", action='store_true')


def run(args):
    # local imports (to prevent tensorflow from being imported to early)
    from calamari_ocr.ocr.scenario import Scenario
    from calamari_ocr.ocr.dataset.data import Data

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

    if args.dataset == DataSetType.GENERATED_LINE or args.validation_dataset == DataSetType.GENERATED_LINE:
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

    params: TrainerParams = Scenario.default_trainer_params()

    # =================================================================================================================
    # Data Params
    data_params: DataParams = params.scenario_params.data_params
    data_params.train = PipelineParams(
        type=args.dataset,
        skip_invalid=not args.no_skip_invalid_gt,
        remove_invalid=True,
        files=args.files,
        text_files=args.text_files,
        gt_extension=args.gt_extension,
        data_reader_args=dataset_args,
        batch_size=args.batch_size,
        num_processes=args.num_threads,
    )
    if args.validation:
        data_params.val = PipelineParams(
            type=args.validation_dataset,
            files=args.validation,
            text_files=args.validation_text_files,
            skip_invalid=not args.no_skip_invalid_gt,
            gt_extension=args.validation_extension,
            data_reader_args=dataset_args,
            batch_size=args.batch_size,
            num_processes=args.num_threads,
        )
    else:
        data_params.val = None

    data_params.pre_processors_ = SamplePipelineParams(run_parallel=True)
    data_params.post_processors_.run_parallel = SamplePipelineParams(
        run_parallel=False, sample_processors=[
            DataProcessorFactoryParams(ReshapeOutputsProcessor.__name__),
            DataProcessorFactoryParams(CTCDecoderProcessor.__name__),
        ])

    for p in args.data_preprocessing:
        p_p = Data.data_processor_factory().processors[p].default_params()
        if 'pad' in p_p:
            p_p['pad'] = args.pad
        data_params.pre_processors_.sample_processors.append(DataProcessorFactoryParams(p, INPUT_PROCESSOR, p_p))

    # Text pre processing (reading)
    data_params.pre_processors_.sample_processors.extend(
        [
            DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR, {'unicode_normalization': args.text_normalization}),
            DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR, {'replacements': default_text_regularizer_replacements(args.text_regularization)}),
            DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR)
        ])

    # Text post processing (prediction)
    data_params.post_processors_.sample_processors.extend(
        [
            DataProcessorFactoryParams(TextNormalizer.__name__, TARGETS_PROCESSOR,
                                       {'unicode_normalization': args.text_normalization}),
            DataProcessorFactoryParams(TextRegularizer.__name__, TARGETS_PROCESSOR,
                                       {'replacements': default_text_regularizer_replacements(args.text_regularization)}),
            DataProcessorFactoryParams(StripTextProcessor.__name__, TARGETS_PROCESSOR)
        ])
    if args.bidi_dir:
        data_params.pre_processors_.sample_processors.append(
            DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR, {'bidi_direction': args.bidi_dir})
        )
        data_params.post_processors_.sample_processors.append(
            DataProcessorFactoryParams(BidiTextProcessor.__name__, TARGETS_PROCESSOR, {'bidi_direction': args.bidi_dir})
        )

    data_params.pre_processors_.sample_processors.extend([
        DataProcessorFactoryParams(AugmentationProcessor.__name__, {PipelineMode.Training}, {'augmenter_type': 'simple'}),
        DataProcessorFactoryParams(PrepareSampleProcessor.__name__),
    ])

    data_params.data_aug_params = DataAugmentationAmount.from_factor(args.n_augmentations)
    data_params.line_height_ = args.line_height

    # =================================================================================================================
    # Trainer Params
    params.force_eager = args.debug
    params.skip_model_load_test = not args.debug
    params.scenario_params.debug_graph_construction = args.debug
    params.epochs = args.epochs
    params.samples_per_epoch = int(args.samples_per_epoch) if args.samples_per_epoch >= 1 else -1
    params.scale_epoch_size = abs(args.samples_per_epoch) if args.samples_per_epoch < 1 else 1
    params.skip_load_model_test = True
    params.scenario_params.export_frozen = False
    params.checkpoint_save_freq_ = args.checkpoint_frequency if args.checkpoint_frequency >= 0 else args.early_stopping_frequency
    params.checkpoint_dir = args.output_dir
    params.test_every_n = args.display
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.data_aug_retrain_on_original = not args.only_train_on_augmented
    if args.seed > 0:
        params.random_seed = args.seed

    params.optimizer_params.clip_grad = args.gradient_clipping_norm
    params.codec_whitelist = whitelist
    params.keep_loaded_codec = args.keep_loaded_codec
    params.preload_training = not args.train_data_on_the_fly
    params.preload_validation = not args.validation_data_on_the_fly
    params.warmstart_params.model = args.weights

    params.auto_compute_codec = not args.no_auto_compute_codec
    params.progress_bar = not args.no_progress_bars

    params.early_stopping_params.frequency = args.early_stopping_frequency
    params.early_stopping_params.upper_threshold = 0.9
    params.early_stopping_params.lower_threshold = 1.0 - args.early_stopping_at_accuracy
    params.early_stopping_params.n_to_go = args.early_stopping_nbest
    params.early_stopping_params.best_model_name = ''
    params.early_stopping_params.best_model_output_dir = args.early_stopping_best_model_output_dir
    params.scenario_params.default_serve_dir_ = f'{args.early_stopping_best_model_prefix}.ckpt.h5'
    params.scenario_params.trainer_params_filename_ = f'{args.early_stopping_best_model_prefix}.ckpt.json'

    params_from_definition_string(args.network, params)

    scenario = Scenario(params.scenario_params)
    trainer = scenario.create_trainer(params)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    setup_train_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
