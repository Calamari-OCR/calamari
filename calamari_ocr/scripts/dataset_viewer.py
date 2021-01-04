import matplotlib.pyplot as plt
import argparse

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from tfaip.base.data.pipeline.datapipeline import SamplePipelineParams
from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, INPUT_PROCESSOR, \
    TARGETS_PROCESSOR, PipelineMode

from calamari_ocr.ocr.dataset import DataSetType

from calamari_ocr import __version__
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.base import DataReader
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor, PrepareSampleProcessor
from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import default_image_processors
from calamari_ocr.ocr.dataset.params import FileDataReaderArgs, PipelineParams, DataParams
from calamari_ocr.ocr.dataset.textprocessors import TextNormalizer, TextRegularizer, StripTextProcessor, \
    BidiTextProcessor
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import default_text_regularizer_replacements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)
    parser.add_argument("--files", nargs="+",
                        help="List all image files that shall be processed. Ground truth fils with the same "
                             "base name but with '.gt.txt' as extension are required at the same location",
                        required=True)
    parser.add_argument("--text_files", nargs="+", default=None,
                        help="Optional list of GT files if they are in other directory")
    parser.add_argument("--gt_extension", default=None,
                        help="Default extension of the gt files (expected to exist in same dir)")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--line_height", type=int, default=48,
                        help="The line height")
    parser.add_argument("--pad", type=int, default=16,
                        help="Padding (left right) of the line")
    parser.add_argument("--processes", type=int, default=1,
                        help="The number of threads to use for all operations")

    parser.add_argument("--n_cols", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=5)
    parser.add_argument("--select", type=int, nargs="+", default=[])

    # text normalization/regularization
    parser.add_argument("--n_augmentations", type=float, default=0,
                        help="Amount of data augmentation per line (done before training). If this number is < 1 "
                             "the amount is relative.")
    parser.add_argument("--text_regularization", type=str, nargs="+", default=["extended"],
                        help="Text regularization to apply.")
    parser.add_argument("--text_normalization", type=str, default="NFC",
                        help="Unicode text normalization to apply. Defaults to NFC")
    parser.add_argument("--data_preprocessing", nargs="+", type=str,
                        choices=[k for k, p in Data.data_processor_factory().processors.items() if issubclass(p, ImageProcessor)],
                        default=[p.name for p in default_image_processors()])
    parser.add_argument("--bidi_dir", type=str, default=None, choices=["ltr", "rtl", "auto"],
                        help="The default text direction when preprocessing bidirectional text. Supported values "
                             "are 'auto' to automatically detect the direction, 'ltr' and 'rtl' for left-to-right and "
                             "right-to-left, respectively")

    parser.add_argument("--preload", action='store_true', help='Simulate preloading')
    parser.add_argument("--as_validation", action='store_true', help="Access as validation instead of training data.")

    args = parser.parse_args()

    if args.gt_extension is None:
        args.gt_extension = DataSetType.gt_extension(args.dataset)

    dataset_args = FileDataReaderArgs(
        pad=args.pad,
    )

    data_params: DataParams = Data.get_default_params()
    data_params.train = PipelineParams(
        type=args.dataset,
        remove_invalid=True,
        files=args.files,
        text_files=args.text_files,
        gt_extension=args.gt_extension,
        data_reader_args=dataset_args,
        batch_size=1,
        num_processes=args.processes,
    )
    data_params.val = data_params.train

    data_params.pre_processors_ = SamplePipelineParams(run_parallel=True)
    data_params.post_processors_ = SamplePipelineParams(run_parallel=True)
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
        # DataProcessorFactoryParams(PrepareSampleProcessor.__name__),  # NOT THIS, since, we want to access raw input
    ])

    data_params.data_aug_params = DataAugmentationAmount.from_factor(args.n_augmentations)
    data_params.line_height_ = args.line_height

    data = Data(data_params)
    data_pipeline = data.get_val_data() if args.as_validation else data.get_train_data()
    if not args.preload:
        reader: DataReader = data_pipeline.reader()
        if len(args.select) == 0:
            args.select = range(len(reader))
        else:
            reader._samples = [reader.samples()[i] for i in args.select]
    else:
        data.preload()
        data_pipeline = data.get_val_data() if args.as_validation else data.get_train_data()
        samples = data_pipeline.samples
        if len(args.select) == 0:
            args.select = range(len(samples))
        else:
            data_pipeline.samples = [samples[i] for i in args.select]

    f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
    row, col = 0, 0
    with data_pipeline as dataset:
        for i, (id, sample) in enumerate(zip(args.select, dataset.generate_input_samples(auto_repeat=False))):
            line, text, params = sample
            if args.n_cols == 1:
                ax[row].imshow(line.transpose())
                ax[row].set_title("ID: {}\n{}".format(id, text))
            else:
                ax[row, col].imshow(line.transpose())
                ax[row, col].set_title("ID: {}\n{}".format(id, text))

            row += 1
            if row == args.n_rows:
                row = 0
                col += 1

            if col == args.n_cols or i == len(dataset) - 1:
                plt.show()
                f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
                row, col = 0, 0


if __name__ == "__main__":
    main()
