import matplotlib.pyplot as plt
import argparse
from calamari_ocr.ocr.datasets import create_dataset, DataSetType, DataSetMode
from calamari_ocr.ocr.datasets.input_dataset import StreamingInputDataset
from calamari_ocr import __version__
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.proto import DataPreprocessorParams, TextProcessorParams
from calamari_ocr.ocr.text_processing import \
    default_text_normalizer_params, default_text_regularizer_params
import os
from calamari_ocr.ocr.augmentation.data_augmenter import SimpleDataAugmenter


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
    parser.add_argument("--data_preprocessing", nargs="+", type=DataPreprocessorParams.Type.Value,
                        choices=DataPreprocessorParams.Type.values(), default=[DataPreprocessorParams.DEFAULT_NORMALIZER])

    args = parser.parse_args()

    # Text/Data processing
    if args.data_preprocessing is None or len(args.data_preprocessing) == 0:
        args.data_preprocessing = [DataPreprocessorParams.DEFAULT_NORMALIZER]

    data_preprocessor = DataPreprocessorParams()
    data_preprocessor.type = DataPreprocessorParams.MULTI_NORMALIZER
    for preproc in args.data_preprocessing:
        pp = data_preprocessor.children.add()
        pp.type = preproc
        pp.line_height = args.line_height
        pp.pad = args.pad

    # Text pre processing (reading)
    text_preprocessor = TextProcessorParams()
    text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(text_preprocessor.children.add(), default=args.text_normalization)
    default_text_regularizer_params(text_preprocessor.children.add(), groups=args.text_regularization)
    strip_processor_params = text_preprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    text_preprocessor = text_processor_from_proto(text_preprocessor)
    data_preprocessor = data_processor_from_proto(data_preprocessor)

    print("Resolving input files")
    input_image_files = sorted(glob_all(args.files))
    if not args.text_files:
        if args.gt_extension:
            gt_txt_files = [split_all_ext(f)[0] + args.gt_extension for f in input_image_files]
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
        non_existing_as_empty=True,
    )

    if len(args.select) == 0:
        args.select = range(len(dataset.samples()))
        dataset._samples = dataset.samples()
    else:
        dataset._samples = [dataset.samples()[i] for i in args.select]

    samples = dataset.samples()

    print("Found {} files in the dataset".format(len(dataset)))

    with StreamingInputDataset(dataset,
                               data_preprocessor,
                               text_preprocessor,
                               SimpleDataAugmenter(),
                               args.n_augmentations,
                               ) as input_dataset:
        f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
        row, col = 0, 0
        for i, (id, sample) in enumerate(zip(args.select, input_dataset.generator(args.processes))):
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

            if col == args.n_cols or i == len(samples) - 1:
                plt.show()
                f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
                row, col = 0, 0


if __name__ == "__main__":
    main()
