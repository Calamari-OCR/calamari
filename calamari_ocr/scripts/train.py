import argparse

from calamari_ocr.utils.glob import glob_all
from calamari_ocr.utils.path import split_all_ext
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr.trainer import Trainer
from calamari_ocr.ocr.data_processing.default_data_preprocessor import DefaultDataPreprocessor
from calamari_ocr.ocr.text_processing import DefaultTextPreprocessor, text_processor_from_proto, BidiTextProcessor

from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, \
    network_params_from_definition_string


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs="+",
                        help="List all image files that shall be processed. Ground truth fils with the same "
                             "base name but with '.gt.txt' as extension are required at the same location")
    parser.add_argument("--backend", type=str, default="tensorflow",
                        help="The backend to use for the neural net computation. Currently supported only tensorflow")
    parser.add_argument("--network", type=str, default="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
                        help="The network structure")
    parser.add_argument("--line_height", type=int, default=48,
                        help="The line height")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="The number of threads to use for all operations")
    parser.add_argument("--display", type=int, default=1,
                        help="Frequency of how often an output shall occur during training")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size to use for training")
    parser.add_argument("--checkpoint_frequency", type=int, default=1000,
                        help="The frequency how often to write checkpoints during training")
    parser.add_argument("--max_iters", type=int, default=1000000,
                        help="The number of iterations for training. "
                             "If using early stopping, this is the maximum number of iterations")
    parser.add_argument("--stats_size", type=int, default=100,
                        help="Average this many iterations for computing an average loss, label error rate and "
                             "training time")
    parser.add_argument("--no_skip_invalid_gt", action="store_true", default=False,
                        help="Do no skip invalid gt, instead raise an exception.")
    parser.add_argument("--output_path_prefix", type=str, default="model_",
                        help="Prefix path for storing checkpoints and models")
    parser.add_argument("--bidi_dir", type=str, default=None,
                        help="The default direction of text. Defaults to ltr='left to right'. The other option is 'rtl'")

    args = parser.parse_args()

    input_image_files = glob_all(args.files)
    gt_txt_files = [split_all_ext(f)[0] + ".gt.txt" for f in input_image_files]

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = FileDataSet(input_image_files, gt_txt_files)

    print("Found {} files in the dataset".format(len(dataset)))

    params = CheckpointParams()

    params.max_iters = args.max_iters
    params.stats_size = args.stats_size
    params.batch_size = args.batch_size
    params.checkpoint_frequency = args.checkpoint_frequency
    params.output_path_prefix = args.output_path_prefix
    params.display = args.display
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.processes = args.num_threads

    params.model.data_preprocessor.type = DataPreprocessorParams.DEFAULT_NORMALIZER
    params.model.data_preprocessor.line_height = args.line_height
    params.model.text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    strip_processor_params = params.model.text_preprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER
    params.model.text_postprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    strip_processor_params = params.model.text_postprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    if args.bidi_dir:
        # change bidirectional text direction if desired
        bidi_dir_to_enum = {"rtl": TextProcessorParams.BIDI_RTL, "ltr": TextProcessorParams.BIDI_LTR}

        bidi_processor_params = params.model.text_preprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = bidi_dir_to_enum[args.bidi_dir]

        bidi_processor_params = params.model.text_postprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = bidi_dir_to_enum[args.bidi_dir]

    params.model.line_height = args.line_height

    network_params_from_definition_string(args.network, params.model.network)

    # create the actual trainer
    trainer = Trainer(params,
                      dataset,
                      )
    trainer.train(progress_bar=True)


if __name__ == "__main__":
    main()
