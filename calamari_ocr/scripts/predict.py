import argparse
import os

from bidi.algorithm import get_base_level

from google.protobuf.json_format import MessageToJson

from calamari_ocr import __version__
from calamari_ocr.utils.glob import glob_all
from calamari_ocr.ocr.datasets import DataSetType, create_dataset, DataSetMode
from calamari_ocr.ocr import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams, Predictions, CTCDecoderParams


def create_ctc_decoder_params(args):
    params = CTCDecoderParams()
    params.beam_width = args.beam_width

    if args.dictionary and len(args.dictionary) > 0:
        dictionary = set()
        print("Creating dictionary")
        for path in glob_all(args.dictionary):
            with open(path, 'r') as f:
                dictionary = dictionary.union({word for word in f.read().split()})

        params.dictionary[:] = dictionary
        print("Dictionary with {} unique words successfully created.".format(len(dictionary)))
    else:
        args.dictionary = None

    if args.dictionary:
        print("USING A LANGUAGE MODEL IS CURRENTLY EXPERIMENTAL ONLY. NOTE: THE PREDICTION IS VERY SLOW!")
        params.type = CTCDecoderParams.CTC_WORD_BEAM_SEARCH

    return params


def run(args):
    # check if loading a json file
    if len(args.files) == 1 and args.files[0].endswith("json"):
        import json
        with open(args.files[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                setattr(args, key, value)

    # checks
    if args.extended_prediction_data_format not in ["pred", "json"]:
        raise Exception("Only 'pred' and 'json' are allowed extended prediction data formats")

    # add json as extension, resolve wildcard, expand user, ... and remove .json again
    args.checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in args.checkpoint]
    args.checkpoint = glob_all(args.checkpoint)
    args.checkpoint = [cp[:-5] for cp in args.checkpoint]

    args.extension = args.extension if args.extension else DataSetType.pred_extension(args.dataset)

    # create ctc decoder
    ctc_decoder_params = create_ctc_decoder_params(args)

    # create voter
    voter_params = VoterParams()
    voter_params.type = VoterParams.Type.Value(args.voter.upper())
    voter = voter_from_proto(voter_params)

    # load files
    input_image_files = glob_all(args.files)
    if args.text_files:
        args.text_files = glob_all(args.text_files)

    # skip invalid files and remove them, there wont be predictions of invalid files
    dataset = create_dataset(
        args.dataset,
        DataSetMode.PREDICT,
        input_image_files,
        args.text_files,
        skip_invalid=True,
        remove_invalid=True,
        args={
            'text_index': args.pagexml_text_index,
            'pad': args.dataset_pad,
        },
    )

    print("Found {} files in the dataset".format(len(dataset)))
    if len(dataset) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    # predict for all models
    predictor = MultiPredictor(checkpoints=args.checkpoint, batch_size=args.batch_size, processes=args.processes,
                               ctc_decoder_params=ctc_decoder_params)
    do_prediction = predictor.predict_dataset(dataset, progress_bar=not args.no_progress_bars)

    avg_sentence_confidence = 0
    n_predictions = 0

    # output the voted results to the appropriate files
    for result, sample in do_prediction:
        n_predictions += 1
        for i, p in enumerate(result):
            p.prediction.id = "fold_{}".format(i)

        # vote the results (if only one model is given, this will just return the sentences)
        prediction = voter.vote_prediction_result(result)
        prediction.id = "voted"
        sentence = prediction.sentence
        avg_sentence_confidence += prediction.avg_char_probability
        if args.verbose:
            lr = "\u202A\u202B"
            print("{}: '{}{}{}'".format(sample['id'], lr[get_base_level(sentence)], sentence, "\u202C" ))

        output_dir = args.output_dir

        dataset.store_text(sentence, sample, output_dir=output_dir, extension=args.extension)

        if args.extended_prediction_data:
            ps = Predictions()
            ps.line_path = sample['image_path'] if 'image_path' in sample else sample['id']
            ps.predictions.extend([prediction] + [r.prediction for r in result])
            output_dir = output_dir if output_dir else os.path.dirname(ps.line_path)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if args.extended_prediction_data_format == "pred":
                data = ps.SerializeToString()
            elif args.extended_prediction_data_format == "json":
                # remove logits
                for prediction in ps.predictions:
                    prediction.logits.rows = 0
                    prediction.logits.cols = 0
                    prediction.logits.data[:] = []

                data = MessageToJson(ps, including_default_value_fields=True)
            else:
                raise Exception("Unknown prediction format.")


            dataset.store_extended_prediction(data, sample, output_dir=output_dir, extension=args.extended_prediction_data_format)

    print("Average sentence confidence: {:.2%}".format(avg_sentence_confidence / n_predictions))

    dataset.store(args.extension)
    print("All files written")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    parser.add_argument("--files", nargs="+", required=True, default=[],
                        help="List all image files that shall be processed")
    parser.add_argument("--text_files", nargs="+", default=None,
                        help="Optional list of additional text files. E.g. when updating Abbyy prediction, this parameter must be used for the xml files.")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--extension", type=str, default=None,
                        help="Define the prediction extension. This parameter can be used to override ground truth files.")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[],
                        help="Path to the checkpoint without file extension")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size during the prediction (number of lines to process in parallel)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--voter", type=str, default="confidence_voter_default_ctc",
                        help="The voting algorithm to use. Possible values: confidence_voter_default_ctc (default), "
                             "confidence_voter_fuzzy_ctc, sequence_voter")
    parser.add_argument("--output_dir", type=str,
                        help="By default the prediction files will be written to the same directory as the given files. "
                             "You can use this argument to specify a specific output dir for the prediction files.")
    parser.add_argument("--extended_prediction_data", action="store_true",
                        help="Write: Predicted string, labels; position, probabilities and alternatives of chars to a .pred (protobuf) file")
    parser.add_argument("--extended_prediction_data_format", type=str, default="json",
                        help="Extension format: Either pred or json. Note that json will not print logits.")
    parser.add_argument("--no_progress_bars", action="store_true",
                        help="Do not show any progress bars")
    parser.add_argument("--dictionary", nargs="+", default=[],
                        help="List of text files that will be used to create a dictionary")
    parser.add_argument("--beam_width", type=int, default=25,
                        help='Number of beams when using the CTCWordBeamSearch decoder')

    # dataset extra args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=1)

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
