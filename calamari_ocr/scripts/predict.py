import argparse
import codecs
import os
import json


from calamari_ocr.utils.glob import glob_all
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams, Predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--files", nargs="+", required=True, default=[],
                        help="List all image files that shall be processed")
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
    parser.add_argument("--extended_prediction_data_ext", type=str, default=".pred",
                        help="Extension")

    args = parser.parse_args()

    # allow user to specify json file for model definition, but remove the file extension
    # for further processing
    args.checkpoint = [(cp[:-5] if cp.endswith(".json") else cp) for cp in args.checkpoint]

    # create voter
    voter_params = VoterParams()
    voter_params.type = VoterParams.Type.Value(args.voter.upper())
    voter = voter_from_proto(voter_params)

    # load files
    input_image_files = sorted(glob_all(args.files))

    dataset = FileDataSet(input_image_files)

    print("Found {} files in the dataset".format(len(dataset)))
    if len(dataset) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    # predict for all models
    predictor = MultiPredictor(checkpoints=args.checkpoint)
    do_prediction = predictor.predict_dataset(dataset, batch_size=args.batch_size,
                                                processes=args.processes, progress_bar=True)

    # output the voted results to the appropriate files
    for (result, sample), filepath in zip(do_prediction, input_image_files):
        for i, p in enumerate(result):
            p.prediction.id = "fold_{}".format(i)

        # vote the results (if only one model is given, this will just return the sentences)
        prediction = voter.vote_prediction_result(result)
        prediction.id = "voted"
        sentence = prediction.sentence
        if args.verbose:
            print("{}: '{}'".format(sample['id'], sentence))

        output_dir = args.output_dir if args.output_dir else os.path.dirname(filepath)

        with codecs.open(os.path.join(output_dir, sample['id'] + ".pred.txt"), 'w', 'utf-8') as f:
            f.write(sentence)

        if args.extended_prediction_data:
            with open(os.path.join(output_dir, sample['id'] + args.extended_prediction_data_ext), 'wb') as f:
                ps = Predictions()
                ps.line_path = filepath
                ps.predictions.extend([prediction] + [r.prediction for r in result])
                f.write(ps.SerializeToString())

    print("All files written")


if __name__ == "__main__":
    main()
