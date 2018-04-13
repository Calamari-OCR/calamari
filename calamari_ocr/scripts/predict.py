import argparse
import codecs
import os

from calamari_ocr.utils.glob import glob_all
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--files", nargs="+", required=True, default=[],
                        help="List all image files that shall be processed")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[],
                        help="Path to the checkpoint without file extension")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--voter", type=str, default="confidence_voter_default_ctc",
                        help="The voting algorithm to use. Possible values: confidence_voter_default_ctc (default), "
                             "confidence_voter_fuzzy_ctc, sequence_voter")

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
    result, samples = predictor.predict_dataset(dataset, args.processes, progress_bar=True)

    # vote the results (if only one model is given, this will just return the sentences)
    voted_sentences = voter.vote_prediction_results(result)

    # output the voted results to the appropriate files
    for sentence, sample, filepath in zip(voted_sentences, samples, input_image_files):
        if args.verbose:
            print("{}: '{}'".format(sample['id'], sentence))

        with codecs.open(os.path.join(os.path.dirname(filepath), sample['id'] + ".pred.txt"), 'w', 'utf-8') as f:
            f.write(sentence)

    print("All files written")


if __name__ == "__main__":
    main()
