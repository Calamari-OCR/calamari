import argparse
import codecs
import os
import pickle

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr.voting import VoterParams, voter_from_proto
from calamari_ocr.ocr import FileDataSet, MultiPredictor, Evaluator, RawDataSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_imgs", type=str, nargs="+", required=True,
                        help="The evaluation files")
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[],
                        help="Path to the checkpoint without file extension")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--voter", type=str, nargs="+", default=["sequence_voter", "confidence_voter_default_ctc", "confidence_voter_fuzzy_ctc"],
                        help="The voting algorithm to use. Possible values: confidence_voter_default_ctc (default), "
                             "confidence_voter_fuzzy_ctc, sequence_voter")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size for prediction")
    parser.add_argument("--dump", type=str,
                        help="Dump the output as serialized pickle object")
    parser.add_argument("--no_skip_invalid_gt", action="store_true",
                        help="Do no skip invalid gt, instead raise an exception.")

    args = parser.parse_args()

    # allow user to specify json file for model definition, but remove the file extension
    # for further processing
    args.checkpoint = [(cp[:-5] if cp.endswith(".json") else cp) for cp in args.checkpoint]

    # load files
    gt_images = sorted(glob_all(args.eval_imgs))
    gt_txts = [split_all_ext(path)[0] + ".gt.txt" for path in sorted(glob_all(args.eval_imgs))]

    dataset = FileDataSet(images=gt_images, texts=gt_txts, skip_invalid=not args.no_skip_invalid_gt)

    print("Found {} files in the dataset".format(len(dataset)))
    if len(dataset) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    # predict for all models
    predictor = MultiPredictor(checkpoints=args.checkpoint)
    predictions = list(predictor.predict_dataset(dataset, batch_size=args.batch_size,
                                                processes=args.processes, progress_bar=True))
    result = [r for r, s in predictions]
    samples = [s for r, s in predictions]

    # vote results
    all_voter_sentences = []
    for voter in args.voter:
        # create voter
        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value(voter.upper())
        voter = voter_from_proto(voter_params)

        # vote the results (if only one model is given, this will just return the sentences)
        voted_sentences = voter.vote_prediction_results(result)
        voted_sentences = [s.sentence for s in voted_sentences]
        all_voter_sentences.append(voted_sentences)

    # evaluation
    evaluator = Evaluator()

    def single_evaluation(predicted_sentences):
        if len(predicted_sentences) != len(dataset):
            raise Exception("Mismatch in number of gt and pred files: {} != {}. Probably, the prediction did "
                            "not succeed".format(len(dataset), len(predicted_sentences)))

        pred_data_set = RawDataSet(texts=predicted_sentences)

        r = evaluator.run(gt_dataset=dataset, pred_dataset=pred_data_set, progress_bar=True)

        return r

    full_evaluation = {}
    for id, data in [(str(i), [r[i].sentence for r in result]) for i in range(len(result[0]))] + list(zip(args.voter, all_voter_sentences)):
        full_evaluation[id] = {"eval": single_evaluation(data), "data": data}

    if args.verbose:
        print(full_evaluation)

    if args.dump:
        import pickle
        with open(args.dump, 'wb') as f:
            pickle.dump(full_evaluation, f)


if __name__=="__main__":
    main()
