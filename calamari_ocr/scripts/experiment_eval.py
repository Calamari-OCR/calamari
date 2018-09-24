import argparse
import codecs
import os
import pickle

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr.voting import VoterParams, voter_from_proto
from calamari_ocr.ocr import create_dataset, DataSetType, MultiPredictor, Evaluator, RawDataSet, DataSetMode
from calamari_ocr.ocr.text_processing import text_processor_from_proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_imgs", type=str, nargs="+", required=True,
                        help="The evaluation files")
    parser.add_argument("--eval_dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[],
                        help="Path to the checkpoint without file extension")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--voter", type=str, nargs="+", default=["sequence_voter", "confidence_voter_default_ctc", "confidence_voter_fuzzy_ctc"],
                        help="The voting algorithm to use. Possible values: confidence_voter_default_ctc (default), "
                             "confidence_voter_fuzzy_ctc, sequence_voter")
    parser.add_argument("--batch_size", type=int, default=10,
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

    dataset = create_dataset(
        args.eval_dataset,
        DataSetMode.TRAIN,
        images=gt_images,
        texts=gt_txts,
        skip_invalid=not args.no_skip_invalid_gt
    )

    print("Found {} files in the dataset".format(len(dataset)))
    if len(dataset) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    # predict for all models
    n_models = len(args.checkpoint)
    predictor = MultiPredictor(checkpoints=args.checkpoint, batch_size=args.batch_size, processes=args.processes)
    do_prediction = predictor.predict_dataset(dataset, progress_bar=True)

    voters = []
    all_voter_sentences = []
    all_prediction_sentences = [[] for _ in range(n_models)]

    for voter in args.voter:
        # create voter
        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value(voter.upper())
        voters.append(voter_from_proto(voter_params))
        all_voter_sentences.append([])

    for prediction, sample in do_prediction:
        for sent, p in zip(all_prediction_sentences, prediction):
            sent.append(p.sentence)

        # vote results
        for voter, voter_sentences in zip(voters, all_voter_sentences):
            voter_sentences.append(voter.vote_prediction_result(prediction).sentence)

    # evaluation
    text_preproc = text_processor_from_proto(predictor.predictors[0].model_params.text_preprocessor)
    evaluator = Evaluator(text_preprocessor=text_preproc)
    evaluator.preload_gt(gt_dataset=dataset, progress_bar=True)

    def single_evaluation(predicted_sentences):
        if len(predicted_sentences) != len(dataset):
            raise Exception("Mismatch in number of gt and pred files: {} != {}. Probably, the prediction did "
                            "not succeed".format(len(dataset), len(predicted_sentences)))

        pred_data_set = create_dataset(
            DataSetType.RAW,
            DataSetMode.EVAL,
            texts=predicted_sentences)

        r = evaluator.run(pred_dataset=pred_data_set, progress_bar=True, processes=args.processes)

        return r

    full_evaluation = {}
    for id, data in [(str(i), sent) for i, sent in enumerate(all_prediction_sentences)] + list(zip(args.voter, all_voter_sentences)):
        full_evaluation[id] = {"eval": single_evaluation(data), "data": data}

    if args.verbose:
        print(full_evaluation)

    if args.dump:
        import pickle
        with open(args.dump, 'wb') as f:
            pickle.dump({"full": full_evaluation, "gt_txts": gt_txts, "gt": dataset.text_samples()}, f)


if __name__=="__main__":
    main()
