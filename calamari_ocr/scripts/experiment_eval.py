import argparse

from calamari_ocr.ocr.dataset.params import FileDataReaderArgs
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.ocr import DataSetType, PipelineParams


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
    parser.add_argument("--batch_size", type=int, default=10,
                        help="The batch size for prediction")
    parser.add_argument("--dump", type=str,
                        help="Dump the output as serialized pickle object")
    parser.add_argument("--no_skip_invalid_gt", action="store_true",
                        help="Do no skip invalid gt, instead raise an exception.")

    # dataset extra args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=1)

    args = parser.parse_args()

    # allow user to specify json file for model definition, but remove the file extension
    # for further processing
    args.checkpoint = [(cp[:-5] if cp.endswith(".json") else cp) for cp in args.checkpoint]

    # load files
    gt_images = sorted(glob_all(args.eval_imgs))
    gt_txts = [split_all_ext(path)[0] + ".gt.txt" for path in sorted(glob_all(args.eval_imgs))]

    gt_params = PipelineParams(
        type=args.eval_dataset,
        skip_invalid=not args.no_skip_invalid_gt,
        remove_invalid=True,
        files=gt_images,
        text_files=gt_txts,
        data_reader_args=FileDataReaderArgs(
            pad=args.dataset_pad,
            text_index=args.pagexml_text_index,
        ),
        batch_size=args.batch_size,
        num_processes=args.processes,
    )

    from calamari_ocr.ocr.predict.predictor import MultiPredictor
    voter_params = VoterParams()
    predictor = MultiPredictor.from_paths(checkpoints=args.checkpoint, voter_params=voter_params,
                                          predictor_params=PredictorParams(silent=True, progress_bar=True))
    do_prediction = predictor.predict(gt_params)

    all_voter_sentences = []
    all_prediction_sentences = {}

    for s in do_prediction:
        inputs, (result, prediction), meta = s.inputs, s.outputs, s.meta
        sentence = prediction.sentence
        for i, p in enumerate(prediction.voter_predictions):
            if i not in all_prediction_sentences:
                all_prediction_sentences[i] = []
            all_prediction_sentences[i].append(p.sentence)
        all_voter_sentences.append(sentence)

    # evaluation
    from calamari_ocr.ocr.evaluator import Evaluator
    evaluator = Evaluator(predictor.data)
    evaluator.preload_gt(gt_dataset=gt_params, progress_bar=True)

    def single_evaluation(predicted_sentences):
        if len(predicted_sentences) != len(evaluator.preloaded_gt):
            raise Exception("Mismatch in number of gt and pred files: {} != {}. Probably, the prediction did "
                            "not succeed".format(len(evaluator.preloaded_gt), len(predicted_sentences)))

        r = evaluator.evaluate(gt_data=evaluator.preloaded_gt, pred_data=predicted_sentences,
                               progress_bar=True, processes=args.processes)
        return r

    full_evaluation = {}
    for id, data in [(str(i), sent) for i, sent in all_prediction_sentences.items()] + [('voted', all_voter_sentences)]:
        full_evaluation[id] = {"eval": single_evaluation(data), "data": data}

    if args.verbose:
        print(full_evaluation)

    if args.dump:
        import pickle
        with open(args.dump, 'wb') as f:
            pickle.dump({"full": full_evaluation, "gt_txts": gt_txts, "gt": evaluator.preloaded_gt}, f)


if __name__ == "__main__":
    main()
