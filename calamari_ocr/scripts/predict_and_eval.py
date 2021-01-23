import argparse

from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.params import FileDataReaderArgs
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.scripts.eval import print_confusions, print_worst_lines
from calamari_ocr.utils import glob_all
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr.ocr import DataSetType, PipelineParams
from calamari_ocr import __version__


def main():
    parser = argparse.ArgumentParser()

    # GENERAL/SHARED PARAMETERS
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    parser.add_argument("--files", nargs="+", required=True, default=[],
                        help="List all image files that shall be processed")
    parser.add_argument("--text_files", nargs="+", default=None,
                        help="Optional list of additional text files. E.g. when updating Abbyy prediction, this parameter must be used for the xml files.")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--gt_extension", type=str, default=None,
                        help="Define the gt extension.")
    parser.add_argument("-j", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="The batch size during the prediction (number of lines to process in parallel)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--no_progress_bars", action="store_true",
                        help="Do not show any progress bars")
    parser.add_argument("--dump", type=str,
                        help="Dump the output as serialized pickle object")
    parser.add_argument("--no_skip_invalid_gt", action="store_true",
                        help="Do no skip invalid gt, instead raise an exception.")
    # dataset extra args
    parser.add_argument("--dataset_pad", default=None, nargs='+', type=int)
    parser.add_argument("--pagexml_text_index", default=1)

    # PREDICT PARAMETERS
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="Path to the checkpoint without file extension")

    # EVAL PARAMETERS
    parser.add_argument("--output_individual_voters", action='store_true', default=False)
    parser.add_argument("--n_confusions", type=int, default=10,
                        help="Only print n most common confusions. Defaults to 10, use -1 for all.")

    args = parser.parse_args()

    # allow user to specify json file for model definition, but remove the file extension
    # for further processing
    args.checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in args.checkpoint]
    args.checkpoint = glob_all(args.checkpoint)
    args.checkpoint = [cp[:-5] for cp in args.checkpoint]
    # load files
    if args.gt_extension is None:
        args.gt_extension = DataSetType.gt_extension(args.dataset)

    pipeline_params = PipelineParams(
        type=args.dataset,
        skip_invalid=not args.no_skip_invalid_gt,
        remove_invalid=True,
        files=args.files,
        gt_extension=args.gt_extension,
        text_files=args.text_files,
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
    do_prediction = predictor.predict(pipeline_params)

    all_voter_sentences = []
    all_prediction_sentences = {}

    for s in do_prediction:
        inputs, (result, prediction), meta = s.inputs, s.outputs, s.meta
        sentence = prediction.sentence
        if prediction.voter_predictions is not None and args.output_individual_voters:
            for i, p in enumerate(prediction.voter_predictions):
                if i not in all_prediction_sentences:
                    all_prediction_sentences[i] = []
                all_prediction_sentences[i].append(p.sentence)
        all_voter_sentences.append(sentence)

    # evaluation
    from calamari_ocr.ocr.evaluator import Evaluator
    evaluator = Evaluator(predictor.data)
    evaluator.preload_gt(gt_dataset=pipeline_params, progress_bar=True)

    def single_evaluation(label, predicted_sentences):
        if len(predicted_sentences) != len(evaluator.preloaded_gt):
            raise Exception("Mismatch in number of gt and pred files: {} != {}. Probably, the prediction did "
                            "not succeed".format(len(evaluator.preloaded_gt), len(predicted_sentences)))

        r = evaluator.evaluate(gt_data=evaluator.preloaded_gt, pred_data=predicted_sentences,
                               progress_bar=True, processes=args.processes)

        print("=================")
        print(f"Evaluation result of {label}")
        print("=================")
        print("")
        print("Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
            r["avg_ler"], r["total_char_errs"], r["total_chars"], r["total_sync_errs"]))
        print()
        print()

        # sort descending
        print_confusions(r, args.n_confusions)

        return r

    full_evaluation = {}
    for id, data in [(str(i), sent) for i, sent in all_prediction_sentences.items()] + [('voted', all_voter_sentences)]:
        full_evaluation[id] = {"eval": single_evaluation(id, data), "data": data}

    if args.verbose:
        print(full_evaluation)

    if args.dump:
        import pickle
        with open(args.dump, 'wb') as f:
            pickle.dump({"full": full_evaluation, "gt": evaluator.preloaded_gt}, f)


if __name__ == "__main__":
    main()
