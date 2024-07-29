from dataclasses import dataclass, field
from typing import Optional, List

from paiargparse import pai_dataclass, pai_meta, PAIArgumentParser

from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES
from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.scripts.eval import print_confusions
from calamari_ocr.utils import glob_all
from calamari_ocr.ocr.voting import VoterParams
from calamari_ocr import __version__


@pai_dataclass
@dataclass
class PredictAndEvalArgs:
    checkpoint: List[str] = field(
        metadata=pai_meta(mode="flat", nargs="+", help="Path to the checkpoint without file extension")
    )
    dump: Optional[str] = field(
        default=None,
        metadata=pai_meta(mode="flat", help="Dump the output as serialized pickle object"),
    )
    output_individual_voters: bool = field(default=False, metadata=pai_meta(mode="flat"))
    n_confusions: int = field(
        default=10,
        metadata=pai_meta(
            mode="flat",
            help="Only print n most common confusions. Defaults to 10, use -1 for all.",
        ),
    )
    data: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(mode="flat", help="Input data", choices=DATA_GENERATOR_CHOICES),
    )
    predictor: PredictorParams = field(
        default_factory=PredictorParams,
        metadata=pai_meta(mode="flat", help="Predictor data"),
    )
    skip_empty_gt: bool = field(
        default=False,
        metadata=pai_meta(mode="flat", help="Do not evaluate if the GT is empty"),
    )

    def __post_init__(self):
        assert self.data is not None
        assert len(self.checkpoint) > 0


def run():
    main(parse_args())


def parse_args(args=None):
    parser = PAIArgumentParser()

    # GENERAL/SHARED PARAMETERS
    parser.add_argument("--version", action="version", version="%(prog)s v" + __version__)
    parser.add_root_argument("args", PredictAndEvalArgs)
    return parser.parse_args(args=args).args


def main(args: PredictAndEvalArgs):
    # allow user to specify json file for model definition, but remove the file extension
    # for further processing
    args.checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in args.checkpoint]
    args.checkpoint = glob_all(args.checkpoint)
    args.checkpoint = [cp[:-5] for cp in args.checkpoint]

    from calamari_ocr.ocr.predict.predictor import MultiPredictor

    voter_params = VoterParams()
    predictor = MultiPredictor.from_paths(
        checkpoints=args.checkpoint,
        voter_params=voter_params,
        predictor_params=args.predictor,
    )
    do_prediction = predictor.predict(args.data)

    all_voter_sentences = {}
    all_prediction_sentences = {}

    for s in do_prediction:
        (result, prediction) = s.outputs
        sentence = prediction.sentence
        if prediction.voter_predictions is not None and args.output_individual_voters:
            for i, p in enumerate(prediction.voter_predictions):
                if i not in all_prediction_sentences:
                    all_prediction_sentences[i] = {}
                all_prediction_sentences[i][s.meta["id"]] = p.sentence
        all_voter_sentences[s.meta["id"]] = sentence

    # evaluation
    from calamari_ocr.ocr.evaluator import Evaluator, EvaluatorParams

    evaluator_params = EvaluatorParams(
        setup=args.predictor.pipeline,
        progress_bar=args.predictor.progress_bar,
        skip_empty_gt=args.skip_empty_gt,
    )
    evaluator = Evaluator(evaluator_params, predictor.data)
    evaluator.preload_gt(gt_dataset=args.data, progress_bar=True)

    def single_evaluation(label, predicted_sentences):
        r = evaluator.evaluate(gt_data=evaluator.preloaded_gt, pred_data=predicted_sentences)

        print("=================")
        print(f"Evaluation result of {label}")
        print("=================")
        print("")
        print(
            "Got mean normalized label error rate of {:.2%} ({} errs, {} total chars, {} sync errs)".format(
                r["avg_ler"],
                r["total_char_errs"],
                r["total_chars"],
                r["total_sync_errs"],
            )
        )
        print()
        print()

        # sort descending
        print_confusions(r, args.n_confusions)

        return r

    full_evaluation = {}
    for id, data in [(str(i), sent) for i, sent in all_prediction_sentences.items()] + [("voted", all_voter_sentences)]:
        full_evaluation[id] = {"eval": single_evaluation(id, data), "data": data}

    if not args.predictor.silent:
        print(full_evaluation)

    if args.dump:
        import pickle

        with open(args.dump, "wb") as f:
            pickle.dump({"full": full_evaluation, "gt": evaluator.preloaded_gt}, f)

    return full_evaluation


if __name__ == "__main__":
    run()
