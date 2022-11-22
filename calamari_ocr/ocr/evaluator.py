from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING
import logging

from edit_distance import edit_distance

from collections import namedtuple

from paiargparse import pai_dataclass, pai_meta
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.definitions import PipelineMode
from tfaip.util.multiprocessing.parallelmap import parallel_map, tqdm_wrapper

from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.textprocessors import synchronize

if TYPE_CHECKING:
    from calamari_ocr.ocr.dataset.data import Data

logger = logging.getLogger(__name__)
SingleEvalData = namedtuple("SingleEvalData", ["chars", "char_errs", "sync_errs", "conf", "gt_pred"])


@pai_dataclass
@dataclass
class EvaluatorParams:
    setup: DataPipelineParams = field(default_factory=DataPipelineParams)
    progress_bar: bool = True
    skip_empty_gt: bool = False
    non_existing_pred_handling_mode: str = field(
        default="empty",
        metadata=pai_meta(
            mode="flat",
            choices=["error", "skip", "empty"],
            help="How to handle non existing prediction data. Possible modes: skip, empty, error. "
            "'Skip' will simply skip the evaluation of that file (not counting it to errors). "
            "'Empty' will handle this file as would it be empty (fully checking for errors). "
            "'Error' will throw an exception if a file is not existing.",
        ),
    )


class Evaluator:
    def __init__(self, params: EvaluatorParams, data: "Data"):
        """Class to evaluation the CER and errors of two dataset"""
        self.params = params
        self.data = data
        self.preloaded_gt = None
        self.params.setup.mode = PipelineMode.TARGETS

    def preload_gt(self, gt_dataset: CalamariDataGeneratorParams, progress_bar=False):
        """Preload gt to be used for several experiments

        Use this method to specify ground truth data to be tested versus many predictions

        Parameters
        ----------
        gt_dataset : Dataset
            the ground truth
        progress_bar : bool, optional
            show a progress bar

        """
        pipeline = self.data.create_pipeline(self.params.setup, gt_dataset).generate_input_samples()
        with pipeline as samples:
            self.preloaded_gt = {
                sample.meta["id"]: sample.targets
                for sample in tqdm_wrapper(
                    samples,
                    total=len(pipeline),
                    progress_bar=progress_bar,
                    desc="Loading GT",
                )
            }

        if len(self.preloaded_gt) == 0:
            raise ValueError("Empty GT dataset.")

    def run(self, *, gt_dataset: CalamariDataGeneratorParams, pred_dataset: CalamariDataGeneratorParams):
        """evaluate on the given dataset
        Returns
        -------
        evaluation dictionary
        """
        if self.preloaded_gt:
            gt_data = self.preloaded_gt
        else:
            pipeline = self.data.create_pipeline(self.params.setup, gt_dataset).generate_input_samples()
            with pipeline as samples:
                gt_data = {
                    sample.meta["id"]: sample.targets
                    for sample in tqdm_wrapper(
                        samples,
                        total=len(pipeline),
                        progress_bar=self.params.progress_bar,
                        desc="Loading GT",
                    )
                }

        pipeline = self.data.create_pipeline(self.params.setup, pred_dataset).generate_input_samples()
        with pipeline as samples:
            pred_data = {
                sample.meta["id"]: sample.targets
                for sample in tqdm_wrapper(
                    samples,
                    total=len(pipeline),
                    progress_bar=self.params.progress_bar,
                    desc="Loading Prediction",
                )
            }

        return self.evaluate(gt_data=gt_data, pred_data=pred_data)

    @staticmethod
    def evaluate_single_args(args):
        return Evaluator.evaluate_single(**args)

    @staticmethod
    def evaluate_single(_sentinel=None, gt="", pred="", skip_empty_gt=False):
        """Evaluate a single pair of data

        Parameters
        ----------
        _sentinel : None
            Sentinel to force to specify gt and pred manually
        gt : str
            ground truth
        pred : str
            prediction
        skip_empty_gt : bool
            skip gt text lines that are empty

        Returns
        -------
        int
            length of ground truth
        int
            number of errors
        int
            number of synchronisation errors
        dict
            confusions dictionary
        tuple(str, str)
            ground_truth, prediction (same as input)

        """
        confusion = {}
        total_sync_errs = 0

        if len(gt) == 0 and skip_empty_gt:
            return 0, 0, 0, confusion, (gt, pred)

        errs, trues = edit_distance(gt, pred)
        synclist = synchronize([gt, pred])
        for sync in synclist:
            gt_str, pred_str = sync.get_text()
            if gt_str != pred_str:
                key = (gt_str, pred_str)
                total_sync_errs += max(len(gt_str), len(pred_str))
                if key not in confusion:
                    confusion[key] = 1
                else:
                    confusion[key] += 1

        return len(gt), errs, total_sync_errs, confusion, (gt, pred)

    @staticmethod
    def evaluate_single_list(eval_results, store_all=False):
        # sum all errors up
        all_eval = []
        total_instances = 0
        total_chars = 0
        total_char_errs = 0
        confusion = {}
        total_sync_errs = 0
        for chars, char_errs, sync_errs, conf, gt_pred in eval_results:
            if store_all:
                all_eval.append(SingleEvalData(chars, char_errs, sync_errs, conf, gt_pred))

            total_instances += 1
            total_chars += chars
            total_char_errs += char_errs
            total_sync_errs += sync_errs
            for key, value in conf.items():
                if key not in confusion:
                    confusion[key] = value
                else:
                    confusion[key] += value

        # Note the sync errs can be higher than the true edit distance because
        # replacements are counted as 1
        # e.g. ed(in ewych, ierg ch) = 5
        #      sync(in ewych, ierg ch) = [{i: i}, {n: erg}, {ewy: }, {ch: ch}] = 6

        return {
            "single": all_eval,
            "total_instances": total_instances,
            "avg_ler": total_char_errs / total_chars,
            "total_chars": total_chars,
            "total_char_errs": total_char_errs,
            "total_sync_errs": total_sync_errs,
            "confusion": confusion,
        }

    def evaluate(self, *, gt_data: Dict[str, str], pred_data: Dict[str, str]):
        """evaluate on the given raw data

        Parameters
        ----------
        gt_data : Dataset, optional
            the ground truth
        pred_data : Dataset
            the prediction dataset

        Returns
        -------
        evaluation dictionary
        """
        if self.params.non_existing_pred_handling_mode != "error":
            n_empty = 0
            mapped_pred_data = {}
            for sample_id in list(gt_data.keys()):
                if sample_id in pred_data:
                    mapped_pred_data[sample_id] = pred_data[sample_id]
                else:
                    if self.params.non_existing_pred_handling_mode == "empty":
                        mapped_pred_data[sample_id] = ""
                    else:
                        del gt_data[sample_id]  # skip
                    n_empty += 1
            logger.info(f"{n_empty}/{len(gt_data)} lines could not be matched during the evaluation.")
            if n_empty == len(gt_data):
                raise ValueError(
                    f"No lines could be matched by their ID. First 10 gt ids "
                    f"{list(gt_data.keys())[:10]}, first 10 pred ids {list(pred_data.keys())[:10]}"
                )
            pred_data = mapped_pred_data

        gt_ids, pred_ids = set(gt_data.keys()), set(pred_data.keys())
        if len(gt_ids) != len(gt_data):
            raise ValueError("Non unique keys in ground truth data.")
        if gt_ids != pred_ids:
            raise Exception(
                f"Mismatch in gt and pred. Samples could not be matched by ID. "
                f"GT without PRED: {gt_ids.difference(pred_ids)}. "
                f"PRED without GT: {pred_ids.difference(gt_ids)}"
            )

        gt_pred = [(gt_data[s_id], pred_data[s_id]) for s_id in gt_ids]
        # evaluate single lines
        out = parallel_map(
            Evaluator.evaluate_single_args,
            [{"gt": gt, "pred": pred, "skip_empty_gt": self.params.skip_empty_gt} for gt, pred in gt_pred],
            processes=self.params.setup.num_processes,
            progress_bar=self.params.progress_bar,
            desc="Evaluation",
        )

        res = Evaluator.evaluate_single_list(out, True)
        res["ids"] = gt_ids
        return res
