from dataclasses import dataclass, field

from edit_distance import edit_distance

from collections import namedtuple

from paiargparse import pai_dataclass
from tfaip import DataGeneratorParams
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.definitions import PipelineMode
from tfaip.util.multiprocessing.parallelmap import parallel_map, tqdm_wrapper

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.textprocessors import synchronize

SingleEvalData = namedtuple('SingleEvalData', ['chars', 'char_errs', 'sync_errs', 'conf', 'gt_pred'])


@pai_dataclass
@dataclass
class EvaluatorParams:
    setup: DataPipelineParams = field(default_factory=DataPipelineParams)
    progress_bar: bool = True
    skip_empty_gt: bool = False


class Evaluator:
    def __init__(self, params: EvaluatorParams, data: Data):
        """ Class to evaluation the CER and errors of two dataset
        """
        self.params = params
        self.data = data
        self.preloaded_gt = None
        self.params.setup.mode = PipelineMode.TARGETS

    def preload_gt(self, gt_dataset: DataGeneratorParams, progress_bar=False):
        """ Preload gt to be used for several experiments

        Use this method to specify ground truth data to be tested versus many predictions

        Parameters
        ----------
        gt_dataset : Dataset
            the ground truth
        progress_bar : bool, optional
            show a progress bar

        """
        with self.data.create_pipeline(self.params.setup, gt_dataset) as dataset:
            self.preloaded_gt = [sample.targets for sample in tqdm_wrapper(dataset.generate_input_samples(),
                                                                           total=len(dataset),
                                                                           progress_bar=progress_bar,
                                                                           desc="Loading GT",
                                                                           )]

    def run(self, *, gt_dataset: DataGeneratorParams, pred_dataset: DataGeneratorParams):
        """ evaluate on the given dataset
        Returns
        -------
        evaluation dictionary
        """
        if self.preloaded_gt:
            gt_data = self.preloaded_gt
        else:
            with self.data.create_pipeline(self.params.setup, gt_dataset) as data:
                gt_data = [sample.targets for sample in tqdm_wrapper(data.generate_input_samples(),
                                                                     total=len(data),
                                                                     progress_bar=self.params.progress_bar,
                                                                     desc="Loading GT",
                                                                     )]

        with self.data.create_pipeline(self.params.setup, pred_dataset) as data:
            pred_data = [sample.targets for sample in tqdm_wrapper(data.generate_input_samples(),
                                                                   total=len(data),
                                                                   progress_bar=self.params.progress_bar,
                                                                   desc="Loading Prediction"
                                                                   )]

        return self.evaluate(gt_data=gt_data, pred_data=pred_data)

    @staticmethod
    def evaluate_single_args(args):
        return Evaluator.evaluate_single(**args)

    @staticmethod
    def evaluate_single(_sentinel=None, gt='', pred='', skip_empty_gt=False):
        """ Evaluate a single pair of data

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

    def evaluate(self, *, gt_data=None, pred_data=None):
        """ evaluate on the given raw data

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
        if len(gt_data) != len(pred_data):
            raise Exception("Mismatch in gt and pred files count: {} vs {}".format(len(gt_data), len(pred_data)))

        # evaluate single lines
        out = parallel_map(Evaluator.evaluate_single_args, [{'gt': gt, 'pred': pred, 'skip_empty_gt': self.params.skip_empty_gt} for gt, pred in zip(gt_data, pred_data)],
                           processes=self.params.setup.num_processes, progress_bar=self.params.progress_bar, desc="Evaluation")

        return Evaluator.evaluate_single_list(out, True)
