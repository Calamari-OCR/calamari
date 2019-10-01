from edit_distance import edit_distance

from calamari_ocr.ocr.text_processing import synchronize
from calamari_ocr.ocr.text_processing import DefaultTextPreprocessor
from calamari_ocr.utils import parallel_map, tqdm_wrapper
from calamari_ocr.ocr.datasets import StreamingInputDataset
from collections import namedtuple


SingleEvalData = namedtuple('SingleEvalData', ['chars', 'char_errs', 'sync_errs', 'conf', 'gt_pred'])


class Evaluator:
    def __init__(self, text_preprocessor=None, skip_empty_gt=False):
        """ Class to evaluation the CER and errors of two datasets

        Parameters
        ----------
        text_preprocessor : TextProcessor
            a text preprocessor to apply before computing the errors
        skip_empty_gt : bool
            skip gt text lines that are empty
        """
        self.text_preprocessor = text_preprocessor if text_preprocessor is not None else DefaultTextPreprocessor()
        self.skip_empty_gt = skip_empty_gt
        self.preloaded_gt = None

    def preload_gt(self, gt_dataset, progress_bar=False):
        """ Preload gt to be used for several experiments

        Use this method to specify ground truth data to be tested versus many predictions

        Parameters
        ----------
        gt_dataset : Dataset
            the ground truth
        progress_bar : bool, optional
            show a progress bar

        """
        gt_dataset.load_samples(progress_bar=progress_bar)
        self.preloaded_gt = self.text_preprocessor.apply(gt_dataset.text_samples(), progress_bar=progress_bar)

    def run(self, _sentinel=None, gt_dataset=None, pred_dataset=None, processes=1, progress_bar=False):
        """ evaluate on the given dataset

        Parameters
        ----------
        _sentinel : do not use
            Forcing the use of `gt_dataset` and `pred_dataset` fore safety
        gt_dataset : Dataset, optional
            the ground truth
        pred_dataset : Dataset
            the prediction dataset
        processes : int, optional
            the processes to use for preprocessing and evaluation
        progress_bar : bool, optional
            show a progress bar

        Returns
        -------
        evaluation dictionary
        """
        if _sentinel:
            raise Exception("You must call run by using parameter names.")

        if self.preloaded_gt:
            gt_data = self.preloaded_gt
        else:
            # gt_dataset.load_samples(progress_bar=progress_bar)
            # gt_data = self.text_preprocessor.apply(gt_dataset.text_samples(), progress_bar=progress_bar)
            with StreamingInputDataset(gt_dataset, None, self.text_preprocessor, processes=processes) as gt_input_dataset:
                gt_data = [txt for _, txt, _ in tqdm_wrapper(gt_input_dataset.generator(text_only=True),
                                                             total=len(gt_dataset),
                                                             progress_bar=progress_bar,
                                                             desc="Loading GT",
                                                             )]

        with StreamingInputDataset(pred_dataset, None, self.text_preprocessor, processes=processes) as pred_input_dataset:
            pred_data = [txt for _, txt, _ in tqdm_wrapper(pred_input_dataset.generator(text_only=True),
                                                           total=len(pred_dataset),
                                                           progress_bar=progress_bar,
                                                           desc="Loading Prediction"
                                                           )]

        return self.evaluate(gt_data=gt_data, pred_data=pred_data, processes=processes, progress_bar=progress_bar,
                             skip_empty_gt=self.skip_empty_gt)

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
        if _sentinel is not None:
            raise Exception('Call this function by specifying gt and pred explicitly')


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

    @staticmethod
    def evaluate(_sentinel=None, gt_data=None, pred_data=None, processes=1, progress_bar=False, skip_empty_gt=False):
        """ evaluate on the given raw data

        Parameters
        ----------
        _sentinel : do not use
            Forcing the use of `gt_dataset` and `pred_dataset` fore safety
        gt_data : Dataset, optional
            the ground truth
        pred_data : Dataset
            the prediction dataset
        processes : int, optional
            the processes to use for preprocessing and evaluation
        progress_bar : bool, optional
            show a progress bar
        skip_empty_gt : bool
            skip gt text lines that are empty

        Returns
        -------
        evaluation dictionary
        """
        if len(gt_data) != len(pred_data):
            raise Exception("Mismatch in gt and pred files count: {} vs {}".format(len(gt_data), len(pred_data)))

        # evaluate single lines
        out = parallel_map(Evaluator.evaluate_single_args, [{'gt': gt, 'pred': pred, 'skip_empty_gt': skip_empty_gt} for gt, pred in zip(gt_data, pred_data)],
                           processes=processes, progress_bar=progress_bar, desc="Evaluation")

        return Evaluator.evaluate_single_list(out, True)
