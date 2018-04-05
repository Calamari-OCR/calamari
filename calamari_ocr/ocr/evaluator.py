from edit_distance import edit_distance

from calamari_ocr.ocr.text_processing import synchronize
from calamari_ocr.ocr.text_processing import DefaultTextPreprocessor


class Evaluator:
    def __init__(self, text_preprocessor=None):
        self.text_preprocessor = text_preprocessor if text_preprocessor is not None else DefaultTextPreprocessor()

    def run(self, _sentinel=None, gt_dataset=None, pred_dataset=None, progress_bar=False):
        if _sentinel:
            raise Exception("You must call run by using parameter names.")

        gt_dataset.load_samples(progress_bar=progress_bar)
        pred_dataset.load_samples(progress_bar=progress_bar)

        gt_data = self.text_preprocessor.apply(gt_dataset.text_samples(), progress_bar=progress_bar)
        pred_data = self.text_preprocessor.apply(pred_dataset.text_samples(), progress_bar=progress_bar)

        if len(gt_data) != len(pred_data):
            raise Exception("Mismatch in gt and pred files count: {} vs {}".format(len(gt_data), len(pred_data)))

        total_chars = 0
        total_char_errs = 0
        confusion = {}
        for gt, pred in zip(gt_data, pred_data):
            # TODO: more metrics, e.g. confusion matrix (using sequence alignment)
            errs, trues = edit_distance(gt, pred)
            total_chars += len(gt)
            total_char_errs += errs
            synclist = synchronize([gt, pred])
            for sync in synclist:
                gt_str, pred_str = sync.get_text()
                if gt_str != pred_str:
                    key = (gt_str, pred_str)
                    if key not in confusion:
                        confusion[key] = 1
                    else:
                        confusion[key] += 1

        return {
            "avg_ler": total_char_errs / total_chars,
            "total_chars": total_chars,
            "total_char_errs": total_char_errs,
            "confusion": confusion,
        }

