import zlib

from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.datareader.base import DataReader
from calamari_ocr.ocr.predict.params import Predictions
from calamari_ocr.utils import split_all_ext

import codecs
from typing import List


class ExtendedPredictionDataSet(DataReader):
    def __init__(self, texts: List[str] = None):
        super().__init__(PipelineMode.Evaluation)

        if texts is None:
            texts = []

        for text in texts:
            text_bn, text_ext = split_all_ext(text)
            sample = {
                "image_path": None,
                "pred_path": text,
                "id": text_bn,
            }
            self._load_sample(sample, False)
            self.add_sample(sample)

    def _load_sample(self, sample, text_only):
        gt_txt_path = sample['pred_path']
        if gt_txt_path is None:
            return None, None

        if gt_txt_path.endswith('.json'):
            with codecs.open(gt_txt_path, 'r', 'utf-8') as f:
                p = Predictions.from_json(f.read())
        elif gt_txt_path.endswith('.pred'):
            with open(gt_txt_path, 'rb') as f:
                p = Predictions.from_json(zlib.decompress(f.read()).decode('utf-8'))

        if len(p.predictions) == 0:
            return None, None

        voted_p = p.predictions[0]
        for vp in p.predictions:
            if vp.id == 'voted':
                voted_p = vp

        sample['best_prediction'] = voted_p
        sample['predictions'] = p

        return None, voted_p.sentence
