from calamari_ocr.ocr.datasets import DataSet, DataSetMode, DatasetGenerator
from calamari_ocr.utils import split_all_ext

from google.protobuf.json_format import Parse
from calamari_ocr.proto import Predictions

import codecs
from typing import List


class ExtendedPredictionDataSet(DataSet):
    def __init__(self, texts: List[str] = None):
        super().__init__(DataSetMode.EVAL)

        if texts is None:
            texts = []

        for text in texts:
            text_bn, text_ext = split_all_ext(text)
            sample = {
                "image_path": None,
                "pred_path": text,
                "id": text_bn,
            }
            self._load_sample(sample)
            self.add_sample(sample)

    def _load_sample(self, sample):
        gt_txt_path = sample['pred_path']
        if gt_txt_path is None:
            return None, None

        if gt_txt_path.endswith('.json'):
            with codecs.open(gt_txt_path, 'r', 'utf-8') as f:
                p = Parse(str(f.read()), Predictions())
                if len(p.predictions) == 0:
                    return None, None

                voted_p = p.predictions[0]
                for vp in p.predictions:
                    if vp.id == 'voted':
                        voted_p = vp

                sample['best_prediction'] = voted_p
                sample['predictions'] = p

                return None, voted_p.sentence

    def create_generator(self, output_queue, epochs, text_only) -> DatasetGenerator:
        raise NotImplemented()







