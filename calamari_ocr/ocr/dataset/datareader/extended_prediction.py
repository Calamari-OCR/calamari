import codecs
import zlib
from dataclasses import dataclass, field
from typing import List, Type

from paiargparse import pai_dataclass

from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    CalamariDataGeneratorParams,
)
from calamari_ocr.ocr.predict.params import Predictions
from calamari_ocr.utils import split_all_ext, glob_all


@pai_dataclass
@dataclass
class ExtendedPredictionDataParams(CalamariDataGeneratorParams):
    files: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.files)

    def to_prediction(self):
        raise NotImplementedError

    def select(self, indices: List[int]):
        raise NotImplementedError

    @staticmethod
    def cls() -> Type["CalamariDataGenerator"]:
        return ExtendedPredictionDataSet

    def __post_init__(self):
        self.files = sorted(glob_all(self.files))


class ExtendedPredictionDataSet(CalamariDataGenerator[ExtendedPredictionDataParams]):
    def __init__(self, mode, params: ExtendedPredictionDataParams):
        super().__init__(mode, params)
        for text in params.files:
            text_bn, text_ext = split_all_ext(text)
            sample = {
                "image_path": None,
                "pred_path": text,
                "id": text_bn,
            }
            self._load_sample(sample, False)
            self.add_sample(sample)

    def store_text_prediction(self, prediction, sample_id, output_dir):
        raise NotImplementedError

    def _load_sample(self, sample, text_only):
        gt_txt_path = sample["pred_path"]
        if gt_txt_path is None:
            return None, None

        if gt_txt_path.endswith(".json"):
            with codecs.open(gt_txt_path, "r", "utf-8") as f:
                p = Predictions.from_json(f.read())
        elif gt_txt_path.endswith(".pred"):
            with open(gt_txt_path, "rb") as f:
                p = Predictions.from_json(zlib.decompress(f.read()).decode("utf-8"))

        if len(p.predictions) == 0:
            return None, None

        voted_p = p.predictions[0]
        for vp in p.predictions:
            if vp.id == "voted":
                voted_p = vp

        sample["best_prediction"] = voted_p
        sample["predictions"] = p

        return None, voted_p.sentence
