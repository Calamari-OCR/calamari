from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip.base import ScenarioBaseParams

from calamari_ocr.ocr import DataParams
from calamari_ocr.ocr.model.params import ModelParams


@pai_dataclass
@dataclass
class ScenarioParams(ScenarioBaseParams[DataParams, ModelParams]):
    def __post_init__(self):
        self.data.ensemble = self.model.ensemble
