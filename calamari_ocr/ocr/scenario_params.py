from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip import ScenarioBaseParams

from calamari_ocr.ocr import DataParams
from calamari_ocr.ocr.model.params import ModelParams


@pai_dataclass
@dataclass
class CalamariScenarioBaseParams(ScenarioBaseParams[DataParams, ModelParams]):
    def __post_init__(self):
        self.data.downscale_factor = self.model.compute_downscale_factor().x


@pai_dataclass
@dataclass
class CalamariScenarioParams(CalamariScenarioBaseParams):
    pass


@pai_dataclass
@dataclass
class CalamariEnsembleScenarioParams(CalamariScenarioBaseParams):
    def __post_init__(self):
        super().__post_init__()
        self.data.ensemble = self.model.ensemble
