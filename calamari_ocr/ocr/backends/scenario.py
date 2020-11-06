from typing import Type

from tfaip.base.scenario import ScenarioBase

from calamari_ocr.ocr.backends.dataset import CalamariData
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import CalamariModel
from calamari_ocr.ocr.trainer import CalamariTrainer


class CalamariScenario(ScenarioBase):
    @classmethod
    def data_cls(cls) -> Type['DataBase']:
        return CalamariData

    @classmethod
    def model_cls(cls):
        return CalamariModel

    @classmethod
    def trainer_cls(cls):
        return CalamariTrainer

    def __init__(self, params):
        super(CalamariScenario, self).__init__(params)
