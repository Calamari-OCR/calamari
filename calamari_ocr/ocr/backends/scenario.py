import os
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

    @classmethod
    def default_params(cls):
        scenario_params = super(CalamariScenario, cls).default_params()
        scenario_params.export_serve = True
        scenario_params.export_frozen = False
        scenario_params.export_net_config_ = False
        scenario_params.default_serve_dir_ = 'best.ckpt.h5'
        scenario_params.scenario_params_filename_ = None  # should never be written!
        scenario_params.trainer_params_filename_ = 'best.ckpt.json'
        return scenario_params

    @classmethod
    def default_trainer_params(cls) -> 'TrainerParams':
        trainer_params = super(CalamariScenario, cls).default_trainer_params()
        trainer_params.export_final = False
        trainer_params.checkpoint_sub_dir_ = os.path.join('checkpoint', 'checkpoint_{epoch:04d}')
        trainer_params.checkpoint_save_freq_ = None
        return trainer_params

    def __init__(self, params):
        super(CalamariScenario, self).__init__(params)
