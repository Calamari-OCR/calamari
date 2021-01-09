import os
from typing import Type

from tfaip.base.scenario.scenariobase import ScenarioBase

from calamari_ocr.ocr.model.model import Model
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.training.trainer import Trainer


class Scenario(ScenarioBase):
    @classmethod
    def data_cls(cls) -> Type['Data']:
        return Data

    @classmethod
    def model_cls(cls):
        return Model

    @classmethod
    def trainer_cls(cls):
        return Trainer

    @classmethod
    def default_params(cls):
        scenario_params = super(Scenario, cls).default_params()
        scenario_params.export_serve = True
        scenario_params.export_frozen = False
        scenario_params.export_net_config_ = False
        scenario_params.default_serve_dir_ = 'best.ckpt.h5'
        scenario_params.scenario_params_filename_ = 'scenario_params.json'  # should never be written!
        scenario_params.trainer_params_filename_ = 'best.ckpt.json'
        return scenario_params

    @classmethod
    def default_trainer_params(cls) -> 'TrainerParams':
        trainer_params = super(Scenario, cls).default_trainer_params()
        trainer_params.export_final = False
        trainer_params.checkpoint_sub_dir_ = os.path.join('checkpoint', 'checkpoint_{epoch:04d}')
        trainer_params.checkpoint_save_freq_ = None
        return trainer_params

    def __init__(self, params):
        super(Scenario, self).__init__(params)
