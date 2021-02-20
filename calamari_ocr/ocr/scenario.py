import os

from tfaip.base.scenario.scenariobase import ScenarioBase
from tfaip.base.trainer.scheduler import Constant

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.model.ensemblemodel import EnsembleModel
from calamari_ocr.ocr.model.model import Model
from calamari_ocr.ocr.scenario_params import ScenarioParams
from calamari_ocr.ocr.training.params import CalamariDefaultTrainValGeneratorParams, TrainerParams
from calamari_ocr.ocr.training.trainer import Trainer


class Scenario(ScenarioBase[Data, Model, ScenarioParams, CalamariDefaultTrainValGeneratorParams]):
    def create_model(self):
        if self._params.model.ensemble <= 0:
            return Model(self._params.model)
        else:
            return EnsembleModel(self._params.model)

    @classmethod
    def trainer_cls(cls):
        return Trainer

    @classmethod
    def default_params(cls):
        scenario_params = super(Scenario, cls).default_params()
        scenario_params.export_serve = True
        scenario_params.export_net_config = False
        scenario_params.default_serve_dir = 'best.ckpt.h5'
        scenario_params.scenario_params_filename = 'scenario_params.json'  # should never be written!
        scenario_params.trainer_params_filename = 'best.ckpt.json'
        return scenario_params

    @classmethod
    def default_trainer_params(cls) -> 'TrainerParams':
        trainer_params = super(Scenario, cls).default_trainer_params()
        trainer_params.export_final = False
        trainer_params.checkpoint_sub_dir = os.path.join('checkpoint', 'checkpoint_{epoch:04d}')
        trainer_params.early_stopping.upper_threshold = 0.9
        trainer_params.early_stopping.lower_threshold = 0.0
        trainer_params.early_stopping.frequency = 1
        trainer_params.early_stopping.n_to_go = 5
        trainer_params.skip_model_load_test = True
        trainer_params.optimizer.clip_grad = 5
        trainer_params.learning_rate = Constant()
        return trainer_params

    def __init__(self, params):
        super(Scenario, self).__init__(params)
