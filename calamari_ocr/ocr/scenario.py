import os
from typing import Type, TYPE_CHECKING

from tfaip.scenario.scenariobase import ScenarioBase, TModel, TScenarioParams, TTrainerPipelineParams
from tfaip.trainer.scheduler import Constant

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.model.ensemblemodel import EnsembleModel
from calamari_ocr.ocr.model.model import Model
from calamari_ocr.ocr.scenario_params import CalamariScenarioParams, CalamariEnsembleScenarioParams
from calamari_ocr.ocr.training.pipeline_params import CalamariDefaultTrainerPipelineParams, \
    CalamariTrainOnlyPipelineParams

if TYPE_CHECKING:
    from calamari_ocr.ocr.training.params import TrainerParams


class CalamariScenarioBase(ScenarioBase[Data, TModel, TScenarioParams, CalamariDefaultTrainerPipelineParams]):
    @classmethod
    def trainer_cls(cls):
        from calamari_ocr.ocr.training.trainer import Trainer
        return Trainer

    @classmethod
    def default_params(cls):
        scenario_params = super(CalamariScenarioBase, cls).default_params()
        scenario_params.export_serve = True
        scenario_params.export_net_config = False
        scenario_params.default_serve_dir = 'best.ckpt.h5'
        scenario_params.scenario_params_filename = 'scenario_params.json'  # should never be written!
        scenario_params.trainer_params_filename = 'best.ckpt.json'
        return scenario_params

    @classmethod
    def default_trainer_params(cls) -> 'TrainerParams':
        trainer_params = super(CalamariScenarioBase, cls).default_trainer_params()
        trainer_params.export_final = False
        trainer_params.checkpoint_sub_dir = os.path.join('checkpoint', 'checkpoint_{epoch:04d}')
        trainer_params.early_stopping.upper_threshold = 0.9
        trainer_params.early_stopping.lower_threshold = 0.0
        trainer_params.early_stopping.frequency = 1
        trainer_params.early_stopping.n_to_go = 5
        trainer_params.skip_model_load_test = True
        trainer_params.optimizer.global_clip_norm = 5
        trainer_params.learning_rate = Constant()
        return trainer_params


class CalamariScenario(CalamariScenarioBase[Model, CalamariScenarioParams]):
    pass


class CalamariEnsembleScenario(CalamariScenarioBase[EnsembleModel, CalamariEnsembleScenarioParams]):
    @classmethod
    def default_params(cls):
        p = super().default_params()
        p.model.ensemble = 5
        return p

    @classmethod
    def trainer_pipeline_params_cls(cls) -> Type[TTrainerPipelineParams]:
        return CalamariTrainOnlyPipelineParams
