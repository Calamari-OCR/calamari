import logging
import os

from tensorflow import keras

from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import TrainerParams

logger = logging.getLogger(__name__)


def update_model(params: dict, path: str):
    logger.info(f"Updating model at {path}")

    trainer_params = TrainerParams.from_dict(params)
    scenario_params = trainer_params.scenario
    scenario = CalamariScenario(scenario_params)
    inputs = scenario.data.create_input_layers()
    outputs = scenario.graph.predict(inputs)
    pred_model = keras.models.Model(inputs, outputs)
    pred_model.load_weights(path + ".h5")

    logger.info(f"Writing converted model at {path}.tmp")
    pred_model.save(path + ".tmp", include_optimizer=False)
    logger.info(f"Attempting to load converted model at {path}.tmp")
    keras.models.load_model(
        path + ".tmp",
        custom_objects=CalamariScenario.model_cls().all_custom_objects(),
    )
    logger.info(f"Replacing old model at {path}.h5")
    os.remove(path + ".h5")
    os.rename(path + ".tmp", path)
    logger.info(f"New model successfully written")
    keras.backend.clear_session()
