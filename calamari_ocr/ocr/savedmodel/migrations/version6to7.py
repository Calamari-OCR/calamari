import logging
import os
from shutil import rmtree

from tensorflow import keras

from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import TrainerParams

logger = logging.getLogger(__name__)


def update_model(params: dict, path: str):
    logger.info(f"Updating model at {path}")

    #model = keras.models.load_model(path, custom_objects=CalamariScenario.model_cls().all_custom_objects())
    
    trainer_params = TrainerParams.from_dict(params)
    scenario_params = trainer_params.scenario
    scenario = CalamariScenario(scenario_params)
    inputs = scenario.data.create_input_layers()
    outputs = scenario.graph.predict(inputs)
    pred_model = keras.models.Model(inputs, outputs)
    pred_model.load_weights(path)


    logger.info(f"Writing converted model at {path}.tmp.keras")
    pred_model.save(path + ".tmp.keras", save_format="keras")

    logger.info(f"Attempting to load converted model at {path}.tmp.keras")
    keras.models.load_model(
        path + ".tmp.keras",
        custom_objects=CalamariScenario.model_cls().all_custom_objects(),
        safe_mode=True
    )

    logger.info(f"Replacing old model at {path}")
    rmtree(path)
    os.rename(path + ".tmp.keras", path + ".keras")
    logger.info(f"New model successfully written")
    keras.backend.clear_session()
