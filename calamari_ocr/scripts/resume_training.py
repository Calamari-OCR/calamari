import json
import os
from argparse import ArgumentParser
import logging

from tfaip.util.logging import setup_log

from calamari_ocr.ocr.scenario import Scenario
from calamari_ocr.ocr.training.params import TrainerParams

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, help='path to the checkpoint dir to resume from')

    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info(f"RESUMING TRAINING from {args.checkpoint_dir}")
    logger.info("=================================================================")

    with open(os.path.join(args.checkpoint_dir, 'trainer_params.json')) as f:
        d = json.load(f)

    trainer_params: TrainerParams = Scenario.trainer_params_from_dict(d)
    setup_log(trainer_params.checkpoint_dir, append=True)
    logger.info("trainer_params=" + trainer_params.to_json(indent=2))

    scenario_params = trainer_params.scenario_params
    scenario = Scenario(scenario_params)
    trainer = scenario.create_trainer(trainer_params, restore=True)
    trainer.train()


if __name__ == '__main__':
    main()
