from contextlib import ExitStack
from typing import TYPE_CHECKING, Dict

from paiargparse import PAIArgumentParser
from tfaip.util.logging import logger, WriteToLogFile
from tfaip.util.typing import AnyNumpy

from calamari_ocr import __version__

if TYPE_CHECKING:
    from calamari_ocr.ocr.training.params import TrainerParams

logger = logger(__name__)


def run():
    main(parse_args())


def main(trainer_params: "TrainerParams") -> Dict[str, AnyNumpy]:
    with ExitStack() as stack:
        if trainer_params.output_dir:
            stack.enter_context(WriteToLogFile(trainer_params.output_dir, append=False))

        logger.info("trainer_params=" + trainer_params.to_json(indent=2))

        # create the trainer and run it
        trainer = trainer_params.scenario.cls().create_trainer(trainer_params)
        return trainer.train()


def parse_args(args=None):
    from calamari_ocr.ocr.scenario import CalamariScenario

    parser = PAIArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s v" + __version__)

    default_trainer_params = CalamariScenario.default_trainer_params()
    parser.add_root_argument("trainer", default_trainer_params.__class__, default=default_trainer_params)
    params = parser.parse_args(args).trainer
    return params


if __name__ == "__main__":
    run()
