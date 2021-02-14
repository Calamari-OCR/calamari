import os

from paiargparse import PAIArgumentParser
from tfaip.util.logging import setup_log, logger

from calamari_ocr import __version__
from calamari_ocr.ocr.scenario import Scenario
from calamari_ocr.ocr.training.params import TrainerParams

logger = logger(__name__)


def run():
    main(parse_args())


def main(trainer_params):
    if trainer_params.checkpoint_dir:
        setup_log(trainer_params.checkpoint_dir, append=False)

    logger.info("trainer_params=" + trainer_params.to_json(indent=2))

    # create the trainer and run it
    trainer = Scenario.create_trainer(trainer_params)
    trainer.train()


def parse_args(args=None):
    parser = PAIArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    default_trainer_params = Scenario.default_trainer_params()
    parser.add_root_argument('trainer', default_trainer_params.__class__, default=default_trainer_params)

    return parser.parse_args(args).trainer


if __name__ == '__main__':
    run()


def setup_train_args(parser: PAIArgumentParser, omit=None):
    parser.add_root_argument('root', TrainerParams)

    return

    # required params for args
    if omit is None:
        omit = []

    if "files" not in omit:
        pass

    parser.add_argument("--network", type=str, default="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
                        help="The network structure")
    parser.add_argument("--display", type=int, default=1,
                        help="Frequency of how often an output shall occur during training [epochs]")
    parser.add_argument("--checkpoint_frequency", type=int, default=-1,
                        help="The frequency how often to write checkpoints during training [epochs]"
                             "If -1 (default), the early_stopping_frequency will be used. If 0 no checkpoints are written")

    if "output_dir" not in omit:
        parser.add_argument("--output_dir", type=str, default="",
                            help="Default directory where to store checkpoints and models")
    if "output_model_prefix" not in omit:
        parser.add_argument("--output_model_prefix", type=str, default="model_",
                            help="Prefix for storing checkpoints and models")

    # early stopping
    if "validation" not in omit:
        parser.add_argument("--use_train_as_val", action='store_true', default=False)

    parser.add_argument("--n_augmentations", type=float, default=0,
                        help="Amount of data augmentation per line (done before training). If this number is < 1 "
                             "the amount is relative.")

    # additional dataset args
    parser.add_argument("--debug", action='store_true')


def run(args):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        setup_log(args.output_dir, append=False)

    # =================================================================================================================
    # Trainer Params
    params.use_training_as_validation = args.use_train_as_val
