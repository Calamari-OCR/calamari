from paiargparse import PAIArgumentParser
from tfaip.util.logging import logger

from calamari_ocr.ocr.training.cross_fold_trainer import (
    CrossFoldTrainer,
    CrossFoldTrainerParams,
)

logger = logger(__name__)


def run():
    return main(parse_args())


def parse_args(args=None):
    parser = PAIArgumentParser()
    parser.add_root_argument("root", CrossFoldTrainerParams, CrossFoldTrainerParams())
    params: CrossFoldTrainerParams = parser.parse_args(args).root
    # TODO: add the training args (omit those params, that are set by the cross fold training)
    # setup_train_args(parser, omit=["files", "validation", "weights",
    #                              "early_stopping_best_model_output_dir", "early_stopping_best_model_prefix",
    #                              "output_dir"])
    return params


def main(params):
    trainer = CrossFoldTrainer(params)
    logger.info("Running cross fold train with params")
    logger.info(params.to_json(indent=2))
    trainer.run()


if __name__ == "__main__":
    run()
