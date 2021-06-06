import os
import tempfile
import unittest
from glob import glob
from subprocess import check_call
import uuid

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestCommandList(unittest.TestCase):
    def test_resume_training(self):
        with tempfile.TemporaryDirectory() as d:
            # Test training
            check_call(
                [
                    "calamari-train",
                    "--train.images",
                    os.path.join(this_dir, "data", "uw3_50lines", "train", "*.bin.png"),
                    "--trainer.epochs",
                    "1",
                    "--trainer.samples_per_epoch",
                    "16",
                    "--trainer.output_dir",
                    d,
                    "--n_augmentations",
                    "2",
                    "--trainer.gen",
                    "SplitTrain",
                    "--network=cnn=4:3x3,pool=4x4,lstm=10,dropout=0.5",
                ]
            )
            check_call(
                ["calamari-resume-training", os.path.join(d, "checkpoint", "checkpoint_0001", "trainer_params.json")]
            )

    def test_command_line(self):
        pred_extension = "." + str(uuid.uuid4()) + ".pred.txt"
        try:
            with tempfile.TemporaryDirectory() as d:
                # Test training
                check_call(
                    [
                        "calamari-train",
                        "--train.images",
                        os.path.join(this_dir, "data", "uw3_50lines", "train", "*.bin.png"),
                        "--trainer.epochs",
                        "1",
                        "--trainer.samples_per_epoch",
                        "16",
                        "--trainer.output_dir",
                        d,
                        "--n_augmentations",
                        "2",
                        "--trainer.gen",
                        "SplitTrain",
                        "--network=cnn=4:3x3,pool=4x4,lstm=10,dropout=0.5",
                    ]
                )
                # Test single model prediction
                check_call(
                    [
                        "calamari-predict",
                        "--data.images",
                        os.path.join(this_dir, "data", "uw3_50lines", "test", "*.bin.png"),
                        "--data.pred_extension",
                        pred_extension,
                        "--checkpoint",
                        os.path.join(d, "best.ckpt.json"),
                    ]
                )
                # Test voting
                check_call(
                    [
                        "calamari-predict",
                        "--data.images",
                        os.path.join(this_dir, "data", "uw3_50lines", "test", "*.bin.png"),
                        "--data.pred_extension",
                        pred_extension,
                        "--checkpoint",
                        os.path.join(d, "best.ckpt.json"),
                        os.path.join(d, "best.ckpt.json"),
                    ]
                )
                # Test evaluation
                check_call(
                    [
                        "calamari-eval",
                        "--gt.texts",
                        os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt"),
                        "--gt.pred_extension",
                        pred_extension,
                    ]
                )
        finally:
            for file in glob(os.path.join(this_dir, "data", "uw3_50lines", "test", "*" + pred_extension)):
                os.remove(file)
