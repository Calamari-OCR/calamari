import os
import tempfile
import unittest
from glob import glob
from subprocess import check_call, run
import uuid

this_dir = os.path.dirname(os.path.realpath(__file__))


def assert_run(args):
    # cannot use check_call, because it blocks if the stdout FIFO is full
    # (i.e. there is sufficient data on the OS pipe buffer) and it is not read from
    # also, we want to see what failed in case of error
    # check_call(args)
    ret = run(args, encoding="utf-8", capture_output=True)
    assert ret.returncode == 0


class TestCommandList(unittest.TestCase):
    def test_resume_training(self):
        with tempfile.TemporaryDirectory() as d:
            # Test training
            assert_run(
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
            assert_run(
                ["calamari-resume-training", os.path.join(d, "checkpoint", "checkpoint_0001", "trainer_params.json")]
            )

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as d:
            # Test training
            assert_run(
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
            assert_run(
                [
                    "calamari-predict",
                    "--data.images",
                    os.path.join(this_dir, "data", "uw3_50lines", "test", "*.bin.png"),
                    "--output_dir",
                    d,
                    "--checkpoint",
                    os.path.join(d, "best.ckpt.json"),
                ]
            )
            # Test voting
            assert_run(
                [
                    "calamari-predict",
                    "--data.images",
                    os.path.join(this_dir, "data", "uw3_50lines", "test", "*.bin.png"),
                    "--output_dir",
                    d,
                    "--checkpoint",
                    os.path.join(d, "best.ckpt.json"),
                    os.path.join(d, "best.ckpt.json"),
                ]
            )
            # Test evaluation
            assert_run(
                [
                    "calamari-eval",
                    "--gt.texts",
                    os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt"),
                ]
            )
