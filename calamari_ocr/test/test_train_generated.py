import json
import os
import tempfile
import unittest

import pytest

from calamari_ocr.ocr.dataset.datareader.generated_line_dataset import (
    TextGeneratorParams,
    LineGeneratorParams,
)
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset.params import (
    GeneratedLineDatasetParams,
)
from calamari_ocr.scripts.train import main
from calamari_ocr.test.calamari_test_scenario import CalamariTestScenario

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_trainer_params():
    p = CalamariTestScenario.default_trainer_params()

    with open(os.path.join(this_dir, "data", "line_generation_config", "text_gen_params.json")) as f:
        text_gen_params = TextGeneratorParams.from_dict(json.load(f))

    with open(os.path.join(this_dir, "data", "line_generation_config", "line_gen_params.json")) as f:
        line_gen_params = LineGeneratorParams.from_dict(json.load(f))

    p.codec.include_files = os.path.join(this_dir, "data", "line_generation_config", "whilelist.txt")
    p.codec.auto_compute = False

    p.gen.train = GeneratedLineDatasetParams(
        lines_per_epoch=10,
        preload=False,
        text_generator=text_gen_params,
        line_generator=line_gen_params,
    )
    p.gen.val = GeneratedLineDatasetParams(
        lines_per_epoch=10,
        preload=False,
        text_generator=text_gen_params,
        line_generator=line_gen_params,
    )

    p.gen.setup.val.batch_size = 1
    p.gen.setup.val.num_processes = 1
    p.gen.setup.train.batch_size = 1
    p.gen.setup.train.num_processes = 1
    p.epochs = 1
    p.scenario.data.pre_proc.run_parallel = False
    p.gen.__post_init__()
    p.scenario.data.__post_init__()
    p.scenario.__post_init__()
    p.__post_init__()
    return p


@pytest.mark.skipif(os.name != "posix", reason="Do not run on windows due to missing font.")
class TestTrainGenerated(unittest.TestCase):
    def test_train(self):
        trainer_params = default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)
