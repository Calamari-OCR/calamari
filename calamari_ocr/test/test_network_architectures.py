import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
from calamari_ocr.ocr.model.layers.concat import ConcatLayerParams
from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
from calamari_ocr.ocr.model.layers.dilatedblock import DilatedBlockLayerParams
from calamari_ocr.ocr.model.layers.layer import IntVec2D
from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
from calamari_ocr.ocr.model.layers.transposedconv2d import TransposedConv2DLayerParams
from calamari_ocr.ocr.model.params import default_layers
from calamari_ocr.ocr.training.params import parse_network_param
from calamari_ocr.scripts.train import main, parse_args
from calamari_ocr.test.test_train_file import uw3_trainer_params


class TestNetworkArchitectures(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = 10024

    def tearDown(self) -> None:
        clear_session()

    def test_predefined_networks(self):
        self.assertListEqual(default_layers(), parse_network_param("def"))

    def test_default_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = default_layers()

        # make some minor modifications
        trainer_params.scenario.model.layers[0].activation = "leaky_relu"  # test that leaky_relu is a valid layer

        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_pure_lstm_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            BiLSTMLayerParams(hidden_nodes=10),
            BiLSTMLayerParams(hidden_nodes=20),
        ]
        post_init(trainer_params)
        cmd_line_trainer_params = parse_args(["--network", "lstm=10,lstm=20"])
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        cmd_line_trainer_params = parse_args(
            [
                "--model.layers",
                "BiLSTM",
                "BiLSTM",
                "--model.layers.0.hidden_nodes",
                "10",
                "--model.layers.1.hidden_nodes",
                "20",
            ]
        )
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_pure_cnn_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            Conv2DLayerParams(filters=10),
            MaxPool2DLayerParams(),
            Conv2DLayerParams(filters=20, strides=IntVec2D(2, 2), kernel_size=IntVec2D(4, 4)),
            Conv2DLayerParams(filters=30),
        ]
        post_init(trainer_params)
        cmd_line_trainer_params = parse_args(["--network", "conv=10,pool=2x2,conv=20:4x4:2x2,conv=30"])
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        cmd_line_trainer_params = parse_args(
            [
                "--model.layers",
                "Conv",
                "Pool",
                "Conv",
                "Conv",
                "--model.layers.0.filters",
                "10",
                "--model.layers.2.filters",
                "20",
                "--model.layers.2.kernel_size.x",
                "4",
                "--model.layers.2.kernel_size.y",
                "4",
                "--model.layers.2.strides.x",
                "2",
                "--model.layers.2.strides.y",
                "2",
                "--model.layers.3.filters",
                "30",
            ]
        )
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_dilated_block_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            Conv2DLayerParams(filters=10),
            MaxPool2DLayerParams(strides=IntVec2D(2, 2)),
            DilatedBlockLayerParams(filters=10),
            DilatedBlockLayerParams(filters=10),
            Conv2DLayerParams(filters=10),
        ]
        post_init(trainer_params)
        cmd_line_trainer_params = parse_args(["--network", "conv=10,pool=2x2:2x2,db=10:2,db=10:2,conv=10"])
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        cmd_line_trainer_params = parse_args(
            [
                "--model.layers",
                "Conv",
                "Pool",
                "DilatedBlock",
                "DilatedBlock",
                "Conv",
                "--model.layers.0.filters",
                "10",
                "--model.layers.1.strides",
                "2",
                "2",
                "--model.layers.2.filters",
                "10",
                "--model.layers.3.filters",
                "10",
                "--model.layers.4.filters",
                "10",
            ]
        )
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_transposed_conv_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            Conv2DLayerParams(filters=10),
            MaxPool2DLayerParams(),
            DilatedBlockLayerParams(filters=10),
            TransposedConv2DLayerParams(filters=10),
            Conv2DLayerParams(filters=10),
            BiLSTMLayerParams(hidden_nodes=10),
        ]
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_concat_cnn_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            Conv2DLayerParams(filters=10),
            MaxPool2DLayerParams(),
            DilatedBlockLayerParams(filters=10),
            TransposedConv2DLayerParams(filters=10),
            ConcatLayerParams(concat_indices=[1, 4]),  # corresponds to output of first and fourth layer
            Conv2DLayerParams(filters=10),
            BiLSTMLayerParams(hidden_nodes=10),
        ]
        post_init(trainer_params)
        cmd_line_trainer_params = parse_args(
            ["--network", "conv=10,pool=2x2,db=10:2,tconv=10,concat=1:4,conv=10,lstm=10"]
        )
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        cmd_line_trainer_params = parse_args(
            [
                "--model.layers",
                "Conv",
                "Pool",
                "DilatedBlock",
                "TConv",
                "Concat",
                "Conv",
                "BiLSTM",
                "--model.layers.0.filters",
                "10",
                "--model.layers.2.filters",
                "10",
                "--model.layers.3.filters",
                "10",
                "--model.layers.4.concat_indices",
                "1",
                "4",
                "--model.layers.5.filters",
                "10",
                "--model.layers.6.hidden_nodes",
                "10",
            ]
        )
        self.assertDictEqual(trainer_params.scenario.model.to_dict(), cmd_line_trainer_params.scenario.model.to_dict())
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)
