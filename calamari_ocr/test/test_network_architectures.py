import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
from calamari_ocr.ocr.model.layers.concat import ConcatLayerParams
from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
from calamari_ocr.ocr.model.layers.dilatedblock import DilatedBlockLayerParams
from calamari_ocr.ocr.model.layers.layer import IntVec2D
from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
from calamari_ocr.ocr.model.layers.transposedconv2d import TransposedConv2DLayerParams
from calamari_ocr.ocr.model.params import default_layers
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import uw3_trainer_params


class TestNetworkArchitectures(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_default_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = default_layers()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_pure_lstm_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            BiLSTMLayerParams(hidden_nodes=10),
            BiLSTMLayerParams(hidden_nodes=20),
        ]
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
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_dilated_block_architecture(self):
        trainer_params = uw3_trainer_params()
        trainer_params.scenario.model.layers = [
            Conv2DLayerParams(filters=10),
            MaxPool2DLayerParams(),
            DilatedBlockLayerParams(filters=10),
            DilatedBlockLayerParams(filters=10),
            Conv2DLayerParams(filters=10),
        ]
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
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

