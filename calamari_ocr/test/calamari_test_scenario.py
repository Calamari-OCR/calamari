from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr.model.layers.bilstm import BiLSTMLayerParams
from calamari_ocr.ocr.model.layers.conv2d import Conv2DLayerParams
from calamari_ocr.ocr.model.layers.dropout import DropoutLayerParams
from calamari_ocr.ocr.model.layers.layer import IntVec2D
from calamari_ocr.ocr.model.layers.pool2d import MaxPool2DLayerParams
from calamari_ocr.ocr.scenario import CalamariScenario, CalamariEnsembleScenario


class CalamariTestScenario(CalamariScenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.scenario.model.layers = [
            Conv2DLayerParams(filters=2),
            MaxPool2DLayerParams(pool_size=IntVec2D(4, 4)),
            BiLSTMLayerParams(hidden_nodes=2),
            DropoutLayerParams(rate=0.5),
        ]
        p.gen.setup.val.batch_size = 1
        p.gen.setup.val.num_processes = 1
        p.gen.setup.train.batch_size = 1
        p.gen.setup.train.num_processes = 1
        p.epochs = 1
        p.samples_per_epoch = 2
        p.scenario.data.pre_proc.run_parallel = False
        post_init(p)
        return p


class CalamariTestEnsembleScenario(CalamariEnsembleScenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.scenario.model.layers = [
            Conv2DLayerParams(filters=2),
            MaxPool2DLayerParams(pool_size=IntVec2D(4, 4)),
            BiLSTMLayerParams(hidden_nodes=2),
            DropoutLayerParams(rate=0.5),
        ]
        p.gen.setup.val.batch_size = 1
        p.gen.setup.val.num_processes = 1
        p.gen.setup.train.batch_size = 1
        p.gen.setup.train.num_processes = 1
        p.epochs = 1
        p.samples_per_epoch = 2
        p.scenario.data.pre_proc.run_parallel = False
        post_init(p)
        return p
