import os
import tempfile
import unittest
from glob import glob
from subprocess import check_call
import uuid

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestCommandList(unittest.TestCase):
    def test_command_line(self):
        pred_extension = '.' + str(uuid.uuid4()) + '.pred.txt'
        try:
            with tempfile.TemporaryDirectory() as d:
                check_call(["calamari-train",
                            '--train.images', os.path.join(this_dir, 'data', 'uw3_50lines', 'train', '*.bin.png'),
                            '--trainer.epochs', '1',
                            '--trainer.samples_per_epoch', '16',
                            '--trainer.output_dir', d,
                            '--n_augmentations', '2',
                            '--trainer.gen', 'SplitTrain',
                            '--network=cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5',
                            ])
                check_call(['calamari-predict',
                            '--data.images', os.path.join(this_dir, 'data', 'uw3_50lines', 'test', '*.bin.png'),
                            '--data.pred_extension', pred_extension,
                            '--checkpoint', os.path.join(d, 'best.ckpt.json'),
                            ])
                check_call(['calamari-eval',
                            '--gt.texts', os.path.join(this_dir, 'data', 'uw3_50lines', 'test', '*.gt.txt'),
                            '--gt.pred_extension', pred_extension,
                            ])
        finally:
            for file in glob(os.path.join(this_dir, 'data', 'uw3_50lines', 'test', '*' + pred_extension)):
                os.remove(file)
