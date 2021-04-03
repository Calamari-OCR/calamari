import unittest

from tensorflow import keras

from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.test.test_train_file import uw3_trainer_params
from calamari_ocr.test.test_train_pagexml import default_trainer_params


class TestTrainConsistency(unittest.TestCase):
    def test_train_val_error(self):
        trainer_params = uw3_trainer_params()
        trainer_params.force_eager = True
        trainer_params.epochs = 1
        trainer_params.random_seed = 9412
        trainer_params.samples_per_epoch = 6
        trainer_params.gen.setup.train.batch_size = trainer_params.gen.setup.val.batch_size = 6
        trainer_params.learning_rate.lr = 0
        trainer_params.gen.train.images = trainer_params.gen.train.images[:trainer_params.samples_per_epoch]
        del trainer_params.scenario.model.layers[-1]  # no dropout
        trainer = CalamariScenario.create_trainer(trainer_params)

        class FinalLogsCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.train_logs = {}

            def on_train_end(self, logs=None):
                self.train_logs = logs

        cb = FinalLogsCallback()
        trainer.train(callbacks=[cb])

        assert len(cb.train_logs) > 0

        for k, v in cb.train_logs.items():
            if k.startswith("val_"):
                continue
            self.assertEqual(v, cb.train_logs['val_' + k])
