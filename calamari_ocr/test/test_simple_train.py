import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr import DataSetType
from calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import default_image_processors
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import FinalPreparation
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeightProcessor
from calamari_ocr.scripts.train import run
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


class Attrs():
    def __init__(self):
        self.dataset = DataSetType.FILE
        self.gt_extension = DataSetType.gt_extension(self.dataset)
        self.files = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")])
        self.seed = 24
        self.backend = "tensorflow"
        self.network = "cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5"
        self.line_height = 48
        self.pad = 16
        self.num_threads = 1
        self.display = 1
        self.batch_size = 1
        self.checkpoint_frequency = 1000
        self.epochs = 1
        self.samples_per_epoch = 8
        self.stats_size = 100
        self.no_skip_invalid_gt = False
        self.no_progress_bars = True
        self.output_dir = None
        self.output_model_prefix = "uw3_50lines"
        self.bidi_dir = None
        self.weights = None
        self.ema_weights = False
        self.whitelist_files = []
        self.whitelist = []
        self.gradient_clipping_norm = 5
        self.validation = None
        self.validation_dataset = DataSetType.FILE
        self.validation_extension = None
        self.validation_split_ratio = None
        self.early_stopping_frequency = -1
        self.early_stopping_nbest = 10
        self.early_stopping_at_accuracy = 0.99
        self.early_stopping_best_model_prefix = "uw3_50lines_best"
        self.early_stopping_best_model_output_dir = self.output_dir
        self.n_augmentations = 0
        self.num_inter_threads = 0
        self.num_intra_threads = 0
        self.text_regularization = ["extended"]
        self.text_normalization = "NFC"
        self.text_generator_params = None
        self.line_generator_params = None
        self.pagexml_text_index = 0
        self.text_files = None
        self.only_train_on_augmented = False
        self.data_preprocessing = [p.name for p in default_image_processors()]
        self.shuffle_buffer_size = 1000
        self.keep_loaded_codec = False
        self.train_data_on_the_fly = False
        self.validation_data_on_the_fly = False
        self.no_auto_compute_codec = False
        self.dataset_pad = 0
        self.debug = False
        self.train_verbose = True
        self.use_train_as_val = False
        self.ensemble = -1
        self.masking_mode = 1


class TestSimpleTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        args = Attrs()
        with tempfile.TemporaryDirectory() as d:
            args.output_dir = d
            run(args)

    def test_train_without_center_normalizer(self):
        args = Attrs()
        args.data_preprocessing = [
            DataRangeNormalizer.__name__,
            ScaleToHeightProcessor.__name__,
            FinalPreparation.__name__,
        ]
        with tempfile.TemporaryDirectory() as d:
            args.output_dir = d
            run(args)


if __name__ == "__main__":
    unittest.main()
