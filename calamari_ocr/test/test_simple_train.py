import unittest
import os

from calamari_ocr.ocr import DataSetType
from calamari_ocr.proto import DataPreprocessorParams
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
        self.max_iters = 1000
        self.stats_size = 100
        self.no_skip_invalid_gt = False
        self.no_progress_bars = True
        self.output_dir = os.path.join(this_dir, "test_models")
        self.output_model_prefix = "uw3_50lines"
        self.bidi_dir = None
        self.weights = None
        self.whitelist_files = []
        self.whitelist = []
        self.gradient_clipping_mode = "AUTO"
        self.gradient_clipping_const = 0
        self.validation = None
        self.validation_dataset = DataSetType.FILE
        self.validation_extension = None
        self.early_stopping_frequency = -1
        self.early_stopping_nbest = 10
        self.early_stopping_best_model_prefix = "uw3_50lines_best"
        self.early_stopping_best_model_output_dir = self.output_dir
        self.n_augmentations = 0
        self.fuzzy_ctc_library_path = ""
        self.num_inter_threads = 0
        self.num_intra_threads = 0
        self.text_regularization = ["extended"]
        self.text_normalization = "NFC"
        self.text_generator_params = None
        self.line_generator_params = None
        self.pagexml_text_index = 0
        self.text_files = None
        self.only_train_on_augmented = False
        self.data_preprocessing = [DataPreprocessorParams.DEFAULT_NORMALIZER]
        self.shuffle_buffer_size = 1000
        self.keep_loaded_codec = False
        self.train_data_on_the_fly = False
        self.validation_data_on_the_fly = False
        self.no_auto_compute_codec = False


class TestSimpleTrain(unittest.TestCase):
    def test_simple_train(self):
        args = Attrs()
        run(args)


if __name__ == "__main__":
    unittest.main()