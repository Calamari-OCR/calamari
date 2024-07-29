import os
import unittest

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestDatasetViewer(unittest.TestCase):
    def run_dataset_viewer(self, add_args):
        from calamari_ocr.scripts.dataset_viewer import main

        main(add_args + ["--no_plot"])
        main(add_args + ["--no_plot", "--as_validation"])
        main(add_args + ["--no_plot", "--as_predict"])
        main(add_args + ["--no_plot", "preload"])
        main(add_args + ["--no_plot", "--select", "0", "2"])
        main(add_args + ["--no_plot", "--n_augmentations", "5"])

    def test_dataset_viewer_files(self):
        images = os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")
        self.run_dataset_viewer(["--gen.images", images])

    def test_dataset_viewer_pagexml(self):
        images = os.path.join(this_dir, "data", "avicanon_pagexml", "*.nrm.png")
        self.run_dataset_viewer(["--gen", "PageXML", "--gen.images", images])

    def test_dataset_viewer_abbyyxml(self):
        images = os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg")
        self.run_dataset_viewer(["--gen", "Abbyy", "--gen.images", images])

    def test_dataset_viewer_hdf5(self):
        files = os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")
        self.run_dataset_viewer(["--gen", "Hdf5", "--gen.files", files])


class TestDatasetStatistics(unittest.TestCase):
    def run_dataset_statistics(self, add_args):
        from calamari_ocr.scripts.dataset_statistics import main

        main(add_args)

    def test_dataset_viewer_files(self):
        images = os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")
        self.run_dataset_statistics(["--data.images", images])

    def test_dataset_viewer_pagexml(self):
        images = os.path.join(this_dir, "data", "avicanon_pagexml", "*.nrm.png")
        self.run_dataset_statistics(["--data", "PageXML", "--data.images", images])

    def test_dataset_viewer_abbyyxml(self):
        images = os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg")
        self.run_dataset_statistics(["--data", "Abbyy", "--data.images", images])

    def test_dataset_viewer_hdf5(self):
        files = os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")
        self.run_dataset_statistics(["--data", "Hdf5", "--data.files", files])
