import os
import unittest

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestPageXML(unittest.TestCase):
    def run_dataset_viewer(self, add_args):
        from calamari_ocr.scripts.dataset_viewer import main

        main(add_args + ["--no_plot"])

    def test_cut_modes(self):
        images = os.path.join(this_dir, "data", "avicanon_pagexml", "*.nrm.png")
        self.run_dataset_viewer(["--gen", "PageXML", "--gen.images", images, "--gen.cut_mode", "BOX"])
        self.run_dataset_viewer(["--gen", "PageXML", "--gen.images", images, "--gen.cut_mode", "MBR"])
