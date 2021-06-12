import unittest

from tfaip import PipelineMode, Sample

from calamari_ocr.ocr.dataset.textprocessors import TextRegularizerProcessorParams
from calamari_ocr.ocr.dataset.textprocessors.text_regularizer import ReplacementGroup


class TextTextRegularizer(unittest.TestCase):
    def test_space_quotes(self):
        n = TextRegularizerProcessorParams(
            replacement_groups=[ReplacementGroup.Quotes, ReplacementGroup.Spaces]
        ).create(None, mode=PipelineMode.TRAINING)
        self.assertEqual(n(Sample(targets="“Resolve quotes”")).targets, "''Resolve quotes''")
        self.assertEqual(n(Sample(targets="  “Resolve   spaces  ”   ")).targets, "''Resolve spaces ''")

    def test_none(self):
        n = TextRegularizerProcessorParams(replacement_groups=[ReplacementGroup.No]).create(
            None, mode=PipelineMode.TRAINING
        )
        self.assertNotEqual(n(Sample(targets="“Resolve quotes”")).targets, "''Resolve quotes''")
        self.assertNotEqual(n(Sample(targets="  “Resolve   spaces  ”   ")).targets, "''Resolve spaces ''")
