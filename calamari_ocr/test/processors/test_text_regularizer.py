import unittest

from tfaip import PipelineMode, Sample

from calamari_ocr.ocr.dataset.textprocessors import TextRegularizerProcessorParams


class TextTextRegularizer(unittest.TestCase):
    def test_space_quotes(self):
        n = TextRegularizerProcessorParams(
            rulesets=["quotes", "spaces"],
            rulegroups=[],
        ).create(None, PipelineMode.TARGETS)

        self.assertEqual(n(Sample(targets="“Resolve quotes”")).targets, "''Resolve quotes''")
        self.assertEqual(n(Sample(targets="  “Resolve   spaces  ”   ")).targets, "''Resolve spaces ''")

    def test_none(self):
        n = TextRegularizerProcessorParams(
            rulesets=[],
            rulegroups=["no"],
        ).create(None, PipelineMode.TARGETS)
        self.assertNotEqual(n(Sample(targets="“Resolve quotes”")).targets, "''Resolve quotes''")
        self.assertNotEqual(n(Sample(targets="  “Resolve   spaces  ”   ")).targets, "''Resolve spaces ''")

    def test_rule_sets(self):
        def assert_str(p_, in_s, out_s):
            computed = list(p_.apply_on_samples([Sample(targets=in_s)]))[0].targets
            self.assertEqual(out_s, computed, f"Wrong output for string {in_s}.")

        p = TextRegularizerProcessorParams(
            rulesets=[],
            rulegroups=[],
        ).create(None, PipelineMode.TARGETS)

        assert_str(p, "This \"''\"`is a  test..", "This \"''\"`is a  test..")

        p = TextRegularizerProcessorParams(
            rulesets=["spaces"],
            rulegroups=[],
        ).create(None, PipelineMode.TARGETS)

        assert_str(p, "This \"''\"`is a  test..", "This \"''\"`is a test..")

        p = TextRegularizerProcessorParams(
            rulesets=["quotes"],
            rulegroups=[],
        ).create(None, PipelineMode.TARGETS)

        assert_str(p, "This \"''\"`is a  test..", "This '''''''is a  test..")

        p = TextRegularizerProcessorParams(
            rulesets=["punctuation"],
            rulegroups=[],
        ).create(None, PipelineMode.TARGETS)

        assert_str(p, "This is .  . a test..", "This is. . a test. .")

        p = TextRegularizerProcessorParams(
            rulesets=[],
            rulegroups=["all"],
        ).create(None, PipelineMode.TARGETS)

        assert_str(p, "This is .  . a  test..", "This is. . a test. .")
