from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional

from tfaip.data.pipeline.definitions import Sample

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import Prediction


class Voter(ABC):
    def __init__(self, text_postproc: Optional[TextProcessor] = None):
        super().__init__()
        self.text_postproc = text_postproc

    def vote_prediction_result(self, prediction_results):
        if len(prediction_results) == 0:
            raise Exception("Empty prediction results")
        elif len(prediction_results) == 1:
            # no voting required
            return deepcopy(prediction_results[0].prediction)
        else:
            return self.vote_prediction_result_tuple(tuple(prediction_results))

    def vote_prediction_results(self, prediction_results):
        return [self.vote_prediction_result(p) for p in prediction_results]

    def vote_prediction_result_tuple(self, predictions):
        p = Prediction()
        p.is_voted_result = True
        self._apply_vote(predictions, p)

        # postprocessing after voting
        # option 1: Use custom text postprocessor
        # option 2: (Not implemented) Use only the first text postprocessor
        # option 3: Apply all known postprocessors and apply a sequence voting if different results are received
        if self.text_postproc:
            p.sentence = self.text_postproc.apply_on_sample(Sample(inputs="", outputs=p.sentence)).outputs
        else:
            sentences = [
                pred.text_postproc.apply_on_sample(Sample(inputs="", outputs=p.sentence)).outputs
                for pred in predictions
            ]

            if all([s == sentences[0] for s in sentences[1:]]):
                # usually all postproc should yield the same results
                p.sentence = sentences[0]
            else:
                # we need to vote again
                from calamari_ocr.ocr.voting import SequenceVoter

                sv = SequenceVoter()
                p.sentence = "".join([c for c, _ in sv.process_text(sentences)])

        p.avg_char_probability = 0
        for pos in p.positions:
            if len(pos.chars) > 0:
                p.avg_char_probability += pos.chars[0].probability
        p.avg_char_probability /= len(p.positions) if len(p.positions) > 0 else 1

        return p

    @abstractmethod
    def _apply_vote(self, predictions, p):
        pass
