from copy import deepcopy
from abc import ABC, abstractmethod

from calamari_ocr.proto import Prediction


class Voter(ABC):
    def __init__(self, text_postproc=None):
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
            p.sentence = self.text_postproc.apply(p.sentence)
        else:
            sentences = [pred.text_postproc.apply(p.sentence) for pred in predictions]

            if all([s == sentences[0] for s in sentences[1:]]):
                # usually all postproc should yield the same results
                p.sentence = sentences[0]
            else:
                # we need to vote again
                from calamari_ocr.ocr.voting import SequenceVoter
                sv = SequenceVoter()
                p.sentence = [c for c, _ in sv.process_text(sentences)]

        return p

    @abstractmethod
    def _apply_vote(self, predictions, p):
        pass

