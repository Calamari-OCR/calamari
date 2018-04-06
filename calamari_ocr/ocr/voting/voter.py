from abc import ABC, abstractmethod

from calamari_ocr.ocr.predictor import PredictionResult


class Voter(ABC):
    def __abs__(self):
        super().__init__()

    def vote_prediction_results(self, prediction_results):
        if len(prediction_results) == 0:
            raise Exception("Empty prediction results")
        elif len(prediction_results) == 1:
            # no voting required
            return [p.sentence for p in prediction_results[0]]
        else:
            return [self.vote_prediction_result_tuple(tuple(r)) for r in zip(*prediction_results)]

    def vote_prediction_result_tuple(self, prediction_result_tuple):
        return self._apply_vote(prediction_result_tuple)

    @abstractmethod
    def _apply_vote(self, prediction_result_tuple):
        pass
