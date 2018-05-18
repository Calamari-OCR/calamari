from abc import ABC, abstractmethod

from calamari_ocr.ocr.predictor import PredictionResult
from calamari_ocr.proto import Prediction


class Voter(ABC):
    def __abs__(self):
        super().__init__()

    def vote_prediction_result(self, prediction_results):
        if len(prediction_results) == 0:
            raise Exception("Empty prediction results")
        elif len(prediction_results) == 1:
            # no voting required
            return prediction_results[0].prediction
        else:
            return self.vote_prediction_result_tuple(tuple(prediction_results))

    def vote_prediction_results(self, prediction_results):
        return [self.vote_prediction_result(p) for p in prediction_results]

    def vote_prediction_result_tuple(self, predictions):
        p = Prediction()
        p.is_voted_result = True
        self._apply_vote(predictions, p)
        return p

    @abstractmethod
    def _apply_vote(self, predictions, p):
        pass
