from abc import abstractmethod, ABC

from tfaip.base.data.pipeline.dataprocessor import DataProcessor

from calamari_ocr.ocr.model.ctc_decoder.ctc_decoder import Prediction


class TextProcessor(DataProcessor, ABC):
    def apply(self, inputs, targets, meta: dict):
        if isinstance(targets, Prediction):
            prediction: Prediction = targets
            prediction.sentence = self._apply_single(prediction.sentence, meta)
            return inputs, targets
        elif isinstance(targets, dict):
            targets['sentence'] = self._apply_single(targets['sentence'], meta)
            return inputs, targets
        else:
            return inputs, self._apply_single(targets, meta)

    @abstractmethod
    def _apply_single(self, txt: str, meta: dict) -> str:
        raise NotImplementedError


class NoopTextProcessor(TextProcessor):
    def _apply_single(self, txt, meta):
        return txt
