from abc import abstractmethod, ABC

from tfaip.data.pipeline.definitions import Sample
from tfaip.data.pipeline.processor.dataprocessor import MappingDataProcessor, T

from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import Prediction


class TextProcessor(MappingDataProcessor[T], ABC):
    def apply(self, sample: Sample) -> Sample:
        targets: str = sample.targets
        outputs: str = sample.outputs
        meta = sample.meta
        if isinstance(outputs, Prediction):
            prediction: Prediction = outputs
            prediction.sentence = self._apply_single(prediction.sentence, meta)
        if isinstance(targets, dict) and "sentence" in targets:
            targets["sentence"] = self._apply_single(targets["sentence"], meta)
        if isinstance(outputs, dict) and "sentence" in outputs:
            outputs["sentence"] = self._apply_single(outputs["sentence"], meta)
        return sample

    @abstractmethod
    def _apply_single(self, txt: str, meta: dict) -> str:
        raise NotImplementedError
