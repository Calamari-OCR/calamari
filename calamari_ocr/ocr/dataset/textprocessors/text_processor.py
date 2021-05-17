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
            return sample
        elif isinstance(targets, dict) and "sentence" in targets:
            targets["sentence"] = self._apply_single(targets["sentence"], meta)
            return sample
        elif isinstance(outputs, dict) and "sentence" in outputs:
            outputs["sentence"] = self._apply_single(outputs["sentence"], meta)
            return sample
        else:
            if targets:
                sample = sample.new_targets(self._apply_single(targets, meta))
            if outputs:
                sample = sample.new_outputs(self._apply_single(outputs, meta))
            return sample

    @abstractmethod
    def _apply_single(self, txt: str, meta: dict) -> str:
        raise NotImplementedError
