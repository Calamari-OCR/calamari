from abc import abstractmethod, ABC

from tfaip.base.data.pipeline.dataprocessor import DataProcessor


class TextProcessor(DataProcessor, ABC):
    def apply(self, inputs, targets, meta: dict):
        return inputs, self._apply_single(targets, meta)

    @abstractmethod
    def _apply_single(self, txt: str, meta: dict) -> str:
        raise NotImplementedError


class NoopTextProcessor(TextProcessor):
    def _apply_single(self, txt, meta):
        return txt
