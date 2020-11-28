from abc import ABC, abstractmethod
from tfaip.base.data.pipeline.dataprocessor import DataProcessor


class ImageProcessor(DataProcessor, ABC):
    def apply(self, inputs, targets, meta: dict):
        return self._apply_single(inputs, meta), targets

    def local_to_global_pos(self, x, meta):
        return x

    @abstractmethod
    def _apply_single(self, data, meta):
        return data


class NoopDataPreprocessor(ImageProcessor):
    def _apply_single(self, data, meta):
        return data
