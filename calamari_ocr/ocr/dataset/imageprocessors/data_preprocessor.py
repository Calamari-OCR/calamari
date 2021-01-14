from abc import ABC, abstractmethod
from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import Sample


class ImageProcessor(DataProcessor, ABC):
    def apply(self, sample: Sample) -> Sample:
        return sample.new_inputs(self._apply_single(sample.inputs, sample.meta))

    def local_to_global_pos(self, x, meta):
        return x

    @abstractmethod
    def _apply_single(self, data, meta):
        return data


class NoopDataPreprocessor(ImageProcessor):
    def _apply_single(self, data, meta):
        return data
