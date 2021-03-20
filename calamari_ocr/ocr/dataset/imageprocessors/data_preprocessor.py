from abc import ABC, abstractmethod
from tfaip.data.pipeline.definitions import Sample
from tfaip.data.pipeline.processor.dataprocessor import MappingDataProcessor, T


class ImageProcessor(MappingDataProcessor[T], ABC):
    def apply(self, sample: Sample) -> Sample:
        return sample.new_inputs(self._apply_single(sample.inputs, sample.meta))

    def local_to_global_pos(self, x, meta):
        return x

    @abstractmethod
    def _apply_single(self, data, meta):
        return data
