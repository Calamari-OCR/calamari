from abc import ABC, abstractmethod
from tfaip.data.pipeline.definitions import Sample
from tfaip.data.pipeline.processor.dataprocessor import MappingDataProcessor, T
import logging


logger = logging.getLogger(__name__)


class ImageProcessor(MappingDataProcessor[T], ABC):
    def apply(self, sample: Sample) -> Sample:
        try:
            return sample.new_inputs(self._apply_single(sample.inputs, sample.meta))
        except Exception as e:
            logger.exception(e)
            logger.warning(
                "There was an unknown error when processing a line image. The line is skipped.\n"
                f"The error was caused by the line with meta data: {sample.meta}.\n"
                f"Please report this as an issue including the meta data, stack trace,  the respective "
                f"image file and call.\n"
                f"You can ignore this error if it occurs only very rarely, only this particular line will "
                f"be skipped."
            )
            return sample.new_invalid()

    def local_to_global_pos(self, x, meta):
        return x

    @abstractmethod
    def _apply_single(self, data, meta):
        return data
