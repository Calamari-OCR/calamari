from abc import ABC, abstractmethod
import numpy as np
from tfaip.base.data.pipeline.dataprocessor import DataProcessor


class ImageProcessor(DataProcessor, ABC):
    def apply(self, inputs, targets, meta: dict):
        return self._apply_single(inputs, meta), targets

    # TODO: parallelize
    def _apply(self, data, processes=1, progress_bar=False, max_tasks_per_child=100):
        if isinstance(data, np.ndarray):
            return self._apply_single(data)
        elif isinstance(data, list) or isinstance(data, tuple):
            if len(data) == 0:
                return []

            return parallel_map(self._apply_single, data, desc="Data Preprocessing",
                                processes=processes, progress_bar=progress_bar, max_tasks_per_child=max_tasks_per_child)
        else:
            raise Exception("Unknown instance of data: {}. Supported list and str".format(type(data)))

    def local_to_global_pos(self, x, meta):
        return x

    @abstractmethod
    def _apply_single(self, data, meta):
        return data


class NoopDataPreprocessor(ImageProcessor):
    def _apply_single(self, data, meta):
        return data
