from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import multiprocessing

from calamari_ocr.utils import parallel_map


class DataPreprocessor(ABC):
    def __init__(self):
        super().__init__()

    def apply(self, data, processes=1, progress_bar=False, max_tasks_per_child=100):
        if isinstance(data, np.ndarray):
            return self._apply_single(data)
        elif isinstance(data, list):
            if len(data) == 0:
                return []

            return parallel_map(self._apply_single, data, desc="Data Preprocessing",
                                processes=processes, progress_bar=progress_bar, max_tasks_per_child=max_tasks_per_child)
        else:
            raise Exception("Unknown instance of txts: {}. Supported list and str".format(type(data)))

    def local_to_global_pos(self, x, params):
        return x

    @abstractmethod
    def _apply_single(self, data):
        return data, None


class NoopDataPreprocessor(DataPreprocessor):
    def __init__(self):
        super().__init__()

    def _apply_single(self, data):
        return data, None


class MultiDataProcessor(DataPreprocessor):
    def __init__(self, processors=[]):
        super().__init__()
        self.sub_processors = processors

    def add(self, processor):
        self.sub_processors.append(processor)

    def _apply_single(self, data):
        stacked_params = []
        for proc in self.sub_processors:
            data, params = proc._apply_single(data)
            stacked_params.append(params)

        return data, stacked_params

    def local_to_global_pos(self, x, params):
        assert(len(params) == len(self.sub_processors))
        for i in reversed(range(len(self.sub_processors))):
            x = self.sub_processors[i].local_to_global_pos(x, params[i])

        return x
