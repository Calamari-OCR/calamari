from abc import ABC, abstractmethod
import numpy as np

from calamari_ocr.utils import parallel_map


class DataPreprocessor(ABC):
    @classmethod
    def from_dict(cls, d: dict):
        from calamari_ocr.ocr.data_processing import data_processor_cls
        real_cls = data_processor_cls(d['type'])
        if real_cls != cls:
            return real_cls.from_dict(d)

        return cls(**{k: v for k, v in d.items() if k != 'type'})

    def to_dict(self) -> dict:
        return {'type': self.__class__.__name__}

    def __init__(self):
        super().__init__()

    def apply(self, data, processes=1, progress_bar=False, max_tasks_per_child=100):
        if isinstance(data, np.ndarray):
            return self._apply_single(data)
        elif isinstance(data, list) or isinstance(data, tuple):
            if len(data) == 0:
                return []

            return parallel_map(self._apply_single, data, desc="Data Preprocessing",
                                processes=processes, progress_bar=progress_bar, max_tasks_per_child=max_tasks_per_child)
        else:
            raise Exception("Unknown instance of data: {}. Supported list and str".format(type(data)))

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
    def to_dict(self) -> dict:
        d = super(MultiDataProcessor, self).to_dict()
        d['processors'] = [p.to_dict() for p in self.sub_processors]
        return d

    @classmethod
    def from_dict(cls, d: dict):
        d['processors'] = [DataPreprocessor.from_dict(p) for p in d['processors']]
        return super(MultiDataProcessor, cls).from_dict(d)

    def __init__(self, processors=None):
        super().__init__()
        self.sub_processors = processors if processors else []

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
