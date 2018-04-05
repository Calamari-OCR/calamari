from abc import ABC, abstractmethod
from tqdm import tqdm

from calamari_ocr.utils import parallel_map


class TextProcessor(ABC):
    def __init__(self):
        super().__init__()

    def apply(self, txts, processes=1, progress_bar=False):
        if isinstance(txts, str):
            return self._apply_single(txts)
        elif isinstance(txts, list):
            if len(txts) == 0:
                return []

            return parallel_map(self._apply_single, txts, desc="Text Preprocessing", processes=processes, progress_bar=progress_bar)
        else:
            raise Exception("Unknown instance of txts: {}. Supported list and str".format(type(txts)))

    @abstractmethod
    def _apply_single(self, txt):
        pass


class NoopTextProcessor(TextProcessor):
    def __init__(self):
        super().__init__()

    def _apply_single(self, txt):
        return txt


class MultiTextProcessor(TextProcessor):
    def __init__(self, processors=[]):
        super().__init__()
        self.sub_processors = processors

    def add(self, processor):
        self.sub_processors.append(processor)

    def _apply_single(self, txt):
        for proc in self.sub_processors:
            txt = proc._apply_single(txt)

        return txt

    def child_by_type(self, t):
        for proc in self.sub_processors:
            if type(proc) == t:
                return proc

        return None
