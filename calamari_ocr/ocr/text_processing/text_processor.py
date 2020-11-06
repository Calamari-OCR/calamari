from abc import ABC, abstractmethod

from calamari_ocr.utils import parallel_map


class TextProcessor(ABC):
    @classmethod
    def from_dict(cls, d: dict):
        from calamari_ocr.ocr.text_processing import text_processor_cls
        real_cls = text_processor_cls(d['type'])
        if real_cls != cls:
            return real_cls.from_dict(d)

        return cls(**{k: v for k, v in d.items() if k != 'type'})

    def to_dict(self) -> dict:
        return {'type': self.__class__.__name__}

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
    def to_dict(self) -> dict:
        d = super(MultiTextProcessor, self).to_dict()
        d['processors'] = [p.to_dict() for p in self.sub_processors]
        return d

    @classmethod
    def from_dict(cls, d: dict):
        d['processors'] = [TextProcessor.from_dict(p) for p in d['processors']]
        return super(MultiTextProcessor, cls).from_dict(d)

    def __init__(self, processors=None):
        super().__init__()
        self.sub_processors = processors if processors else []

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
