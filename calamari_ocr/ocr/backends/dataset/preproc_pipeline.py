from functools import partial
from typing import Callable, NamedTuple, Optional

from calamari_ocr.ocr.augmentation import DataAugmenter

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from tfaip.util.multiprocessing.data.pipeline import DataPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker
import numpy as np

from calamari_ocr.ocr.datasets.datasetype import DataSetMode
from calamari_ocr.ocr.backends.dataset.data_types import SampleMeta, InputSample, CalamariDataParams
from calamari_ocr.ocr.data_processing import DataPreprocessor
from calamari_ocr.ocr.text_processing import TextProcessor


class PreprocWorkerParams(NamedTuple):
    data_params: CalamariDataParams
    text_processor: TextProcessor
    data_processor: DataPreprocessor
    data_augmenter: Optional[DataAugmenter]
    data_aug_params: DataAugmentationAmount
    mode: DataSetMode


class PreprocWorker(DataWorker):
    def __init__(self, params: PreprocWorkerParams):
        self.params = params

    def initialize_thread(self):
        pass

    def process(self, *args, **kwargs):
        return self.apply_single(args[0])

    def apply_single(self, sample: InputSample) -> Optional[InputSample]:
        line = sample.image
        text = sample.gt
        meta = sample.meta

        if self.params.data_processor and line is not None:
            line, pp_params = self.params.data_processor.apply([line], 1, False)[0]
        else:
            pp_params = None

        if self.params.text_processor and text is not None:
            text = self.params.text_processor.apply([text], 1, False)[0]

        # data augmentation
        if not self.params.data_aug_params.no_augs() \
                and line is not None \
                and self.params.data_augmenter \
                and np.random.rand() <= self.params.data_aug_params.to_rel():
            line, text = self.params.data_augmenter.augment_single(line, text)

        if self.params.mode == DataSetMode.TRAIN:
            if not text or len(text) == 0:
                return None

        meta.preproc_info = pp_params
        return InputSample(line, text, meta)


def create_worker(params):
    return PreprocWorker(params)


class PreprocPipeline(DataPipeline):
    def __init__(self, image_target_generator, params: PreprocWorkerParams, **kwargs):
        self.params = params
        super(PreprocPipeline, self).__init__(**kwargs)
        self.image_target_generator = image_target_generator

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(create_worker, self.params)

    def generate_input(self):
        return self.image_target_generator
