import copy
from functools import partial
from typing import Iterable

from tfaip.base.data.pipeline.dataprocessor import SequenceProcessor
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper, parallel_map

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount, DataAugmentationAmountReference
from tfaip.base.data.pipeline.definitions import InputTargetSample, PipelineMode
from tfaip.base.data.pipeline.pypipeline import PythonPipeline, RawPythonPipeline, DirectProcPythonPipeline, \
    MultiProcPythonPipeline

from calamari_ocr.ocr.backends.dataset.data_types import CalamariPipelineParams, CalamariDataParams
from calamari_ocr.ocr.backends.dataset.datareader.factory import data_reader_from_params
from calamari_ocr.ocr.backends.dataset.imageprocessors.augmentation import AugmentationProcessor
from calamari_ocr.ocr.backends.dataset.imageprocessors.preparesample import PrepareSampleProcessor


class PrePipeline(MultiProcPythonPipeline):
    def __init__(self, data: 'DataBase', mode: PipelineMode, params: CalamariPipelineParams, reader=None):
        super(PythonPipeline, self).__init__(data, mode, params)
        self.reader = data_reader_from_params(mode, params) if reader is None else reader
        self._has_meta = False

    def __len__(self):
        return len(self.reader)

    def to_mode(self, mode):
        if mode == self.mode:
            # No need to copy
            return self
        reader = copy.copy(self.reader)
        reader.mode = mode
        return PrePipeline(self._data, mode, self.params, reader)

    def generate_samples(self) -> Iterable[InputTargetSample]:
        return self.reader.generate()

    def to_raw_pipeline(self, progress_bar=False):
        # Custom implementation since we want the raw pipeline to comprise augmented data
        # 1. find the AugmentationPreproc
        # 2. create preprocessed samples until then
        # 3. multiply samples by augmented
        # 4. apply remaining pipeline

        # Find AugmentationProcessor and check if augmentation is enabled
        from calamari_ocr.ocr.backends.dataset import CalamariData
        params: CalamariDataParams = copy.copy(self._data.params())
        i = -1
        for i, p in enumerate(params.pre_processors_):
            if p.name == AugmentationProcessor.__name__:
                break
        if i == -1 or self.mode != PipelineMode.Training or params.data_aug_params.no_augs():
            return super(PrePipeline, self).to_raw_pipeline(progress_bar)

        # STEP 1: split processors in before and after augmentation step
        pre, aug, post = params.pre_processors_[:i], params.pre_processors_[i], params.pre_processors_[i + 1:]

        # STEP 2: create samples until augmentation step
        params.pre_processors_ = pre
        with CalamariData(params) as dummy_data:
            no_aug_pipeline = PrePipeline(dummy_data, self.mode, self.params, reader=self.reader)
            unprocessed_samples = list(tqdm_wrapper(no_aug_pipeline.generate_preprocessed_samples(auto_repeat=False),
                                                    total=len(no_aug_pipeline.reader),
                                                    desc=f"Preloading {self.mode.value} data set",
                                                    progress_bar=progress_bar))

        # STEP 3: now augment the samples (real, aug ... aug, real, aug ... aug)
        params.data_aug_params = DataAugmentationAmount(DataAugmentationAmountReference.PERCENTAGE, percentage=1.0)
        aug_processor: AugmentationProcessor = CalamariData.data_processor_factory().create(aug, params, self.mode)
        n_augmentation = self._data.params().data_aug_params.to_abs()  # real number of augmentations

        apply_fn = partial(aug_processor.multi_augment, n_augmentations=n_augmentation)
        augmented_samples = parallel_map(apply_fn, unprocessed_samples,
                                         desc="Augmenting data", processes=self.params.num_processes, progress_bar=progress_bar)
        augmented_samples = sum(list(augmented_samples), [])  # Flatten

        # STEP 4: Post processing
        processor = CalamariData.data_processor_factory().create_sequence(post, params, self.mode)
        augmented_samples = processor.apply_on_samples(augmented_samples,
                                                       drop_invalid=True, progress_bar=progress_bar,
                                                       num_processes=self.params.num_processes,
                                                       )
        augmented_samples = list(augmented_samples)

        return RawPythonPipeline(augmented_samples, self._data, self.mode, self.params)


class CalamariPipeline(DirectProcPythonPipeline):
    def generate_samples(self) -> Iterable[InputTargetSample]:
        procs = self.prep_processors

        def proc(sample):
            return procs.apply_on_sample(sample)

        return filter(lambda x: x is not None, map(proc, self.pre_pipeline.generate_preprocessed_samples()))

    def __init__(self, data: 'DataBase', mode: PipelineMode, params: CalamariPipelineParams, pre_pipeline: PrePipeline = None):
        super(PythonPipeline, self).__init__(data, mode, params)
        self.pre_pipeline = pre_pipeline if pre_pipeline else PrePipeline(data, mode, params)
        self.prep_processors = SequenceProcessor(
            data.params(), mode,
            [PrepareSampleProcessor(data.params(), mode)]
        )

    def __len__(self):
        return len(self.pre_pipeline)

    def to_mode(self, mode):
        if mode == self.mode:
            # No need to copy
            return self
        return CalamariPipeline(self._data, mode, self.params, self.pre_pipeline.to_mode(mode))

    def to_raw_pipeline(self, progress_bar=False):
        raw_pre = self.pre_pipeline.to_raw_pipeline(progress_bar)
        return CalamariPipeline(self._data, self.mode, self.params, raw_pre)
