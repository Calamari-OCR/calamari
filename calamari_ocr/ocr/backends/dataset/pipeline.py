import copy
from functools import partial
from typing import Iterable

from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator
from tfaip.base.data.pipeline.dataprocessor import SequenceProcessor
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper, parallel_map

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount, DataAugmentationAmountReference
from tfaip.base.data.pipeline.definitions import InputTargetSample, PipelineMode

from calamari_ocr.ocr.backends.dataset.data_types import CalamariPipelineParams, CalamariDataParams
from calamari_ocr.ocr.backends.dataset.datareader.factory import data_reader_from_params
from calamari_ocr.ocr.backends.dataset.imageprocessors.augmentation import AugmentationProcessor


class PrePipeline:
    def __init__(self, data: 'DataBase', mode: PipelineMode, params: CalamariPipelineParams, reader=None):
        super(PrePipeline, self).__init__(data, mode, params)
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


class CalamariPipeline(DataPipeline):
    def __init__(self,
                 mode: PipelineMode,
                 data_base: 'DataBase',
                 generator_params: 'DataGeneratorParams',
                 input_processors=None,
                 output_processors=None,
                 ):
        super(CalamariPipeline, self).__init__(mode, data_base, generator_params, input_processors, output_processors)
        self._reader = None

    def reader(self):
        if self._reader is None:
            self._reader = data_reader_from_params(self.mode, self.generator_params)

        return self._reader

    def create_data_generator(self) -> DataGenerator:
        reader = self.reader()

        class Gen(DataGenerator):
            def __len__(self):
                return len(reader)

            def generate(self) -> Iterable[InputTargetSample]:
                return reader.generate()

        return Gen(self.mode, self.generator_params)
