import glob
import os
from abc import ABC, abstractmethod
from typing import Optional, Generator, Iterator
from typeguard import typechecked
import tensorflow as tf
import numpy as np
import logging

from tfaip.base.data.data import DataBase

from calamari_ocr.ocr.backends.dataset.datareader.factory import FileDataReaderFactory, RawDataReaderFactory
from calamari_ocr.utils.multiprocessing import tqdm_wrapper
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.datasets.datasetype import DataSetMode, DataSetType
from calamari_ocr.ocr.backends.dataset.datareader import FileDataReader, DataReader, RawDataReader
from calamari_ocr.ocr.backends.dataset.data_types import InputSample, CalamariDataParams, PreparedSample, SampleMeta
from calamari_ocr.ocr.backends.dataset.preproc_pipeline import PreprocPipeline, PreprocWorkerParams
from calamari_ocr.proto import CheckpointParams
from calamari_ocr.utils import split_all_ext


logger = logging.getLogger(__name__)


def to_tuple(x):
    return tuple(x)


class CalamariData(DataBase):
    @staticmethod
    def get_params_cls():
        return CalamariDataParams

    @typechecked
    def __init__(self, params: CalamariDataParams):
        super(CalamariData, self).__init__(params)
        self._params = params
        params.train_lists = ['DUMMY']
        params.val_list = ['DUMMY']
        self.train_reader = params.train_reader.create() if params.train_reader else None
        self.val_reader = params.val_reader.create() if params.val_reader else None
        self.predict_reader = params.predict_reader.create() if params.predict_reader else None

    def _input_layer_specs(self):
        return {
            'meta': tf.TensorSpec([], dtype=tf.string),
            'img': tf.TensorSpec([None, self._params.line_height_, self._params.input_channels], dtype=tf.float32),
            'img_len': tf.TensorSpec([], dtype=tf.int32),
                }

    def _target_layer_specs(self):
        return {
            'gt': tf.TensorSpec([None], dtype=tf.int32),
            'gt_len': tf.TensorSpec([], dtype=tf.int32),
        }

    def _get_train_data(self):
        assert(self._params.downscale_factor_ > 0)
        return tf.data.Dataset.from_generator(
            lambda: self._prepare_input_sample(self.train_reader.mode, self.get_unprepared_train_data()),
            output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.string),
        ).map(CalamariData.group)

    def _get_val_data(self, val_list: str):
        assert(self._params.downscale_factor_ > 0)
        return tf.data.Dataset.from_generator(
            lambda: self._prepare_input_sample(self.val_reader.mode, self.get_unprepared_val_data()),
            output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.string)
        ).map(CalamariData.group)

    def _get_predict_data(self, predict_list: str):
        assert(self._params.downscale_factor_ > 0)
        return tf.data.Dataset.from_generator(
            lambda: self._prepare_input_sample(self.predict_reader.mode, self._get_unprepared_predict_data()),
            output_shapes=(tf.float32, tf.int32, tf.int32)
        ).map(CalamariData.group_predict)

    def get_unprepared_train_data(self, text_only=False, epochs=1, no_augmentations=False) -> Generator[InputSample, None, None]:
        return self._input_sample_generator(self.train_reader, text_only, self.params().train_num_processes, self.params().train_limit, epochs, no_augmentations)

    def get_unprepared_val_data(self, text_only=False) -> Generator[InputSample, None, None]:
        return self._input_sample_generator(self.val_reader, text_only, self.params().val_num_processes, self.params().val_limit)

    def _get_unprepared_predict_data(self) -> Generator[InputSample, None, None]:
        return self._input_sample_generator(self.predict_reader, False, self.params().val_num_processes, self.params().val_limit)

    def _input_sample_generator(self, reader: DataReader, text_only, processes, limit, epochs=1, no_augmentations=False) -> Generator[InputSample, None, None]:
        if self.params().raw_dataset:
            return reader.generate(text_only, epochs)

        pp_params = PreprocWorkerParams(
            self._params,
            self._params.text_processor,
            self._params.data_processor,
            self._params.data_augmenter if reader.mode == DataSetMode.TRAIN and not no_augmentations else None,
            self._params.data_aug_params,
            reader.mode,
        )
        pipeline = PreprocPipeline(reader.generate(text_only, epochs), pp_params,
                                   data=self, limit=limit, processes=processes)
        return pipeline.output_generator()

    def _prepare_input_sample(self, mode, generator: Generator[Optional[InputSample], None, None]) -> Generator[PreparedSample, None, None]:
        channels = self._params.input_channels
        codec = self._params.codec
        downscale_factor = self.params().downscale_factor_

        def _wrapper(sample: Optional[InputSample]) -> Optional[PreparedSample]:
            if sample is None:
                return None
            # final preparation
            text = np.array(codec.encode(sample.gt) if sample.gt else np.zeros((0,), dtype='int32'))
            line = sample.image

            # gray or binary input, add missing axis
            if len(line.shape) == 2:
                line = np.expand_dims(sample.image, axis=-1)

            if line.shape[-1] != channels:
                raise ValueError(f"Expected {channels} channels but got {line.shape[-1]}. Shape of input {line.shape}")

            if mode == DataSetMode.TRAIN and len(line) // downscale_factor < 2 * len(text) + 1:
                # skip longer outputs than inputs
                logger.warning(f"Skipping line with longer outputs than inputs (id={sample.meta.id})")
                return None

            return PreparedSample(line / 255.0, text, len(line), len(text), sample.meta.to_json())

        for sample in generator:
            sample = _wrapper(sample)
            if not sample:
                continue
            yield sample

    @staticmethod
    def group(*sample: PreparedSample):
        sample = PreparedSample(*sample)
        return {'img': sample.image, 'img_len': sample.image_len, 'meta': sample.serialized_meta}, \
               {'gt': sample.gt, 'gt_len': sample.gt_len}

    @staticmethod
    def group_predict(*sample: PreparedSample):
        sample = PreparedSample(*sample)
        return {'img': sample.image, 'img_len': sample.image_len, 'meta': sample.serialized_meta}

    def to_raw_dataset(self, progress_bar=True) -> 'CalamariDataBase':
        if self._params.raw_dataset:
            return self

        with self:
            def generate_raw(reader: DataReader, dataset: Iterator[InputSample], label: str, processes: int):
                samples = [sample.to_tuple() for sample in
                           tqdm_wrapper(dataset, total=len(reader), desc=f"Preloading {label} data set",
                                        progress_bar=progress_bar) if sample]

                datas, texts, metas = tuple(zip(*samples))
                # TODO: old also PRED_AND_EVAL
                if not self._params.data_aug_params.no_augs() and reader.mode in [DataSetMode.TRAIN]:
                    abs_n_augs = self._params.data_aug_params.to_abs()
                    datas, texts \
                        = self._params.data_augmenter.augment_datas(datas, texts, n_augmentations=abs_n_augs,
                                                                    processes=processes, progress_bar=progress_bar)
                    metas += tuple(SampleMeta(f'aug_{i}', None, True) for i in range(len(datas) - len(metas)))

                assert(len(datas) == len(texts))
                assert(len(texts) == len(metas))

                return RawDataReaderFactory(RawDataReader(reader.mode, datas, texts, metas))

            params: CalamariDataParams = CalamariDataParams.from_dict(self._params.to_dict())
            params.raw_dataset = True
            params.data_augmenter = None
            params.text_processor = None
            params.data_processor = None
            params.train_reader = generate_raw(self.train_reader, self.get_unprepared_train_data(),
                                               "training", self._params.train_num_processes) if self.train_reader else None
            params.val_reader = generate_raw(self.val_reader, self.get_unprepared_val_data(),
                                             "validation", self._params.val_num_processes) if self.val_reader else None
            params.predict_reader = generate_raw(self.predict_reader, self._get_unprepared_predict_data(),
                                                 "prediction", self._params.val_num_processes) if self.predict_reader else None
            return CalamariData(
                params,
            )


if __name__ == '__main__':
    from calamari_ocr.ocr import Codec
    from calamari_ocr.ocr.data_processing import data_processor_from_proto
    from calamari_ocr.ocr.text_processing import text_processor_from_proto

    this_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.abspath(os.path.join(this_dir, '../..', '..', 'test', 'data', 'uw3_50lines', 'train'))
    fdr = FileDataReaderFactory(
        DataSetType.FILE, DataSetMode.TRAIN, [os.path.join(base_path, '*.png')]
    )

    checkpoint_params = CheckpointParams()
    checkpoint_params.model.data_preprocessor.line_height = 64
    txt_preproc = text_processor_from_proto(
        checkpoint_params.model.text_preprocessor, "pre")
    data_preproc = data_processor_from_proto(
        checkpoint_params.model.data_preprocessor)
    params = CalamariDataParams(
        codec=Codec('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;-?+=_()*{}[]`@#$%^&\'"'),
        downscale_factor_=8,
        line_height_=checkpoint_params.model.data_preprocessor.line_height,
        text_processor=txt_preproc,
        data_processor=data_preproc,
        data_aug_params=DataAugmentationAmount(),
        train_reader=fdr,
        val_reader=fdr,
        predict_reader=fdr,
        input_channels_=1,
    )
    params = CalamariDataParams.from_json(params.to_json())
    print(params.to_json(indent=2))

    data = CalamariData(params,
                        )
    with data:
        for d in data.get_train_data():
            pass

    raw_data = data.to_raw_dataset()
    with raw_data:
        for d in raw_data.get_train_data().as_numpy_iterator():
            pass
