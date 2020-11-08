import os
from typing import Callable

from tfaip.base.data.pipeline.base import BasePipeline
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, BasePipelineParams, DataProcessorFactoryParams
from typeguard import typechecked
import tensorflow as tf
import logging

from tfaip.base.data.data import DataBase

from calamari_ocr.ocr.backends.dataset.imageprocessors.augmentation import AugmentationProcessor
from calamari_ocr.ocr.backends.dataset.imageprocessors.preparesample import PrepareSampleProcessor
from calamari_ocr.ocr.backends.dataset.pipeline import CalamariPipeline
from calamari_ocr.ocr.data_processing import ScaleToHeightProcessor, FinalPreparation, DataRangeNormalizer, \
    NoopDataPreprocessor, CenterNormalizer
from calamari_ocr.ocr.data_processing.default_image_processors import default_image_processors
from calamari_ocr.ocr.text_processing import TextRegularizer, TextNormalizer, StripTextProcessor, BidiTextProcessor, \
    NoopTextProcessor
from calamari_ocr.ocr.text_processing.default_text_processor import default_text_pre_processors
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.datasets.datasetype import DataSetType
from calamari_ocr.ocr.backends.dataset.data_types import  CalamariDataParams, CalamariPipelineParams


logger = logging.getLogger(__name__)


def to_tuple(x):
    return tuple(x)


class CalamariData(DataBase):
    @classmethod
    def pipeline_factory(cls) -> Callable[['DataBase', PipelineMode, BasePipelineParams], BasePipeline]:
        from calamari_ocr.ocr.backends.dataset.pipeline import CalamariPipeline
        return CalamariPipeline

    @classmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        return DataProcessorFactory([
            CenterNormalizer,
            NoopDataPreprocessor,
            DataRangeNormalizer,
            FinalPreparation,
            ScaleToHeightProcessor,
            NoopTextProcessor,
            BidiTextProcessor,
            StripTextProcessor,
            TextNormalizer,
            TextRegularizer,
            AugmentationProcessor,
        ])

    @staticmethod
    def get_params_cls():
        return CalamariDataParams

    @typechecked
    def __init__(self, params: CalamariDataParams):
        super(CalamariData, self).__init__(params)

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


if __name__ == '__main__':
    from calamari_ocr.ocr import Codec

    this_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.abspath(os.path.join(this_dir, '../..', '..', 'test', 'data', 'uw3_50lines', 'train'))
    fdr = CalamariPipelineParams(
        num_processes=1,
        type=DataSetType.FILE,
        files=[os.path.join(base_path, '*.png')],
        gt_extension=DataSetType.gt_extension(DataSetType.FILE),
        limit=5,
    )

    params = CalamariDataParams(
        codec=Codec('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;-?+=_()*{}[]`@#$%^&\'"'),
        downscale_factor_=4,
        line_height_=48,
        pre_processors_=
        default_image_processors() +
        default_text_pre_processors() +
        [
            DataProcessorFactoryParams(AugmentationProcessor.__name__, {PipelineMode.Training}),
        ],
        data_aug_params=DataAugmentationAmount(amount=2),
        train=fdr,
        val=fdr,
        input_channels=1,
    )
    params = CalamariDataParams.from_json(params.to_json())
    print(params.to_json(indent=2))

    data = CalamariData(params)
    pipeline: CalamariPipeline = CalamariData.pipeline_factory()(data, PipelineMode.Training, fdr)
    with data:
        for d in pipeline.generate_preprocessed_samples():
            print(d)
        for d in data.get_train_data():
            pass

    with data:
        raw_pipeline = pipeline.to_raw_pipeline()
    for d in raw_pipeline.generate_preprocessed_samples():
        print(d)
