import os
from typing import Type

from tfaip.base.data.pipeline.datapipeline import DataPipeline, SamplePipelineParams
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, DataProcessorFactoryParams, INPUT_PROCESSOR
from typeguard import typechecked
import tensorflow as tf
import logging

from tfaip.base.data.data import DataBase

from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSampleProcessor
from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import NoopDataPreprocessor
from calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.final_preparation import FinalPreparation
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeightProcessor
from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline
from calamari_ocr.ocr.dataset.postprocessors.ctcdecoder import CTCDecoderProcessor
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import default_image_processors
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.datasetype import DataSetType
from calamari_ocr.ocr.dataset.params import  DataParams, PipelineParams
from calamari_ocr.ocr.dataset.postprocessors.reshape import ReshapeOutputsProcessor
from calamari_ocr.ocr.dataset.textprocessors import NoopTextProcessor, BidiTextProcessor, StripTextProcessor, \
    StrToCharList, TextNormalizer, TextRegularizer
from calamari_ocr.ocr.dataset.textprocessors.default_text_processor import default_text_pre_processors

logger = logging.getLogger(__name__)


def to_tuple(x):
    return tuple(x)


class Data(DataBase):
    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline
        return CalamariPipeline

    @classmethod
    def get_default_params(cls) -> DataParams:
        params: DataParams = super(Data, cls).get_default_params()
        params.pre_processors_ = SamplePipelineParams(
            run_parallel=True,
            sample_processors=default_image_processors() +
                              default_text_pre_processors() +
                              [
                                  DataProcessorFactoryParams(AugmentationProcessor.__name__, {PipelineMode.Training}),
                                  DataProcessorFactoryParams(PrepareSampleProcessor.__name__, INPUT_PROCESSOR),
                              ],
        )
        params.post_processors_ = SamplePipelineParams(
            run_parallel=True,
            sample_processors=
            [
                DataProcessorFactoryParams(ReshapeOutputsProcessor.__name__),
                DataProcessorFactoryParams(CTCDecoderProcessor.__name__),
            ] +
            default_text_pre_processors()
        )
        return params

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
            StrToCharList,
            TextNormalizer,
            TextRegularizer,
            AugmentationProcessor,
            PrepareSampleProcessor,

            ReshapeOutputsProcessor,
            CTCDecoderProcessor,
        ])

    @staticmethod
    def get_params_cls():
        return DataParams

    @typechecked
    def __init__(self, params: DataParams):
        super(Data, self).__init__(params)

    def _input_layer_specs(self):
        return {
            'img': tf.TensorSpec([None, self._params.line_height_, self._params.input_channels], dtype=tf.uint8),
            'img_len': tf.TensorSpec([1], dtype=tf.int32),
            'meta': tf.TensorSpec([1], dtype=tf.string),
                }

    def _target_layer_specs(self):
        return {
            'gt': tf.TensorSpec([None], dtype=tf.int32),
            'gt_len': tf.TensorSpec([1], dtype=tf.int32),
        }


if __name__ == '__main__':
    from calamari_ocr.ocr import Codec

    this_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.abspath(os.path.join(this_dir, '..', '..', 'test', 'data', 'uw3_50lines', 'train'))
    fdr = PipelineParams(
        num_processes=8,
        type=DataSetType.FILE,
        files=[os.path.join(base_path, '*.png')],
        gt_extension=DataSetType.gt_extension(DataSetType.FILE),
        limit=1000,
    )

    params = DataParams(
        codec=Codec('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;-?+=_()*{}[]`@#$%^&\'"'),
        downscale_factor_=4,
        line_height_=48,
        pre_processors_=SamplePipelineParams(
            run_parallel=True,
            sample_processors=default_image_processors() +
                              default_text_pre_processors() +
                              [
                                  DataProcessorFactoryParams(AugmentationProcessor.__name__, {PipelineMode.Training}),
                                  DataProcessorFactoryParams(PrepareSampleProcessor.__name__),
                              ],
        ),
        post_processors_=SamplePipelineParams(run_parallel=False),
        data_aug_params=DataAugmentationAmount(amount=2),
        train=fdr,
        val=fdr,
        input_channels=1,
    )
    params = DataParams.from_json(params.to_json())
    print(params.to_json(indent=2))

    data = Data(params)
    pipeline: CalamariPipeline = data.get_train_data()
    if False:
        with pipeline as rd:
            for i, d in enumerate(rd.generate_input_samples()):
                print(i)
            for i, d in enumerate(rd.input_dataset()):
                print(i)

    raw_pipeline = pipeline.as_preloaded()
    with raw_pipeline as rd:
        for i, d in enumerate(rd.generate_input_samples(auto_repeat=False)):
            print(i)
