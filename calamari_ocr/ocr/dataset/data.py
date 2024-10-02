import logging
import os
from typing import Callable, Dict, Type, Optional

import tensorflow as tf
from tfaip.util.tftyping import AnyTensor
from tfaip.data.data import DataBase
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip.data.pipeline.definitions import PipelineMode, INPUT_PROCESSOR
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.imageprocessors.augmentation import (
    AugmentationProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.default_image_processors import (
    default_image_processors,
)
from calamari_ocr.ocr.dataset.imageprocessors.preparesample import (
    PrepareSampleProcessorParams,
)
from calamari_ocr.ocr.dataset.params import DataParams
from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline
from calamari_ocr.ocr.dataset.postprocessors.ctcdecoder import CTCDecoderProcessorParams
from calamari_ocr.ocr.dataset.postprocessors.reshape import (
    ReshapeOutputsProcessorParams,
)
from calamari_ocr.ocr.dataset.textprocessors.default_text_processor import (
    default_text_pre_processors,
)
from calamari_ocr.utils.image import ImageLoaderParams

logger = logging.getLogger(__name__)


def to_tuple(x):
    return tuple(x)


class Data(DataBase[DataParams]):
    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline

        return CalamariPipeline

    @classmethod
    def default_params(cls) -> DataParams:
        params: DataParams = super(Data, cls).default_params()
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=True,
            processors=default_image_processors()
            + default_text_pre_processors()
            + [
                AugmentationProcessorParams(modes={PipelineMode.TRAINING}),
                PrepareSampleProcessorParams(modes=INPUT_PROCESSOR),
            ],
        )
        params.post_proc = SequentialProcessorPipelineParams(
            run_parallel=True,
            processors=[
                ReshapeOutputsProcessorParams(),
                CTCDecoderProcessorParams(),
            ]
            + default_text_pre_processors(),
        )
        return params

    def _input_layer_specs(self):
        return {
            "img": tf.TensorSpec(
                [None, self._params.line_height, self._params.input_channels],
                dtype=tf.uint8,
            ),
            "img_len": tf.TensorSpec([1], dtype=tf.int32),
        }

    def _target_layer_specs(self):
        return {
            "fold_id": tf.TensorSpec([1], dtype=tf.int32),
            "gt": tf.TensorSpec([None], dtype=tf.int32),
            "gt_len": tf.TensorSpec([1], dtype=tf.int32),
        }

    def element_length_fn(self) -> Callable[[Dict[str, AnyTensor]], AnyTensor]:
        def img_len(x):
            return x["img_len"]

        return img_len

    def create_pipeline(
        self,
        pipeline_params: DataPipelineParams,
        params: Optional[CalamariDataGeneratorParams],
    ) -> DataPipeline:
        if params is not None and isinstance(params, ImageLoaderParams):
            params.channels = self.params.input_channels  # set channels
        return self.data_pipeline_cls()(
            pipeline_params,
            self,
            generator_params=params,
        )


if __name__ == "__main__":
    from calamari_ocr.ocr import Codec

    this_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.abspath(os.path.join(this_dir, "..", "..", "test", "data", "uw3_50lines", "train"))

    fdr = FileDataParams(
        num_processes=8,
        images=[os.path.join(base_path, "*.png")],
        limit=1000,
    )

    params = DataParams(
        codec=Codec("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;-?+=_()*{}[]`@#$%^&'\""),
        downscale_factor=4,
        line_height=48,
        pre_proc=SequentialProcessorPipelineParams(
            run_parallel=True,
            processors=default_image_processors()
            + default_text_pre_processors()
            + [
                AugmentationProcessorParams(
                    modes={PipelineMode.TRAINING},
                    data_aug_params=DataAugmentationAmount(amount=2),
                ),
                PrepareSampleProcessorParams(modes=INPUT_PROCESSOR),
            ],
        ),
        post_proc=SequentialProcessorPipelineParams(run_parallel=False),
        train=fdr,
        val=fdr,
        input_channels=1,
    )
    params = DataParams.from_json(params.to_json())
    print(params.to_json(indent=2))

    data = Data(params)
    pipeline: CalamariPipeline = data.train_data()
    if True:
        with pipeline as rd:
            for i, d in enumerate(rd.generate_input_samples()):
                print(i)
            for i, d in enumerate(rd.input_dataset()):
                print(i)

    raw_pipeline = pipeline.as_preloaded()
    with raw_pipeline as rd:
        for i, d in enumerate(rd.generate_input_samples(auto_repeat=False)):
            print(i)
