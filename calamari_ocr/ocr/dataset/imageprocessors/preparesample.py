import json
import logging
from dataclasses import dataclass
from typing import Type

import numpy as np
from paiargparse import pai_dataclass
from tfaip.data.pipeline.definitions import PipelineMode, Sample
from tfaip.data.pipeline.processor.dataprocessor import (
    DataProcessorParams,
    MappingDataProcessor,
)

logger = logging.getLogger(__name__)


@pai_dataclass(alt="PrepareSample")
@dataclass
class PrepareSampleProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return PrepareSample


class PrepareSample(MappingDataProcessor[PrepareSampleProcessorParams]):
    def supports_preload(self):
        return self.data_params.codec is not None

    @staticmethod
    def is_valid_line(text, line_len):
        last_char = -1
        required_blanks = 0
        for char in text:
            if last_char == char:
                required_blanks += 1
            last_char = char

        required_len = len(text) + required_blanks
        return required_len <= line_len

    def apply(self, sample: Sample) -> Sample:
        assert self.data_params.downscale_factor > 0  # Not instantiated yet
        codec = self.data_params.codec
        # final preparation
        text = np.array(codec.encode(sample.targets) if sample.targets else np.zeros((0,), dtype="int32"))
        line = sample.inputs

        # gray or binary input, add missing axis
        if len(line.shape) == 2:
            line = np.expand_dims(line, axis=-1)

        if (
            self.mode
            in {
                PipelineMode.TRAINING,
                PipelineMode.EVALUATION,
            }
            and not self.is_valid_line(text, len(line) // self.data_params.downscale_factor)
        ):
            # skip longer outputs than inputs (also in evaluation due to loss computation)
            logger.warning(f"Skipping line with longer outputs than inputs (id={sample.meta['id']})")
            return sample.new_invalid()

        if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION} and len(text) == 0:
            logger.warning(f"Skipping empty line with empty GT (id={sample.meta['id']})")
            return sample.new_invalid()

        return sample.new_inputs({"img": line.astype(np.uint8), "img_len": [len(line)]}).new_targets(
            {
                "gt": text,
                "gt_len": [len(text)],
                "fold_id": [sample.meta.get("fold_id", -1)],
            }
        )
