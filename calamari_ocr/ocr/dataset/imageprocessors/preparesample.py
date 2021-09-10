import logging
from dataclasses import dataclass, field
from typing import Type

import numpy as np
from paiargparse import pai_dataclass, pai_meta
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

    max_line_width: int = field(
        default=4096, metadata=pai_meta(help="Max width of a line. Set to -1 or 0 to skip this check.")
    )


class PrepareSample(MappingDataProcessor[PrepareSampleProcessorParams]):
    def supports_preload(self):
        return self.data_params.codec is not None

    def is_valid_line(self, text, downscaled_line_len, full_line_len, line_id: str, print_warnings=True):
        """Validate if the given line is valid.

        This function should only be used for training.
        """

        if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
            # Check that the line is not too long (if activated)
            if self.params.max_line_width > 0:
                if full_line_len > self.params.max_line_width:
                    if print_warnings:
                        logger.warning(
                            f"Invalid line, line too long: {full_line_len} > {self.params.max_line_width} (id={line_id})"
                        )
                    return False

            # Check for empty lines
            if len(text) == 0:
                logger.warning(f"Skipping empty line with empty GT (id={line_id})")
                return False

            # Check if the line is long enough so that there is a possible CTC-path (after subsampling)
            last_char = -1
            required_blanks = 0
            for char in text:
                if last_char == char:
                    required_blanks += 1
                last_char = char

            required_len = len(text) + required_blanks
            if required_len > downscaled_line_len:
                if print_warnings:
                    logger.warning(f"Invalid line (longer outputs than inputs) (id={line_id})")
                return False

        return True

    def apply(self, sample: Sample) -> Sample:
        assert self.data_params.downscale_factor > 0  # Not instantiated yet
        codec = self.data_params.codec
        # final preparation
        if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
            text = np.array(codec.encode(sample.targets) if sample.targets else np.zeros((0,), dtype="int32"))
        else:
            text = None

        line = sample.inputs

        # gray or binary input, add missing axis
        if len(line.shape) == 2:
            line = np.expand_dims(line, axis=-1)

        # Validate if the line is valid for training
        if not self.is_valid_line(
            text, len(line) // self.data_params.downscale_factor, len(line), sample.meta.get("id", "Unknown Sample ID")
        ):
            return sample.new_invalid()

        if text is not None:
            sample = sample.new_targets(
                {
                    "gt": np.asarray(text),
                    "gt_len": np.asarray([len(text)]),
                    "fold_id": np.asarray([sample.meta.get("fold_id", -1)]),
                }
            )

        return sample.new_inputs({"img": line.astype(np.uint8), "img_len": np.asarray([len(line)])})
