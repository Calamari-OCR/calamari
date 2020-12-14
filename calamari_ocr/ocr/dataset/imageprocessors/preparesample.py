import json
import logging

import numpy as np

from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import PipelineMode

logger = logging.getLogger(__name__)


class PrepareSampleProcessor(DataProcessor):
    def supports_preload(self):
        return self.params.codec is not None

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

    def apply(self, inputs, targets, meta: dict):
        codec = self.params.codec
        # final preparation
        text = np.array(codec.encode(targets) if targets else np.zeros((0,), dtype='int32'))
        line = inputs

        # gray or binary input, add missing axis
        if len(line.shape) == 2:
            line = np.expand_dims(line, axis=-1)

        if line.shape[-1] != self.params.input_channels:
            raise ValueError(f"Expected {self.params.input_channels} channels but got {line.shape[-1]}. Shape of input {line.shape}")

        if self.mode in {PipelineMode.Training, PipelineMode.Evaluation} and not self.is_valid_line(text, len(line) // self.params.downscale_factor_):
            # skip longer outputs than inputs (also in evaluation due to loss computation)
            logger.warning(f"Skipping line with longer outputs than inputs (id={meta['id']})")
            return None, None

        return {'img': line.astype(np.uint8), 'img_len': [len(line)], 'meta': [json.dumps(meta)]}, {'gt': text, 'gt_len': [len(text)]}
