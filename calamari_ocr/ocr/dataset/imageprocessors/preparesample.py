import json
import logging

import numpy as np

from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import PipelineMode

logger = logging.getLogger(__name__)


class PrepareSampleProcessor(DataProcessor):
    def supports_preload(self):
        return self.params.codec is not None

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

        if self.mode in {PipelineMode.Training, PipelineMode.Evaluation} and len(line) // self.params.downscale_factor_ < 2 * len(text) + 1:
            # skip longer outputs than inputs (also in evaluation due to loss computation)
            logger.warning(f"Skipping line with longer outputs than inputs (id={meta['id']})")
            return None, None

        return {'img': line.astype(np.uint8), 'img_len': [len(line)], 'meta': [json.dumps(meta)]}, {'gt': text, 'gt_len': [len(text)]}
