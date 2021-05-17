from typing import Dict, List, Tuple

import Levenshtein
import bidi.algorithm as bidi
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tfaip import Sample
from tfaip.model.modelbase import ModelBase

from calamari_ocr.ocr.model.params import ModelParams

keras = tf.keras
K = keras.backend
KL = keras.layers


class Model(ModelBase[ModelParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cer_metric = tf.keras.metrics.Mean(name="CER")

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.cer_metric.name

    def _loss(
        self,
        inputs: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        return {
            "ctc-loss": K.ctc_batch_cost(
                targets["gt"] - 1,
                outputs["blank_last_softmax"],
                to_2d_list(outputs["out_len"]),
                to_2d_list(targets["gt_len"]),
            )
        }

    def _metric(
        self,
        inputs: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> List[tf.Tensor]:
        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        def cer(decoded, targets, targets_length):
            greedy_decoded = tf.sparse.from_dense(decoded)
            sparse_targets = tf.cast(
                K.ctc_label_dense_to_sparse(targets, math_ops.cast(K.flatten(targets_length), dtype="int32")),
                "int32",
            )
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        return [
            self.cer_metric(
                cer(outputs["decoded"], targets["gt"], targets["gt_len"]),
                sample_weight=K.flatten(targets["gt_len"]),
            )
        ]

    def _print_evaluate(self, sample: Sample, data, print_fn):
        targets, outputs = sample.targets, sample.outputs
        pred_sentence = outputs.sentence
        gt_sentence = targets["sentence"]
        lr = "\u202A\u202B"
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn(
            "\n  CER:  {}".format(cer)
            + "\n  PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C")
            + "\n  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C")
        )
