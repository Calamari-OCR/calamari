from dataclasses import dataclass

import tensorflow as tf
from typing import Dict, List, Tuple
import bidi.algorithm as bidi
import Levenshtein
from paiargparse import pai_dataclass
from tfaip import Sample

from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor

from calamari_ocr.ocr.model.params import ModelParams
from tensorflow.python.ops import math_ops

from calamari_ocr.ocr.model.ensemblegraph import EnsembleGraph

keras = tf.keras
K = keras.backend
KL = keras.layers


@pai_dataclass
@dataclass
class EnsembleModelParams(ModelParams):
    @staticmethod
    def cls():
        return EnsembleModel

    def graph_cls(self):
        return EnsembleGraph


class EnsembleModel(ModelBase[EnsembleModelParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cer_total = keras.metrics.Mean("CER")
        self.sub_cer = [keras.metrics.Mean(f"CER_{i}") for i in range(self.params.ensemble)]

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.cer_total.name

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        losses = {
            "loss": K.ctc_batch_cost(
                targets["gt"] - 1,
                outputs["blank_last_softmax"],
                to_2d_list(outputs["out_len"]),
                to_2d_list(targets["gt_len"]),
            )
        }

        for i in range(self._params.ensemble):
            losses[f"loss_{i}"] = K.ctc_batch_cost(
                targets["gt"] - 1,
                outputs[f"blank_last_softmax_{i}"],
                to_2d_list(outputs[f"out_len_{i}"]),
                to_2d_list(targets["gt_len"]),
            )

        return losses

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        def cer(decoded, targets, targets_length):
            greedy_decoded = tf.sparse.from_dense(decoded)
            sparse_targets = tf.cast(
                K.ctc_label_dense_to_sparse(targets, math_ops.cast(K.flatten(targets_length), dtype="int32")),
                "int32",
            )
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        sw = K.flatten(targets["gt_len"])
        return [
            self.cer_total(
                cer(outputs["decoded"], targets["gt"], targets["gt_len"]),
                sample_weight=sw,
            )
        ] + [
            self.sub_cer[i](
                cer(outputs[f"decoded_{i}"], targets["gt"], targets["gt_len"]),
                sample_weight=sw * tf.cast(tf.equal(K.flatten(targets["fold_id"]), i), tf.int32),
            )
            for i in range(self.params.ensemble)
        ]

    def _print_evaluate(self, sample: Sample, data, print_fn):
        targets, outputs = sample.targets, sample.outputs
        gt_sentence = targets["sentence"]
        lr = "\u202A\u202B"
        s = ""

        pred_sentence = outputs.sentence
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        s += "\n  PRED (CER={:.2f}): '{}{}{}'".format(
            cer, lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C"
        ) + "\n  TRUE:            '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C")

        print_fn(s)
