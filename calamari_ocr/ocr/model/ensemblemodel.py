import tensorflow as tf
from typing import Dict, Type, List, Tuple, Any
import bidi.algorithm as bidi
import Levenshtein
from tfaip import Sample

from tfaip.model.modelbase import ModelBase, ModelBaseParams
from tfaip.util.typing import AnyNumpy

from calamari_ocr.ocr.model.params import ModelParams
from tensorflow.python.ops import math_ops

from calamari_ocr.ocr.model.ensemblegraph import EnsembleGraph
from calamari_ocr.ocr.predict.params import Prediction

keras = tf.keras
K = keras.backend
KL = keras.layers


class EnsembleModel(ModelBase[ModelParams]):
    @classmethod
    def _additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        return [EnsembleGraph]

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "CER"

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return EnsembleGraph(params)

    def _extended_loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        losses = {
            'loss': K.ctc_batch_cost(inputs['gt'] - 1, outputs['blank_last_softmax'],
                                     to_2d_list(outputs['out_len']),
                                     to_2d_list(inputs['gt_len']))
        }

        for i in range(self._params.ensemble):
            losses[f'loss_{i}'] = K.ctc_batch_cost(inputs['gt'] - 1, outputs[f'blank_last_softmax_{i}'],
                                                   to_2d_list(outputs[f'out_len_{i}']),
                                                   to_2d_list(inputs['gt_len']))

        return losses

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def cer(decoded, targets, targets_length):
            greedy_decoded = tf.sparse.from_dense(decoded)
            sparse_targets = tf.cast(K.ctc_label_dense_to_sparse(targets, math_ops.cast(
                K.flatten(targets_length), dtype='int32')), 'int32')
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        metrics = {
            'CER': cer(outputs['decoded'], inputs['gt'], inputs['gt_len']),
        }

        for i in range(self._params.ensemble):
            metrics[f"CER_{i}"] = cer(outputs[f'decoded_{i}'], inputs['gt'], inputs['gt_len'])

        return metrics

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        weights = {
            "CER": K.flatten(targets['gt_len']),
        }
        for i in range(self._params.ensemble):
            # Only count CERs of this voter for validation
            weights[f"CER_{i}"] = weights["CER"] * tf.cast(tf.equal(K.flatten(targets['fold_id']), i), tf.int32)

        return weights

    def print_evaluate(self, sample: Sample, data, print_fn=print):
        targets, outputs = sample.targets, sample.outputs
        gt_sentence = targets['sentence']
        lr = "\u202A\u202B"
        s = ""

        pred_sentence = outputs.sentence
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        s += (
                "\n  PRED (CER={:.2f}): '{}{}{}'".format(cer, lr[bidi.get_base_level(pred_sentence)], pred_sentence,
                                                         "\u202C") +
                "\n  TRUE:            '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))

        print_fn(s)
