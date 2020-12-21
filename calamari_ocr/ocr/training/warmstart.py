from typing import List
import numpy as np

from tfaip.base.trainer.warmstart.warmstarter import Warmstarter


class WarmstarterWithCodecAdaption(Warmstarter):
    def __init__(self, params, codec_changes):
        super(WarmstarterWithCodecAdaption, self).__init__(params)
        self.codec_changes = codec_changes

    def _trim(self, names: List[str]):
        names = super(WarmstarterWithCodecAdaption, self)._trim(names)

        # Manually trim to support older checkpoints
        to_trim = ['CalamariGraph/', 'keras_debug_model/CalamariGraph/']
        for tt in to_trim:
            names = [name[len(tt):] if name.startswith(tt) else name for name in names]
        return names

    def apply_weights(self, target_model, new_weights):
        if self.codec_changes is None:
            super(WarmstarterWithCodecAdaption, self).apply_weights(target_model, new_weights)
        else:
            self.copy_weights_from_model(target_model, new_weights, *self.codec_changes)

    def copy_weights_from_model(self, target_model, weights, indices_to_delete, indices_to_add):
        for target_weight, source_weight in zip(target_model.weights, weights):
            if 'logits' not in target_weight.name:
                target_weight.assign(source_weight)
                continue

            if 'kernel' in target_weight.name:
                w_val = np.delete(source_weight, [i - 1 for i in indices_to_delete], axis=1)
                # add new indices at the end
                if list(range(w_val.shape[1], w_val.shape[1] + len(indices_to_add))) != list(sorted(indices_to_add)):
                    raise Exception("Additional labels must be added at the end, but got label indices {} != {}".format(
                        range(w_val.shape[1], w_val.shape[1] + len(indices_to_add)), sorted(indices_to_add)))
                w_val = np.concatenate(
                    (w_val[:, :-1], np.random.uniform(-0.1, 0.1, (w_val.shape[0], len(indices_to_add))), w_val[:, -1:]),
                    axis=1)
                target_weight.assign(w_val)
            elif 'bias' in target_weight.name:
                b_val = np.delete(source_weight, [i - 1 for i in indices_to_delete], axis=0)
                b_val = np.concatenate((b_val[:-1], np.zeros((len(indices_to_add),)), b_val[-1:]), axis=0)
                target_weight.assign(b_val)
            else:
                raise NotImplementedError("logits layer is expected to have kernel and bias and nothing else")


