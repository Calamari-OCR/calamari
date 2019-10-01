from calamari_ocr.utils import RunningStatistics
from ..util import sparse_to_lists
import numpy as np
import bidi.algorithm as bidi
import tensorflow as tf
import time
keras = tf.keras


class VisCallback(keras.callbacks.Callback):
    def __init__(self, codec, data_gen, predict_func, checkpoint_params, steps_per_epoch, text_post_proc):
        self.codec = codec
        self.data_gen = data_gen
        self.predict_func = predict_func
        self.checkpoint_params = checkpoint_params
        self.steps_per_epoch = steps_per_epoch
        self.text_post_proc = text_post_proc

        self.loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        self.ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        self.dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)

        display = checkpoint_params.display
        self.display_epochs = display <= 1
        if display <= 0:
            display = 0                                       # do not display anything
        elif self.display_epochs:
            display = max(1, int(display * steps_per_epoch))  # relative to epochs
        else:
            display = max(1, int(display))                    # iterations

        self.display = display
        self.iter_start_time = time.time()
        self.train_start_time = time.time()

    def on_train_begin(self, logs):
        self.iter_start_time = time.time()
        self.train_start_time = time.time()

    def on_train_end(self, logs):
        print("Total training time {}s for {} iterations.".format(time.time() - self.train_start_time, self.checkpoint_params.iter))

    def on_batch_end(self, batch, logs):
        dt = time.time() - self.iter_start_time
        self.iter_start_time = time.time()
        self.dt_stats.push(dt)
        self.loss_stats.push(logs['loss'])
        self.checkpoint_params.iter += 1

        if self.display > 0 and self.checkpoint_params.iter % self.display == 0:
            # apply postprocessing to display the true output
            cer, target, decoded = self._generate(1)
            self.ler_stats.push(cer)
            pred_sentence = self.text_post_proc.apply("".join(self.codec.decode(decoded[0])))
            gt_sentence = self.text_post_proc.apply("".join(self.codec.decode(target[0])))

            if self.display_epochs:
                print("#{:08f}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                    self.checkpoint_params.iter / self.steps_per_epoch, self.loss_stats.mean(), self.ler_stats.mean(), self.dt_stats.mean()))
            else:
                print("#{:08d}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                    self.checkpoint_params.iter, self.loss_stats.mean(), self.ler_stats.mean(), self.dt_stats.mean()))

            # Insert utf-8 ltr/rtl direction marks for bidi support
            lr = "\u202A\u202B"
            print("  PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C"))
            print("  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))

    def on_epoch_end(self, epoch, logs):
        pass

    def _generate(self, count):
        it = iter(self.data_gen)
        cer, target, decoded = zip(*[self.predict_func(next(it)) for _ in range(count)])
        return np.mean(cer), sum(map(sparse_to_lists, target), []), sum(map(sparse_to_lists, decoded), [])
