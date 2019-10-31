from calamari_ocr.ocr import Checkpoint
from ..util import sparse_to_lists
import numpy as np
from google.protobuf import json_format
import tensorflow as tf
import time
import os
from calamari_ocr.utils.multiprocessing import tqdm_wrapper
keras = tf.keras


class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, training_callback,  codec, data_gen, predict_func, checkpoint_params, steps_per_epoch, vis_cb, progress_bar):
        self.training_callback = training_callback
        self.codec = codec
        self.data_gen = data_gen
        self.predict_func = predict_func
        self.checkpoint_params = checkpoint_params
        self.steps_per_epoch = steps_per_epoch
        self.vis_cb = vis_cb
        self.progress_bar = progress_bar

        checkpoint_frequency = checkpoint_params.checkpoint_frequency
        early_stopping_frequency = checkpoint_params.early_stopping_frequency
        if early_stopping_frequency < 0:
            # set early stopping frequency to half epoch
            # batch size only with square root:
            # esf = 0.5 * epoch_size / sqrt(batch_size) = 0.5 * iters_per_epoch * sqrt(batch_size)
            early_stopping_frequency = int(0.5 * steps_per_epoch * checkpoint_params.batch_size ** 0.5)
        elif 0 < early_stopping_frequency <= 1:
            early_stopping_frequency = int(early_stopping_frequency * steps_per_epoch)  # relative to epochs
        else:
            early_stopping_frequency = int(early_stopping_frequency)
        early_stopping_frequency = max(1, early_stopping_frequency)

        if checkpoint_frequency < 0:
            checkpoint_frequency = early_stopping_frequency
        elif 0 < checkpoint_frequency <= 1:
            checkpoint_frequency = int(checkpoint_frequency * steps_per_epoch)  # relative to epochs
        else:
            checkpoint_frequency = int(checkpoint_frequency)

        self.early_stopping_enabled = self.data_gen is not None \
                                 and checkpoint_params.early_stopping_frequency != 0 \
                                 and checkpoint_params.early_stopping_nbest > 1
        self.early_stopping_best_accuracy = checkpoint_params.early_stopping_best_accuracy
        self.early_stopping_best_cur_nbest = checkpoint_params.early_stopping_best_cur_nbest
        self.early_stopping_best_at_iter = checkpoint_params.early_stopping_best_at_iter
        self.early_stopping_at_accuracy = checkpoint_params.early_stopping_at_acc

        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_frequency = early_stopping_frequency
        self.train_start_time = time.time()
        self.last_checkpoint = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs):
        self.train_start_time = time.time()
        # write initial checkpoint (current best)
        print("Creating initial network configuration as current best.")
        self.last_checkpoint = self.make_checkpoint(
            self.checkpoint_params.early_stopping_best_model_output_dir,
            prefix="",
            version=self.checkpoint_params.early_stopping_best_model_prefix,
        )

    def on_train_end(self, logs):
        # output last model always
        self.last_checkpoint = self.make_checkpoint(self.checkpoint_params.output_dir,
                                                    self.checkpoint_params.output_model_prefix,
                                                    version='last')

    def make_checkpoint(self, base_dir, prefix, version=None):
        base_dir = os.path.abspath(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        checkpoint_params = self.checkpoint_params
        if version:
            checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{}.ckpt".format(prefix, version)))
        else:
            checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{:08d}.ckpt".format(prefix, checkpoint_params.iter + 1)))
        print("Storing checkpoint to '{}'".format(checkpoint_path))
        self.model.save(checkpoint_path + '.h5', overwrite=True)
        checkpoint_params.version = Checkpoint.VERSION
        checkpoint_params.loss_stats[:] = self.vis_cb.loss_stats.values
        checkpoint_params.ler_stats[:] = self.vis_cb.ler_stats.values
        checkpoint_params.dt_stats[:] = self.vis_cb.dt_stats.values
        checkpoint_params.total_time = time.time() - self.train_start_time
        checkpoint_params.early_stopping_best_accuracy = self.early_stopping_best_accuracy
        checkpoint_params.early_stopping_best_cur_nbest = self.early_stopping_best_cur_nbest
        checkpoint_params.early_stopping_best_at_iter = self.early_stopping_best_at_iter

        with open(checkpoint_path + ".json", 'w') as f:
            f.write(json_format.MessageToJson(checkpoint_params))

        return checkpoint_path

    def on_batch_end(self, batch, logs):
        iter = self.checkpoint_params.iter

        if iter >= self.checkpoint_params.max_iters:
            print("Reached maximum numbers of iterations")
            self.model.stop_training = True

        if self.checkpoint_frequency > 0 and (iter + 1) % self.checkpoint_frequency == 0:
            self.last_checkpoint = self.make_checkpoint(self.checkpoint_params.output_dir, self.checkpoint_params.output_model_prefix)

        if self.early_stopping_enabled and (iter + 1) % self.early_stopping_frequency == 0:
            print("Checking early stopping model")
            cer = self._compute_current_cer_on_validation_set(self.steps_per_epoch)
            accuracy = 1 - cer

            if accuracy > self.early_stopping_best_accuracy:
                self.early_stopping_best_accuracy = accuracy
                self.early_stopping_best_cur_nbest = 1
                self.early_stopping_best_at_iter = iter + 1
                # overwrite as best model
                print("Found better model with accuracy of {:%}".format(self.early_stopping_best_accuracy))
                self.last_checkpoint = self.make_checkpoint(
                    self.checkpoint_params.early_stopping_best_model_output_dir,
                    prefix="",
                    version=self.checkpoint_params.early_stopping_best_model_prefix,
                )
            else:
                self.early_stopping_best_cur_nbest += 1
                print("No better model found. Currently accuracy of {:%} at iter {} (remaining nbest = {})".
                      format(self.early_stopping_best_accuracy, self.early_stopping_best_at_iter,
                             self.checkpoint_params.early_stopping_nbest - self.early_stopping_best_cur_nbest))

            self.training_callback.early_stopping(self.early_stopping_best_accuracy,
                                                  self.checkpoint_params.early_stopping_nbest,
                                                  self.early_stopping_best_cur_nbest, iter)

            if accuracy > self.early_stopping_at_accuracy > 0:
                self.model.stop_training = True
                print("Early stopping reached accuracy threshold ({}>{})".format(accuracy, self.early_stopping_at_accuracy))

            if accuracy > 0 and self.early_stopping_best_cur_nbest >= self.checkpoint_params.early_stopping_nbest:
                self.model.stop_training = True
                print("Early stopping now.")

            if accuracy >= 1:
                self.model.stop_training = True
                print("Reached perfect score on validation set. Early stopping now.")

    def on_epoch_end(self, epoch, logs):
        pass

    def _compute_current_cer_on_validation_set(self, count):
        def generate_cer():
            it = iter(self.data_gen)
            for _ in range(count):
                yield np.mean(self.predict_func(next(it))[0])

        return np.mean([cer for cer in tqdm_wrapper(generate_cer(), total=count, progress_bar=self.progress_bar, desc="Early stopping") if np.isfinite(cer)])
