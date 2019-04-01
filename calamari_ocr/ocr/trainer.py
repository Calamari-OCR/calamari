from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec, Checkpoint
from calamari_ocr.ocr.augmentation import DataAugmenter
from calamari_ocr.ocr.backends import create_backend_from_proto
import time
import os
import numpy as np
import bidi.algorithm as bidi

from calamari_ocr.utils import RunningStatistics, checkpoint_path

from calamari_ocr.ocr import Predictor, Evaluator

from google.protobuf import json_format

from .datasets import InputDataset


class Trainer:
    def __init__(self, checkpoint_params,
                 dataset,
                 validation_dataset=None,
                 txt_preproc=None,
                 txt_postproc=None,
                 data_preproc=None,
                 data_augmenter: DataAugmenter = None,
                 n_augmentations=0,
                 weights=None,
                 codec=None,
                 codec_whitelist=None,
                 keep_loaded_codec=False,
                 auto_update_checkpoints=True,
                 preload_training=False,
                 preload_validation=False,
                 ):
        """Train a DNN using given preprocessing, weights, and data

        The purpose of the Trainer is handle a default training mechanism.
        As required input it expects a `dataset` and hyperparameters (`checkpoint_params`).

        The steps are
            1. Loading and preprocessing of the dataset
            2. Computation of the codec
            3. Construction of the DNN in the desired Deep Learning Framework
            4. Launch of the training

        During the training the Trainer will perform validation checks if a `validation_dataset` is given
        to determine the best model.
        Furthermore, the current status is printet and checkpoints are written.

        Parameters
        ----------
        checkpoint_params : CheckpointParams
            Proto parameter object that defines all hyperparameters of the model
        dataset : Dataset
            The Dataset used for training
        validation_dataset : Dataset, optional
            The Dataset used for validation, i.e. choosing the best model
        txt_preproc : TextProcessor, optional
            Text preprocessor that is applied on loaded text, before the Codec is computed
        txt_postproc : TextProcessor, optional
            Text processor that is applied on the loaded GT text and on the prediction to receive the final result
        data_preproc : DataProcessor, optional
            Preprocessing for the image lines (e. g. padding, inversion, deskewing, ...)
        data_augmenter : DataAugmenter, optional
            A DataAugmenter object to use for data augmentation. Count is set by `n_augmentations`
        n_augmentations : int, optional
            The number of augmentations performend by the `data_augmenter`
        weights : str, optional
            Path to a trained model for loading its weights
        codec : Codec, optional
            If provided the Codec will not be computed automaticall based on the GT, but instead `codec` will be used
        codec_whitelist : obj:`list` of :obj:`str`
            List of characters to be kept when the loaded `weights` have a different codec than the new one.
        keep_loaded_codec : bool
            Include all characters of the codec of the pretrained model in the new codec
        """
        self.checkpoint_params = checkpoint_params
        self.txt_preproc = txt_preproc if txt_preproc else text_processor_from_proto(checkpoint_params.model.text_preprocessor, "pre")
        self.txt_postproc = txt_postproc if txt_postproc else text_processor_from_proto(checkpoint_params.model.text_postprocessor, "post")
        self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(checkpoint_params.model.data_preprocessor)
        self.weights = checkpoint_path(weights) if weights else None
        self.codec = codec
        self.codec_whitelist = [] if codec_whitelist is None else codec_whitelist
        self.keep_loaded_codec = keep_loaded_codec
        self.auto_update_checkpoints = auto_update_checkpoints
        self.dataset = InputDataset(dataset, self.data_preproc, self.txt_preproc, data_augmenter, n_augmentations)
        self.validation_dataset = InputDataset(validation_dataset, self.data_preproc, self.txt_preproc) if validation_dataset else None
        self.preload_training = preload_training
        self.preload_validation = preload_validation

        if len(self.dataset) == 0:
            raise Exception("Dataset is empty.")

        if self.validation_dataset and len(self.validation_dataset) == 0:
            raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")

    def train(self, auto_compute_codec=False, progress_bar=False):
        """ Launch the training

        Parameters
        ----------
        auto_compute_codec : bool
            Compute the codec automatically based on the provided ground truth.
            Else provide a codec using a whitelist (faster).

        progress_bar : bool
            Show or hide any progress bar

        """
        checkpoint_params = self.checkpoint_params

        train_start_time = time.time() + self.checkpoint_params.total_time

        # load training dataset
        if self.preload_training:
            self.dataset.preload(processes=checkpoint_params.processes, progress_bar=progress_bar)

        # load validation dataset
        if self.validation_dataset and self.preload_validation:
            self.validation_dataset.preload(processes=checkpoint_params.processes, progress_bar=progress_bar)

        # compute the codec
        if self.codec:
            codec = self.codec
        else:
            if len(self.codec_whitelist) == 0 or auto_compute_codec:
                codec = Codec.from_input_dataset([self.dataset, self.validation_dataset],
                                                 whitelist=self.codec_whitelist, progress_bar=progress_bar)
            else:
                codec = Codec.from_texts([], whitelist=self.codec_whitelist)

        # create backend
        network_params = checkpoint_params.model.network
        network_params.features = checkpoint_params.model.line_height
        network_params.classes = len(codec)
        if self.weights:
            # if we load the weights, take care of codec changes as-well
            ckpt = Checkpoint(self.weights + '.json', auto_update=self.auto_update_checkpoints)
            restore_checkpoint_params = ckpt.checkpoint
            restore_model_params = restore_checkpoint_params.model

            # checks
            if checkpoint_params.model.line_height != network_params.features:
                raise Exception("The model to restore has a line height of {} but a line height of {} is requested".format(
                    network_params.features, checkpoint_params.model.line_height
                ))

            # create codec of the same type
            restore_codec = codec.__class__(restore_model_params.codec.charset)

            # the codec changes as tuple (deletions/insertions), and the new codec is the changed old one
            codec_changes = restore_codec.align(codec, shrink=not self.keep_loaded_codec)
            codec = restore_codec
            print("Codec changes: {} deletions, {} appends".format(len(codec_changes[0]), len(codec_changes[1])))
            # The actual weight/bias matrix will be changed after loading the old weights
            if all([c == 0 for c in codec_changes]):
                codec_changes = None  # No codec changes
        else:
            codec_changes = None

        # store the new codec
        checkpoint_params.model.codec.charset[:] = codec.charset
        print("CODEC: {}".format(codec.charset))

        backend = create_backend_from_proto(network_params,
                                            weights=self.weights,
                                            processes=checkpoint_params.processes,
                                            )
        train_net = backend.create_net(self.dataset, codec, restore=None, weights=self.weights, graph_type="train", batch_size=checkpoint_params.batch_size)
        test_net = backend.create_net(self.validation_dataset, codec, restore=None, weights=self.weights, graph_type="test", batch_size=checkpoint_params.batch_size)
        if codec_changes:
            # only required on one net, since the other shares the same variables
            train_net.realign_model_labels(*codec_changes)

        train_net.prepare()
        test_net.prepare()

        if checkpoint_params.current_stage == 0:
            self._run_train(train_net, test_net, codec, train_start_time, progress_bar)

        if checkpoint_params.data_aug_retrain_on_original and self.dataset.data_augmenter and self.dataset.data_augmentation_amount > 0:
            print("Starting training on original data only")
            if checkpoint_params.current_stage == 0:
                checkpoint_params.current_stage = 1
                checkpoint_params.iter = 0
                checkpoint_params.early_stopping_best_at_iter = 0
                checkpoint_params.early_stopping_best_cur_nbest = 0
                checkpoint_params.early_stopping_best_accuracy = 0

            self.dataset.generate_only_non_augmented = True  # this is the important line!
            train_net.prepare()
            test_net.prepare()
            self._run_train(train_net, test_net, codec, train_start_time, progress_bar)

        train_net.prepare()  # reset the state
        test_net.prepare()   # to prevent blocking of tensorflow on shutdown

    def _run_train(self, train_net, test_net, codec, train_start_time, progress_bar):
        checkpoint_params = self.checkpoint_params
        validation_dataset = test_net.input_dataset
        iters_per_epoch = max(1, int(train_net.input_dataset.epoch_size() / checkpoint_params.batch_size))

        loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)

        display = checkpoint_params.display
        display_epochs = display <= 1
        if display <= 0:
            display = 0                                       # to not display anything
        elif display_epochs:
            display = max(1, int(display * iters_per_epoch))  # relative to epochs
        else:
            display = max(1, int(display))                    # iterations

        checkpoint_frequency = checkpoint_params.checkpoint_frequency
        early_stopping_frequency = checkpoint_params.early_stopping_frequency
        if early_stopping_frequency < 0:
            # set early stopping frequency to half epoch
            early_stopping_frequency = int(0.5 * iters_per_epoch)
        elif 0 < early_stopping_frequency <= 1:
            early_stopping_frequency = int(early_stopping_frequency * iters_per_epoch)  # relative to epochs
        else:
            early_stopping_frequency = int(early_stopping_frequency)
        early_stopping_frequency = max(1, early_stopping_frequency)

        if checkpoint_frequency < 0:
            checkpoint_frequency = early_stopping_frequency
        elif 0 < checkpoint_frequency <= 1:
            checkpoint_frequency = int(checkpoint_frequency * iters_per_epoch)  # relative to epochs
        else:
            checkpoint_frequency = int(checkpoint_frequency)

        early_stopping_enabled = self.validation_dataset is not None \
                                 and checkpoint_params.early_stopping_frequency > 0 \
                                 and checkpoint_params.early_stopping_nbest > 1
        early_stopping_best_accuracy = checkpoint_params.early_stopping_best_accuracy
        early_stopping_best_cur_nbest = checkpoint_params.early_stopping_best_cur_nbest
        early_stopping_best_at_iter = checkpoint_params.early_stopping_best_at_iter

        early_stopping_predictor = Predictor(codec=codec, text_postproc=self.txt_postproc,
                                             network=test_net)

        # Start the actual training
        # ====================================================================================

        iter = checkpoint_params.iter

        # helper function to write a checkpoint
        def make_checkpoint(base_dir, prefix, version=None):
            if version:
                checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{}.ckpt".format(prefix, version)))
            else:
                checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{:08d}.ckpt".format(prefix, iter + 1)))
            print("Storing checkpoint to '{}'".format(checkpoint_path))
            train_net.save_checkpoint(checkpoint_path)
            checkpoint_params.version = Checkpoint.VERSION
            checkpoint_params.iter = iter
            checkpoint_params.loss_stats[:] = loss_stats.values
            checkpoint_params.ler_stats[:] = ler_stats.values
            checkpoint_params.dt_stats[:] = dt_stats.values
            checkpoint_params.total_time = time.time() - train_start_time
            checkpoint_params.early_stopping_best_accuracy = early_stopping_best_accuracy
            checkpoint_params.early_stopping_best_cur_nbest = early_stopping_best_cur_nbest
            checkpoint_params.early_stopping_best_at_iter = early_stopping_best_at_iter

            with open(checkpoint_path + ".json", 'w') as f:
                f.write(json_format.MessageToJson(checkpoint_params))

            return checkpoint_path

        try:
            last_checkpoint = None
            n_infinite_losses = 0
            n_max_infinite_losses = 5

            # Training loop, can be interrupted by early stopping
            for iter in range(iter, checkpoint_params.max_iters):
                checkpoint_params.iter = iter

                iter_start_time = time.time()
                result = train_net.train_step()

                if not np.isfinite(result['loss']):
                    n_infinite_losses += 1

                    if n_max_infinite_losses == n_infinite_losses:
                        print("Error: Loss is not finite! Trying to restart from last checkpoint.")
                        if not last_checkpoint:
                            print("Warning: No checkpoint written yet. Reinitializing neural net.")
                            train_net.prepare(uninitialized_variables_only=False, reset_queues=False)
                        else:
                            # reload also non trainable weights, such as solver-specific variables
                            train_net.load_weights(last_checkpoint, restore_only_trainable=False)
                    continue

                n_infinite_losses = 0

                loss_stats.push(result['loss'])
                ler_stats.push(result['ler'])

                dt_stats.push(time.time() - iter_start_time)

                if display > 0 and iter % display == 0:
                    # apply postprocessing to display the true output
                    pred_sentence = self.txt_postproc.apply("".join(codec.decode(result["decoded"][0])))
                    gt_sentence = self.txt_postproc.apply("".join(codec.decode(result["gt"][0])))

                    if display_epochs:
                        print("#{:08f}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                            iter / iters_per_epoch, loss_stats.mean(), ler_stats.mean(), dt_stats.mean()))
                    else:
                        print("#{:08d}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(
                            iter, loss_stats.mean(), ler_stats.mean(), dt_stats.mean()))

                    # Insert utf-8 ltr/rtl direction marks for bidi support
                    lr = "\u202A\u202B"
                    print(" PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C"))
                    print(" TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))

                if checkpoint_frequency > 0 and (iter + 1) % checkpoint_frequency == 0:
                    last_checkpoint = make_checkpoint(checkpoint_params.output_dir, checkpoint_params.output_model_prefix)

                if early_stopping_enabled and (iter + 1) % early_stopping_frequency == 0:
                    print("Checking early stopping model")

                    out_gen = early_stopping_predictor.predict_input_dataset(validation_dataset,
                                                                             progress_bar=progress_bar)
                    result = Evaluator.evaluate_single_list(map(
                        Evaluator.evaluate_single_args,
                        map(lambda d: {'gt': ''.join(d.ground_truth), 'pred': ''.join(d.chars)}, out_gen)))
                    accuracy = 1 - result["avg_ler"]

                    if accuracy > early_stopping_best_accuracy:
                        early_stopping_best_accuracy = accuracy
                        early_stopping_best_cur_nbest = 1
                        early_stopping_best_at_iter = iter + 1
                        # overwrite as best model
                        last_checkpoint = make_checkpoint(
                            checkpoint_params.early_stopping_best_model_output_dir,
                            prefix="",
                            version=checkpoint_params.early_stopping_best_model_prefix,
                        )
                        print("Found better model with accuracy of {:%}".format(early_stopping_best_accuracy))
                    else:
                        early_stopping_best_cur_nbest += 1
                        print("No better model found. Currently accuracy of {:%} at iter {} (remaining nbest = {})".
                              format(early_stopping_best_accuracy, early_stopping_best_at_iter,
                                     checkpoint_params.early_stopping_nbest - early_stopping_best_cur_nbest))

                    if accuracy > 0 and early_stopping_best_cur_nbest >= checkpoint_params.early_stopping_nbest:
                        print("Early stopping now.")
                        break

                    if accuracy >= 1:
                        print("Reached perfect score on validation set. Early stopping now.")
                        break

        except KeyboardInterrupt as e:
            print("Storing interrupted checkpoint")
            make_checkpoint(checkpoint_params.output_dir,
                            checkpoint_params.output_model_prefix,
                            "interrupted")
            raise e

        print("Total time {}s for {} iterations.".format(time.time() - train_start_time, iter))
