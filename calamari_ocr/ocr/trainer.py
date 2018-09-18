from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
import time
import os
import numpy as np
import bidi.algorithm as bidi

from calamari_ocr.utils import RunningStatistics, checkpoint_path

from calamari_ocr.ocr import Predictor, Evaluator
from calamari_ocr.proto import CheckpointParams

from google.protobuf import json_format


class Trainer:
    def __init__(self, checkpoint_params,
                 dataset,
                 validation_dataset=None,
                 txt_preproc=None,
                 txt_postproc=None,
                 data_preproc=None,
                 data_augmenter=None,
                 n_augmentations=0,
                 weights=None,
                 codec=None,
                 codec_whitelist=[]):
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
        """
        self.checkpoint_params = checkpoint_params
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.data_augmenter = data_augmenter
        self.n_augmentations = n_augmentations
        self.txt_preproc = txt_preproc if txt_preproc else text_processor_from_proto(checkpoint_params.model.text_preprocessor, "pre")
        self.txt_postproc = txt_postproc if txt_postproc else text_processor_from_proto(checkpoint_params.model.text_postprocessor, "post")
        self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(checkpoint_params.model.data_preprocessor)
        self.weights = checkpoint_path(weights) if weights else None
        self.codec = codec
        self.codec_whitelist = codec_whitelist

    def train(self, progress_bar=False):
        """ Launch the training

        Parameters
        ----------
        progress_bar : bool
            Show or hide any progress bar

        """
        checkpoint_params = self.checkpoint_params

        train_start_time = time.time() + self.checkpoint_params.total_time

        self.dataset.load_samples(processes=1, progress_bar=progress_bar)
        datas, txts = self.dataset.train_samples(skip_empty=checkpoint_params.skip_invalid_gt)
        if len(datas) == 0:
            raise Exception("Empty dataset is not allowed. Check if the data is at the correct location")

        if self.validation_dataset:
            self.validation_dataset.load_samples(processes=1, progress_bar=progress_bar)
            validation_datas, validation_txts = self.validation_dataset.train_samples(skip_empty=checkpoint_params.skip_invalid_gt)
            if len(validation_datas) == 0:
                raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")
        else:
            validation_datas, validation_txts = [], []

        # preprocessing steps
        texts = self.txt_preproc.apply(txts, processes=checkpoint_params.processes, progress_bar=progress_bar)
        datas, params = [list(a) for a in zip(*self.data_preproc.apply(datas, processes=checkpoint_params.processes, progress_bar=progress_bar))]
        validation_txts = self.txt_preproc.apply(validation_txts, processes=checkpoint_params.processes, progress_bar=progress_bar)
        validation_data_params = self.data_preproc.apply(validation_datas, processes=checkpoint_params.processes, progress_bar=progress_bar)

        # compute the codec
        codec = self.codec if self.codec else Codec.from_texts(texts, whitelist=self.codec_whitelist)

        # data augmentation on preprocessed data
        if self.data_augmenter:
            datas, texts = self.data_augmenter.augment_datas(datas, texts, n_augmentations=self.n_augmentations,
                                                             processes=checkpoint_params.processes, progress_bar=progress_bar)

            # TODO: validation data augmentation
            # validation_datas, validation_txts = self.data_augmenter.augment_datas(validation_datas, validation_txts, n_augmentations=0,
            #                                                  processes=checkpoint_params.processes, progress_bar=progress_bar)

        # create backend
        network_params = checkpoint_params.model.network
        network_params.features = checkpoint_params.model.line_height
        network_params.classes = len(codec)
        if self.weights:
            # if we load the weights, take care of codec changes as-well
            with open(self.weights + '.json', 'r') as f:
                restore_checkpoint_params = json_format.Parse(f.read(), CheckpointParams())
                restore_model_params = restore_checkpoint_params.model

            # checks
            if checkpoint_params.model.line_height != network_params.features:
                raise Exception("The model to restore has a line height of {} but a line height of {} is requested".format(
                    network_params.features, checkpoint_params.model.line_height
                ))

            # create codec of the same type
            restore_codec = codec.__class__(restore_model_params.codec.charset)
            # the codec changes as tuple (deletions/insertions), and the new codec is the changed old one
            codec_changes = restore_codec.align(codec)
            codec = restore_codec
            print("Codec changes: {} deletions, {} appends".format(len(codec_changes[0]), len(codec_changes[1])))
            # The actual weight/bias matrix will be changed after loading the old weights
        else:
            codec_changes = None

        # store the new codec
        checkpoint_params.model.codec.charset[:] = codec.charset
        print("CODEC: {}".format(codec.charset))

        # compute the labels with (new/current) codec
        labels = [codec.encode(txt) for txt in texts]

        backend = create_backend_from_proto(network_params,
                                            weights=self.weights,
                                            )
        train_net = backend.create_net(restore=None, weights=self.weights, graph_type="train", batch_size=checkpoint_params.batch_size)
        test_net = backend.create_net(restore=None, weights=self.weights, graph_type="test", batch_size=checkpoint_params.batch_size)
        train_net.set_data(datas, labels)
        test_net.set_data(validation_datas, validation_txts)
        if codec_changes:
            # only required on one net, since the other shares the same variables
            train_net.realign_model_labels(*codec_changes)

        train_net.prepare()
        test_net.prepare()

        loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)

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
                            raise Exception("No checkpoint written yet. Training must be stopped.")
                        else:
                            # reload also non trainable weights, such as solver-specific variables
                            train_net.load_weights(last_checkpoint, restore_only_trainable=False)
                            continue
                    else:
                        continue

                n_infinite_losses = 0

                loss_stats.push(result['loss'])
                ler_stats.push(result['ler'])

                dt_stats.push(time.time() - iter_start_time)

                if iter % checkpoint_params.display == 0:
                    # apply postprocessing to display the true output
                    pred_sentence = self.txt_postproc.apply("".join(codec.decode(result["decoded"][0])))
                    gt_sentence = self.txt_postproc.apply("".join(codec.decode(result["gt"][0])))

                    print("#{:08d}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(iter, loss_stats.mean(), ler_stats.mean(), dt_stats.mean()))
                    # Insert utf-8 ltr/rtl direction marks for bidi support
                    lr = "\u202A\u202B"
                    print(" PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C"))
                    print(" TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))

                if (iter + 1) % checkpoint_params.checkpoint_frequency == 0:
                    last_checkpoint = make_checkpoint(checkpoint_params.output_dir, checkpoint_params.output_model_prefix)

                if early_stopping_enabled and (iter + 1) % checkpoint_params.early_stopping_frequency == 0:
                    print("Checking early stopping model")

                    out = early_stopping_predictor.predict_raw(validation_data_params,
                                                               progress_bar=progress_bar, apply_preproc=False)
                    pred_texts = [d.sentence for d in out]
                    pred_texts = self.txt_preproc.apply(pred_texts, processes=checkpoint_params.processes, progress_bar=progress_bar)
                    result = Evaluator.evaluate(gt_data=validation_txts, pred_data=pred_texts, progress_bar=progress_bar)
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
