from calamari_ocr.ocr.callbacks import ConsoleTrainingCallback
from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec, Checkpoint
from calamari_ocr.ocr.augmentation import DataAugmenter
from calamari_ocr.ocr.backends import create_backend_from_checkpoint
from calamari_ocr.utils.contextmanager import ExitStackWithPop
import time
import os
import numpy as np
import bidi.algorithm as bidi

from calamari_ocr.utils import RunningStatistics, checkpoint_path

from calamari_ocr.ocr import Predictor, Evaluator

from google.protobuf import json_format

from .datasets import InputDataset, StreamingInputDataset


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
            If provided the Codec will not be computed automatically based on the GT, but instead `codec` will be used
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
        self.data_augmenter = data_augmenter
        self.n_augmentations = n_augmentations
        self.dataset = StreamingInputDataset(dataset, self.data_preproc, self.txt_preproc, data_augmenter, n_augmentations,
                                             processes=self.checkpoint_params.processes)
        self.validation_dataset = StreamingInputDataset(validation_dataset, self.data_preproc, self.txt_preproc,
                                                        processes=self.checkpoint_params.processes
                                                        ) if validation_dataset else None
        self.preload_training = preload_training
        self.preload_validation = preload_validation

        if len(self.dataset) == 0:
            raise Exception("Dataset is empty.")

        if self.validation_dataset and len(self.validation_dataset) == 0:
            raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")

    def train(self, auto_compute_codec=False, progress_bar=False, training_callback=ConsoleTrainingCallback()):
        """ Launch the training

        Parameters
        ----------
        auto_compute_codec : bool
            Compute the codec automatically based on the provided ground truth.
            Else provide a codec using a whitelist (faster).

        progress_bar : bool
            Show or hide any progress bar

        training_callback : TrainingCallback
            Callback for the training process (e.g., for displaying the current cer, loss in the console)

        """
        with ExitStackWithPop() as exit_stack:
            checkpoint_params = self.checkpoint_params

            train_start_time = time.time() + self.checkpoint_params.total_time

            exit_stack.enter_context(self.dataset)
            if self.validation_dataset:
                exit_stack.enter_context(self.validation_dataset)

            # load training dataset
            if self.preload_training:
                new_dataset = self.dataset.to_raw_input_dataset(processes=checkpoint_params.processes, progress_bar=progress_bar)
                exit_stack.pop(self.dataset)
                self.dataset = new_dataset
                exit_stack.enter_context(self.dataset)

            # load validation dataset
            if self.validation_dataset and self.preload_validation:
                new_dataset = self.validation_dataset.to_raw_input_dataset(processes=checkpoint_params.processes, progress_bar=progress_bar)
                exit_stack.pop(self.validation_dataset)
                self.validation_dataset = new_dataset
                exit_stack.enter_context(self.validation_dataset)

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
            network_params.classes = len(codec)
            checkpoint_params.model.codec.charset[:] = codec.charset
            print("CODEC: {}".format(codec.charset))

            backend = create_backend_from_checkpoint(
                checkpoint_params=checkpoint_params,
                processes=checkpoint_params.processes,
            )
            train_net = backend.create_net(codec, graph_type="train",
                                           checkpoint_to_load=Checkpoint(self.weights) if self.weights else None,
                                           batch_size=checkpoint_params.batch_size, codec_changes=codec_changes)

            if checkpoint_params.current_stage == 0:
                self._run_train(train_net, train_start_time, progress_bar, self.dataset, self.validation_dataset, training_callback)

            if checkpoint_params.data_aug_retrain_on_original and self.data_augmenter and self.n_augmentations != 0:
                print("Starting training on original data only")
                if checkpoint_params.current_stage == 0:
                    checkpoint_params.current_stage = 1
                    checkpoint_params.iter = 0
                    checkpoint_params.early_stopping_best_at_iter = 0
                    checkpoint_params.early_stopping_best_cur_nbest = 0
                    checkpoint_params.early_stopping_best_accuracy = 0

                self.dataset.generate_only_non_augmented = True  # this is the important line!
                self._run_train(train_net, train_start_time, progress_bar, self.dataset, self.validation_dataset, training_callback)

    def _run_train(self, train_net, train_start_time, progress_bar, train_dataset, val_dataset, training_callback):
        checkpoint_params = self.checkpoint_params
        train_net.train(train_dataset, val_dataset, checkpoint_params, self.txt_postproc, progress_bar, training_callback)
        print("Total training time {}s for {} iterations.".format(time.time() - train_start_time, self.checkpoint_params.iter))
