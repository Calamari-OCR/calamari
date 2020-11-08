import copy
from typing import Type

from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.base.trainer import Trainer
from calamari_ocr.ocr import Codec, Checkpoint
from calamari_ocr.ocr.backends import create_backend_from_checkpoint
from calamari_ocr.proto.params import TrainerParams, ModelParams
import time

from calamari_ocr.utils import checkpoint_path

from calamari_ocr.ocr.backends.dataset.data import CalamariData


class CalamariTrainer(Trainer):
    @staticmethod
    def get_params_cls() -> Type[TrainerParams]:
        return TrainerParams

    def __init__(self, params: TrainerParams, scenario, restore=False):
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
        """
        super(CalamariTrainer, self).__init__(params, scenario, restore)
        self._params: TrainerParams = params
        if self._params.checkpoint_save_freq_ < 0:
            self._params.checkpoint_save_freq_ = self._params.early_stopping_params.frequency
        self._params.warmstart_params.model = checkpoint_path(self._params.warmstart_params.model) if self._params.warmstart_params.model else None
        self.checkpoint = None
        if self._params.warmstart_params.model:
            self.checkpoint = Checkpoint(self._params.warmstart_params.model)
            self._params.warmstart_params.model = self.checkpoint.ckpt_path + '.h5'

    def train(self, callbacks=None):
        # TODO log total training time
        # train_start_time = time.time() + self.checkpoint_params.total_time

        # load preloaded datasets
        self.scenario.data = self.scenario.create_data()
        data: CalamariData = self.scenario.data
        model: ModelParams = self.scenario.params.model_params

        train_pipeline = data.get_pipeline(PipelineMode.Training, data.params().train)
        if len(train_pipeline) == 0:
            raise Exception("Training dataset is empty.")

        if data.params().val:
            val_pipeline = data.get_pipeline(PipelineMode.Evaluation, data.params().val)
            if len(val_pipeline) == 0:
                raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")
        else:
            val_pipeline = None

        if self._params.preload_training:
            # preload after codec was created
            # TODO: progress bar
            data.preload()

        # compute the codec
        codec = data.params().codec
        if not codec:
            if len(self._params.codec_whitelist) == 0 or self._params.auto_compute_codec:
                with data:
                    codec = Codec.from_input_dataset(filter(lambda x: x, [train_pipeline, val_pipeline]),
                                                     whitelist=self._params.codec_whitelist, progress_bar=self._params.progress_bar)
            else:
                codec = Codec.from_texts([], whitelist=self._params.codec_whitelist)

        data.params().codec = codec
        data.params().downscale_factor_ = model.compute_downscale_factor()
        model.classes = codec.size()

        if not val_pipeline:
            # TODO: Make this optional
            # A val reader is required, copy train dataset but in pred and eval mode
            data._pipelines[PipelineMode.Evaluation] = train_pipeline.to_mode(PipelineMode.Evaluation)
            data.params().val = data.params().train

        super(CalamariTrainer, self).train(callbacks=callbacks)

        if False:
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
                if not any(codec_changes):
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
                self._run_train(train_net, train_start_time, progress_bar, self.dataset, training_callback)

            if checkpoint_params.data_aug_retrain_on_original and self.data_augmenter and self.n_augmentations != 0:
                print("Starting training on original data only")
                # TODO: THIS MUST BE IMPLEMENTED
                if checkpoint_params.current_stage == 0:
                    checkpoint_params.current_stage = 1
                    checkpoint_params.iter = 0
                    checkpoint_params.early_stopping_best_at_iter = 0
                    checkpoint_params.early_stopping_best_cur_nbest = 0
                    checkpoint_params.early_stopping_best_accuracy = 0

                self.dataset.generate_only_non_augmented = True  # this is the important line!
                self._run_train(train_net, train_start_time, progress_bar, self.dataset, training_callback)

    def _run_train(self, train_net, train_start_time, progress_bar, dataset: CalamariData, training_callback):
        checkpoint_params = self.checkpoint_params
        with dataset:
            train_net.train(dataset.get_train_data(), dataset.get_val_data(), checkpoint_params, self.txt_postproc, progress_bar, training_callback)
        print("Total training time {}s for {} iterations.".format(time.time() - train_start_time, self.checkpoint_params.iter))
