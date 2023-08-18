import logging
import os
from typing import Type

from tfaip.data.pipeline.datapipeline import RawDataPipeline
from tfaip.data.pipeline.definitions import PipelineMode
from tfaip.trainer.callbacks.logger_callback import LoggerCallback
from tfaip.trainer.callbacks.tensor_board_callback import TensorBoardCallback
from tfaip.trainer.callbacks.train_params_logger import TrainerCheckpointsCallback
from tfaip.trainer.trainer import Trainer as AIPTrainer
from tfaip.trainer.warmstart.warmstarter import WarmStarter

from calamari_ocr.ocr import Codec, SavedCalamariModel
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.imageprocessors.augmentation import (
    AugmentationProcessorParams,
)
from calamari_ocr.ocr.model.params import ModelParams
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.training.pipeline_params import CalamariTrainOnlyPipelineParams
from calamari_ocr.ocr.training.warmstart import WarmStarterWithCodecAdaption
from calamari_ocr.utils import checkpoint_path

logger = logging.getLogger(__name__)


class Trainer(AIPTrainer):
    @staticmethod
    def params_cls() -> Type[TrainerParams]:
        return TrainerParams

    @staticmethod
    def warmstarter_cls() -> Type[WarmStarter]:
        return WarmStarterWithCodecAdaption

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
        super(Trainer, self).__init__(params, scenario, restore)
        self._params: TrainerParams = params
        if not isinstance(self._params.checkpoint_save_freq, str) and self._params.checkpoint_save_freq < 0:
            self._params.checkpoint_save_freq = self._params.early_stopping_params.frequency
        self._params.warmstart.model = (
            checkpoint_path(self._params.warmstart.model) if self._params.warmstart.model else None
        )
        self.checkpoint = None
        if self._params.warmstart.model:
            # Manually handle loading
            self.checkpoint = SavedCalamariModel(
                self._params.warmstart.model,
                auto_update=self._params.auto_upgrade_checkpoints,
            )
            self._params.warmstart.model = self.checkpoint.ckpt_path
            self._params.warmstart.trim_graph_name = False
            network = self.checkpoint.trainer_params.network
            if self._params.network != network:
                self._params.network = network
                network_str = "def" if network is None else network
                logger.warning(
                    f"Changing the network structure is not supported. Keeping existing network ({network_str})."
                )

        self._codec_changes = None
        data_aug = self._params.scenario.data.pre_proc.processors_of_type(AugmentationProcessorParams)
        self._retrain_on_original = (
            self._params.data_aug_retrain_on_original
            and len(data_aug) > 0
            and any(p.n_augmentations != 0 for p in data_aug)
        )

    def train(self, callbacks=None, **kwargs):
        callbacks = callbacks if callbacks else []
        self.setup_data()

        # load preloaded dataset
        data: Data = self._data
        model: ModelParams = self.scenario.params.model

        use_training_as_validation = model.ensemble > 0 or self.params.gen.__class__ == CalamariTrainOnlyPipelineParams

        # Setup train pipeline
        train_pipeline = self.params.gen.train_data(data)
        if len(train_pipeline.create_data_generator()) == 0:
            raise ValueError("Training dataset is empty.")

        # Setup validation pipeline
        val_pipeline = None
        if self.params.gen.val_gen():
            if model.ensemble > 0:
                logger.warning(
                    "A validation dataset can not be used when training and ensemble. "
                    "Only a training set is required. Ignoring validation data!"
                )
            else:
                val_pipeline = self.params.gen.val_data(data)
                if len(val_pipeline.create_data_generator()) == 0:
                    raise ValueError(
                        "Validation dataset is empty. Provide valid validation data for early stopping. "
                        "Alternative select train only data generator mode."
                    )

        if self.params.gen.train_data(data).generator_params.preload:
            # preload before codec was created (not all processors can be applied, yet)
            data.preload(progress_bar=self._params.progress_bar)
            train_pipeline = self.params.gen.train_data(data)
            if val_pipeline:
                val_pipeline = self.params.gen.val_data(data)

        # compute the codec
        codec = data.params.codec
        if not codec:
            if self._params.codec.auto_compute or len(self._params.codec.resolved_include_chars) == 0:
                codec = Codec.from_input_dataset(
                    filter(lambda x: x, [train_pipeline, val_pipeline]),
                    codec_construction_params=self._params.codec,
                    progress_bar=self._params.progress_bar,
                )
            else:
                codec = Codec(list(self._params.codec.resolved_include_chars))

        data.params.codec = codec
        model.classes = codec.size()

        if self.checkpoint:
            # if we load the weights, take care of codec changes as-well
            restore_checkpoint_params = self.checkpoint.dict
            restore_data_params = restore_checkpoint_params["scenario"]["data"]

            # checks
            if data.params.line_height != restore_data_params["line_height"]:
                raise ValueError(
                    f"The model to restore has a line height of {restore_data_params.line_height}"
                    f" but a line height of {data.params.line_height} is requested"
                )

            # create codec of the same type
            restore_codec = codec.__class__(restore_data_params["codec"]["charset"])

            # the codec changes as tuple (deletions/insertions), and the new codec is the changed old one
            codec_changes = restore_codec.align(codec, shrink=not self._params.codec.keep_loaded)
            codec = restore_codec
            logger.info(f"Codec changes: {len(codec_changes[0])} deletions, {len(codec_changes[1])} appends")
            # The actual weight/bias matrix will be changed after loading the old weights
            if not any(codec_changes):
                codec_changes = None  # No codec changes

            self._codec_changes = codec_changes

        model.classes = codec.size()
        data.params.codec = codec
        logger.info(f"CODEC: {codec.charset}")

        if self.params.gen.train_data(data).generator_params.preload:
            # preload after codec was created
            data.preload(progress_bar=self._params.progress_bar)
            train_pipeline = self.params.gen.train_data(data)

        if use_training_as_validation:
            logger.info("Using training data for validation.")
            assert val_pipeline is None
            if self._params.gen.train.preload:
                data._pipelines[PipelineMode.EVALUATION] = RawDataPipeline(
                    [s for s in train_pipeline.samples if not s.meta["augmented"]],
                    pipeline_params=self._params.gen.setup.train,
                    data_base=data,
                    generator_params=train_pipeline.generator_params,
                    input_processors=train_pipeline._input_processors,
                    output_processors=train_pipeline._output_processors,
                ).to_mode(PipelineMode.EVALUATION)
            else:
                data._pipelines[PipelineMode.EVALUATION] = train_pipeline.to_mode(PipelineMode.EVALUATION)
        else:
            if val_pipeline is None:
                raise ValueError(
                    "No validation data provided."
                    "Set 'trainer.gen TrainOnly' to pass only training data."
                    "Validation will be performed on the training data in this case."
                    "Alternatively, set 'trainer.gen SplitTrain' and to use by "
                    "default 20% of the training data for validation"
                )

        last_logs = None
        if self._params.current_stage == 0:
            last_logs = super().train(callbacks=callbacks)

        if self._retrain_on_original:
            logger.info("Starting training on original data only")
            if self._params.current_stage == 0:
                self._params.current_epoch = 0
                self._params.current_stage = 1
                self._params.early_stopping.current = 1  # CER = 100% as initial value
                self._params.early_stopping.n = 0

            # Remove data augmenter
            self._data.params.pre_proc.erase_all(AugmentationProcessorParams)
            # Remove augmented samples if 'preloaded"
            if isinstance(train_pipeline, RawDataPipeline):
                train_pipeline.samples = [s for s in train_pipeline.samples if not s.meta.get("augmented", False)]

            logger.info(f"Training on {len(train_pipeline.create_data_generator())} samples.")

            ses_bkp = self.params.scale_epoch_size
            if self.params.scale_epoch_size_no_da_train > 0:
                self.params.scale_epoch_size = self.params.scale_epoch_size_no_da_train
            super().setup_steps_per_epoch()
            self.params.learning_rate.epochs = self.params.epochs
            self.params.learning_rate.steps_per_epoch = self._steps_per_epoch
            self.params.scale_epoch_size = ses_bkp

            # replace callbacks that require steps per epoch as parameter
            first = True
            for i, cb in enumerate(self._callbacks[:]):
                if isinstance(cb, TensorBoardCallback):
                    self._callbacks[i] = TensorBoardCallback(
                        os.path.join(self.params.output_dir, "real_data"), self._steps_per_epoch, cb.extracted_logs_cb
                    )

                if isinstance(cb, TrainerCheckpointsCallback):
                    if first:
                        self._callbacks[i] = self.create_train_params_logger_callback(
                            store_params=False, store_weights=True
                        )
                        first = False
                    else:
                        self._callbacks[i] = self.create_train_params_logger_callback(
                            store_params=True, store_weights=False
                        )
            logger_callback = next(c for c in self._callbacks if isinstance(c, LoggerCallback))
            super().fit()
            last_logs = logger_callback.last_logs

        logger.info("Training finished")
        return last_logs

    def create_warmstarter(self) -> WarmStarter:
        return WarmStarterWithCodecAdaption(self.params.warmstart, codec_changes=self._codec_changes)

    def setup_callbacks(
        self,
        optimizer,
        callbacks=None,
    ):
        cbs = super().setup_callbacks(optimizer, callbacks)
        if self._retrain_on_original:
            for i, cb in enumerate(cbs[:]):
                if isinstance(cb, TensorBoardCallback):
                    cb.log_dir = os.path.join(cb.log_dir, "aug_data")

        return cbs
