import logging
from functools import partial
from typing import Type

from tfaip.base.data.pipeline.datapipeline import RawDataPipeline
from tfaip.base.data.pipeline.definitions import PipelineMode
from tfaip.base.trainer.callbacks.tensor_board_callback import TensorBoardCallback
from tfaip.base.trainer.callbacks.train_params_logger import TrainParamsLoggerCallback
from tfaip.base.trainer.scheduler import Constant
from tfaip.base.trainer.trainer import Trainer as AIPTrainer
from tfaip.base.trainer.warmstart.warmstarter import Warmstarter

from calamari_ocr.ocr import Codec, SavedCalamariModel
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.imageprocessors.augmentation import Augmentation
from calamari_ocr.ocr.model.params import ModelParams
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.training.warmstart import WarmstarterWithCodecAdaption
from calamari_ocr.utils import checkpoint_path

logger = logging.getLogger(__name__)


class Trainer(AIPTrainer):
    @staticmethod
    def params_cls() -> Type[TrainerParams]:
        return TrainerParams

    @staticmethod
    def warmstarter_cls() -> Type[Warmstarter]:
        return WarmstarterWithCodecAdaption

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
        self._params.learning_rate_params = Constant
        if not isinstance(self._params.checkpoint_save_freq, str) and self._params.checkpoint_save_freq < 0:
            self._params.checkpoint_save_freq = self._params.early_stopping_params.frequency
        self._params.warmstart.model = checkpoint_path(
            self._params.warmstart.model) if self._params.warmstart.model else None
        self.checkpoint = None
        if self._params.warmstart.model:
            # Manually handle loading
            self.checkpoint = SavedCalamariModel(self._params.warmstart_params.model,
                                                 auto_update=self._params.auto_upgrade_checkpoints)
            self._params.warmstart.model = self.checkpoint.ckpt_path + '.h5'
            self._params.warmstart.trim_graph_name = False

    def train(self, callbacks=None, **kwargs):
        callbacks = callbacks if callbacks else []

        # load preloaded dataset
        data: Data = self._data
        model: ModelParams = self.scenario.params.model

        if model.ensemble > 0:
            self._params.use_training_as_validation = True

        # Setup train pipeline
        train_pipeline = data.train_data()
        if len(train_pipeline.create_data_generator()) == 0:
            raise ValueError("Training dataset is empty.")

        # Setup validation pipeline
        val_pipeline = None
        if data.params.val:
            if model.ensemble > 0:
                logger.warning("A validation dataset can not be used when training and ensemble. "
                               "Only a training set is required. Ignoring validation data!")
            else:
                val_pipeline = data.val_data()
                if len(val_pipeline.create_data_generator()) == 0:
                    raise ValueError("Validation dataset is empty. Provide valid validation data for early stopping. "
                                     "Alternative select train only data generator mode.")

        if data.params.train.preload:
            # preload before codec was created (not all processors can be applied, yet)
            data.preload(progress_bar=self._params.progress_bar)
            train_pipeline = data.train_data()
            if val_pipeline:
                val_pipeline = data.val_data()

        # compute the codec
        codec = data.params.codec
        if not codec:
            if self._params.codec.auto_compute or len(self._params.codec.resolved_include_chars) == 0:
                codec = Codec.from_input_dataset(filter(lambda x: x, [train_pipeline, val_pipeline]),
                                                 codec_construction_params=self._params.codec,
                                                 progress_bar=self._params.progress_bar)
            else:
                codec = Codec(list(self._params.codec.resolved_include_chars))

        data.params.codec = codec
        data.params.downscale_factor = model.compute_downscale_factor()
        model.classes = codec.size()

        if self.checkpoint:
            # if we load the weights, take care of codec changes as-well
            restore_checkpoint_params = self.checkpoint.dict
            restore_data_params = restore_checkpoint_params['scenario_params']['data_params']

            # checks
            if data.params.line_height_ != restore_data_params['line_height_']:
                raise ValueError(f"The model to restore has a line height of {restore_data_params.line_height_}"
                                 f" but a line height of {data.params.line_height_} is requested")

            # create codec of the same type
            restore_codec = codec.__class__(restore_data_params['codec']['charset'])

            # the codec changes as tuple (deletions/insertions), and the new codec is the changed old one
            codec_changes = restore_codec.align(codec, shrink=not self._params.keep_loaded_codec)
            codec = restore_codec
            logger.info(f"Codec changes: {len(codec_changes[0])} deletions, {len(codec_changes[1])} appends")
            # The actual weight/bias matrix will be changed after loading the old weights
            if not any(codec_changes):
                codec_changes = None  # No codec changes
        else:
            codec_changes = None

        model.classes = codec.size()
        data.params.codec = codec
        logger.info(f"CODEC: {codec.charset}")

        if data.params.train.preload:
            # preload after codec was created
            data.preload(progress_bar=self._params.progress_bar)
            train_pipeline = data.train_data()

        if self._params.use_training_as_validation:
            assert (val_pipeline is None)
            if self._params.preload_training:
                data._pipelines[PipelineMode.Evaluation] = RawDataPipeline(
                    [s for s in train_pipeline.samples if not s.meta['augmented']],
                    mode=PipelineMode.Evaluation,
                    data_base=data,
                    generator_params=train_pipeline.generator_params,
                    input_processors=train_pipeline._input_processors,
                    output_processors=train_pipeline._output_processors,
                )
            else:
                data._pipelines[PipelineMode.Evaluation] = train_pipeline.to_mode(PipelineMode.Evaluation)

        if self._params.current_stage == 0:
            super(Trainer, self).train(
                callbacks=callbacks,
                warmstart_fn=partial(WarmstarterWithCodecAdaption, codec_changes=codec_changes),
            )

        data_aug = self._data.params.pre_proc.processors_of_type(Augmentation)
        if self._params.data_aug_retrain_on_original and len(data_aug) > 0 and any(
                p.data_aug_params.to_abs() > 0 for p in data_aug):
            logger.info("Starting training on original data only")
            if self._params.current_stage == 0:
                self._params.current_epoch = 0
                self._params.current_stage = 1
                self._params.early_stopping.current_ = 1  # CER = 100% as initial value
                self._params.early_stopping.n_ = 0

            # Remove data augmenter
            self._data.params.pre_proc.erase_all(Augmentation)
            # Remove augmented samples if 'preloaded"
            if isinstance(train_pipeline, RawDataPipeline):
                train_pipeline.samples = [s for s in train_pipeline.samples if not s.meta.get('augmented', False)]

            logger.info(f"Training on {len(train_pipeline.create_data_generator())} samples.")

            super(Trainer, self).setup_steps_per_epoch()

            # replace callbacks that require steps per epoch as parameter
            first = True
            for i, cb in enumerate(self._callbacks[:]):
                if isinstance(cb, TensorBoardCallback):
                    cb.steps_per_epoch = self._steps_per_epoch

                if isinstance(cb, TrainParamsLoggerCallback):
                    if first:
                        self._callbacks[i] = self.create_train_params_logger_callback(store_params=False,
                                                                                      store_weights=True)
                        first = False
                    else:
                        self._callbacks[i] = self.create_train_params_logger_callback(store_params=True,
                                                                                      store_weights=False)

            super(Trainer, self).fit()

        logger.info("Training finished")
