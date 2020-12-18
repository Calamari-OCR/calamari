from functools import partial
from typing import Type
from tfaip.base.data.pipeline.datapipeline import RawDataPipeline
import logging

from tfaip.base.trainer import Trainer as AIPTrainer
from tfaip.base.trainer.warmstart.warmstarter import Warmstarter

from calamari_ocr.ocr import Codec, SavedCalamariModel
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor
from calamari_ocr.ocr.model.params import ModelParams
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.training.warmstart import WarmstarterWithCodecAdaption
from calamari_ocr.utils import checkpoint_path
from calamari_ocr.ocr.dataset.data import Data


logger = logging.getLogger(__name__)


class Trainer(AIPTrainer):
    @staticmethod
    def get_params_cls() -> Type[TrainerParams]:
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
        if not isinstance(self._params.checkpoint_save_freq_, str) and self._params.checkpoint_save_freq_ < 0:
            self._params.checkpoint_save_freq_ = self._params.early_stopping_params.frequency
        self._params.warmstart_params.model = checkpoint_path(self._params.warmstart_params.model) if self._params.warmstart_params.model else None
        self.checkpoint = None
        if self._params.warmstart_params.model:
            # Manually handle loading
            self.checkpoint = SavedCalamariModel(self._params.warmstart_params.model, auto_update=self._params.auto_upgrade_checkpoints)
            self._params.warmstart_params.model = self.checkpoint.ckpt_path + '.h5'
            self._params.warmstart_params.trim_graph_name = False

    def train(self, callbacks=None, **kwargs):
        callbacks = callbacks if callbacks else []

        # load preloaded dataset
        self.scenario.data = self.scenario.create_data()
        data: Data = self.scenario.data
        model_params: ModelParams = self.scenario.params.model_params

        train_pipeline = data.get_train_data()
        if len(train_pipeline.create_data_generator()) == 0:
            raise ValueError("Training dataset is empty.")

        if data.params().val:
            val_pipeline = data.get_val_data()
            if len(val_pipeline.create_data_generator()) == 0:
                raise ValueError("Validation dataset is empty. Provide valid validation data for early stopping.")
        else:
            val_pipeline = None

        if self._params.preload_training:
            # preload before codec was created (not all processors can be applied, yet)
            data.preload(progress_bar=self._params.progress_bar)
            train_pipeline = data.get_train_data()
            if val_pipeline:
                val_pipeline = data.get_val_data()

        # compute the codec
        codec = data.params().codec
        if not codec:
            if len(self._params.codec_whitelist) == 0 or self._params.auto_compute_codec:
                codec = Codec.from_input_dataset(filter(lambda x: x, [train_pipeline, val_pipeline]),
                                                 whitelist=self._params.codec_whitelist, progress_bar=self._params.progress_bar)
            else:
                codec = Codec.from_texts([], whitelist=self._params.codec_whitelist)

        data.params().codec = codec
        data.params().downscale_factor_ = model_params.compute_downscale_factor()
        model_params.classes = codec.size()

        if self.checkpoint:
            # if we load the weights, take care of codec changes as-well
            restore_checkpoint_params = self.checkpoint.dict
            restore_data_params = restore_checkpoint_params['scenario_params']['data_params']

            # checks
            if data.params().line_height_ != restore_data_params['line_height_']:
                raise ValueError(f"The model to restore has a line height of {restore_data_params.line_height_}"
                                 f" but a line height of {data.params().line_height_} is requested")

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

        model_params.classes = codec.size()
        data.params().codec = codec
        logger.info(f"CODEC: {codec.charset}")

        if self._params.preload_training:
            # preload after codec was created
            data.preload(progress_bar=self._params.progress_bar)

        if self._params.current_stage == 0:
            super(Trainer, self).train(
                callbacks=callbacks,
                warmstart_fn=partial(WarmstarterWithCodecAdaption, codec_changes=codec_changes),
            )

        if self._params.data_aug_retrain_on_original and self._params.scenario_params.data_params.data_aug_params.to_abs() > 0:
            logger.info("Starting training on original data only")
            if self._params.current_stage == 0:
                self._params.current_epoch = 0
                self._params.current_stage = 1
                self._params.early_stopping_params.current_ = 1  # CER = 100% as initial value
                self._params.early_stopping_params.n_ = 0

            # Remove data augmenter
            self._data.params().pre_processors_.sample_processors = [p for p in self._data.params().pre_processors_.sample_processors if p.name != AugmentationProcessor.__name__]
            # Remove augmented samples if 'preloaded"
            if isinstance(train_pipeline, RawDataPipeline):
                train_pipeline.samples = [s for s in train_pipeline.samples if not s.meta.get('augmented', False)]

            logger.info(f"Training on {len(train_pipeline.create_data_generator())} samples.")

            super(Trainer, self).fit()

        logger.info("Training finished")
