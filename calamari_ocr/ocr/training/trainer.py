from functools import partial
from typing import Type, List
import numpy as np
from tensorflow import keras
from tfaip.base.data.pipeline.datapipeline import RawDataPipeline

from tfaip.base.trainer import Trainer
from tfaip.base.trainer.warmstart.warmstarter import Warmstarter

from calamari_ocr.ocr import Codec, SavedModel
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessor
from calamari_ocr.ocr.model.params import ModelParams
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.utils import checkpoint_path
from calamari_ocr.ocr.dataset.data import CalamariData


class NoopWarmstarter(Warmstarter):
    def warmstart(self, *args, **kwargs):
        return None


class WarmstarterWithCodecAdaption(Warmstarter):
    def __init__(self, params, codec_changes):
        super(WarmstarterWithCodecAdaption, self).__init__(params)
        self.codec_changes = codec_changes

    def _trim(self, names: List[str]):
        names = super(WarmstarterWithCodecAdaption, self)._trim(names)

        # Manually trim to support older checkpoints
        names = [name[14:] if name.startswith('CalamariGraph/') else name for name in names]
        return names

    def apply_weights(self, target_model, new_weights):
        if self.codec_changes is None:
            target_model.load_weights(new_weights)
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


class CalamariTrainer(Trainer):
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
        super(CalamariTrainer, self).__init__(params, scenario, restore)
        self._params: TrainerParams = params
        if not isinstance(self._params.checkpoint_save_freq_, str) and self._params.checkpoint_save_freq_ < 0:
            self._params.checkpoint_save_freq_ = self._params.early_stopping_params.frequency
        self._params.warmstart_params.model = checkpoint_path(self._params.warmstart_params.model) if self._params.warmstart_params.model else None
        self.checkpoint = None
        if self._params.warmstart_params.model:
            # Manually handle loading
            self.checkpoint = SavedModel(self._params.warmstart_params.model, auto_update=self._params.auto_upgrade_checkpoints)
            self._params.warmstart_params.model = self.checkpoint.ckpt_path + '.h5'
            self._params.warmstart_params.trim_graph_name = False

    def train(self, callbacks=None):
        callbacks = callbacks if callbacks else []
        # TODO log total training time
        # train_start_time = time.time() + self.checkpoint_params.total_time

        # load preloaded dataset
        self.scenario.data = self.scenario.create_data()
        data: CalamariData = self.scenario.data
        model_params: ModelParams = self.scenario.params.model_params

        train_pipeline = data.get_train_data()
        if len(train_pipeline.create_data_generator()) == 0:
            raise Exception("Training dataset is empty.")

        if data.params().val:
            val_pipeline = data.get_val_data()
            if len(val_pipeline.create_data_generator()) == 0:
                raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")
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
            restore_checkpoint_params = self.checkpoint.json
            restore_data_params = restore_checkpoint_params['scenario_params']['data_params']

            # checks
            if data.params().line_height_ != restore_data_params['line_height_']:
                raise Exception(f"The model to restore has a line height of {restore_data_params.line_height_}"
                                f" but a line height of {data.params().line_height_} is requested")

            # create codec of the same type
            restore_codec = codec.__class__(restore_data_params['codec']['charset'])

            # the codec changes as tuple (deletions/insertions), and the new codec is the changed old one
            codec_changes = restore_codec.align(codec, shrink=not self._params.keep_loaded_codec)
            codec = restore_codec
            print("Codec changes: {} deletions, {} appends".format(len(codec_changes[0]), len(codec_changes[1])))
            # The actual weight/bias matrix will be changed after loading the old weights
            if not any(codec_changes):
                codec_changes = None  # No codec changes
        else:
            codec_changes = None

        model_params.classes = codec.size()
        data.params().codec = codec
        print("CODEC: {}".format(codec.charset))

        if self._params.preload_training:
            # preload after codec was created
            data.preload(progress_bar=self._params.progress_bar)

        if self._params.current_stage == 0:
            super(CalamariTrainer, self).train(
                callbacks=callbacks,
                warmstart_fn=partial(WarmstarterWithCodecAdaption, codec_changes=codec_changes),
            )

        if self._params.data_aug_retrain_on_original and self._params.scenario_params.data_params.data_aug_params.to_abs() > 0:
            print("Starting training on original data only")
            if self._params.current_stage == 0:
                self._params.current_epoch = 0
                self._params.current_stage = 1
                self._params.early_stopping_params.current_ = 0
                self._params.early_stopping_params.n_ = 0

            # Remove data augmenter
            self._data.params().pre_processors_.sample_processors = [p for p in self._data.params().pre_processors_.sample_processors if p.name != AugmentationProcessor.__name__]
            # Remove augmented samples if 'preloaded"
            if isinstance(train_pipeline, RawDataPipeline):
                train_pipeline.samples = [s for s in train_pipeline.samples if not s.meta.get('augmented', False)]

            print(f"Training on {len(train_pipeline.create_data_generator())} samples.")

            super(CalamariTrainer, self).fit()
