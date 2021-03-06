import shutil
import json
import os
import logging

from calamari_ocr import __version__
from calamari_ocr.ocr.savedmodel.migrations.version0to1 import rename
from calamari_ocr.utils import split_all_ext

logger = logging.getLogger(__name__)


class SavedCalamariModel:
    VERSION = 4

    def __init__(self, json_path: str, auto_update=True, dry_run=False):
        self.json_path = json_path if json_path.endswith('.json') else json_path + '.json'
        self.json_path = os.path.abspath(os.path.expanduser(os.path.expandvars(self.json_path)))
        self.ckpt_path = os.path.splitext(self.json_path)[0]
        self.dry_run = dry_run
        self.dirname = os.path.dirname(self.ckpt_path)
        self.basename = os.path.basename(split_all_ext(self.ckpt_path)[0])

        # do not parse as proto, since some parameters might have changed
        with open(self.json_path, 'r') as f:
            self.dict = json.load(f)

            self.version = self.dict['version'] if 'version' in self.dict else 0

        if self.version != SavedCalamariModel.VERSION:
            if auto_update:
                self.update_checkpoint()
            else:
                raise Exception("Version of checkpoint is {} but {} is required. Please upgrade the model or "
                                "set the auto update flag.".format(self.version, SavedCalamariModel.VERSION))

        else:
            logger.info(f"Checkpoint version {self.version} is up-to-date.")

        from calamari_ocr.ocr.training.params import TrainerParams
        if 'scenario' in self.dict:
            self.trainer_params = TrainerParams.from_dict(self.dict)
        else:
            self.trainer_params = TrainerParams.from_dict({'scenario': self.dict})
        self.scenario_params = self.trainer_params.scenario

    def update_checkpoint(self):
        while self.version != SavedCalamariModel.VERSION:
            if SavedCalamariModel.VERSION < self.version:
                raise Exception("Downgrading of models is not supported ({} to {}). Please upgrade your Calamari "
                                "instance (currently installed: {})"
                                .format(self.version, SavedCalamariModel.VERSION, __version__))
            self._single_upgrade()

        logger.info(f"Successfully upgraded checkpoint version to {SavedCalamariModel.VERSION}")

    def _single_upgrade(self):
        logger.info(f'Upgrading from version {self.version}')
        shutil.copyfile(self.json_path, self.json_path + f'_v{self.version}')
        shutil.copyfile(self.ckpt_path + '.h5', self.ckpt_path + f'.h5_v{self.version}')
        if self.version < 2:
            raise Exception(f"Models of checkpoint-version lower than {self.version} are not supported anymore. "
                            f"Use an older version of calamari to upgrade them to an supported version.")
        elif self.version == 2:
            from calamari_ocr.ocr.savedmodel.migrations.version2to4 import migrate2to4, update_model
            # Calamari 1.3 -> Calamari 2.0
            self.dict = migrate2to4(self.dict)
            update_model(self.dict, self.ckpt_path)
            self.version = 4
        elif self.version == 3:
            from calamari_ocr.ocr.savedmodel.migrations.version3to4 import migrate3to4, update_model
            self.dict = migrate3to4(self.dict)
            update_model(self.dict, self.ckpt_path)
            self.version = 4

        self._update_json_version()

    def _update_json_version(self):
        self.dict['version'] = self.version

        if not self.dry_run:
            s = json.dumps(self.dict, indent=2)

            with open(self.json_path, 'w') as f:
                f.write(s)
