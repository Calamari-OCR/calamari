import shutil
import sys

import calamari_ocr.scripts.tensorflow_rename_variables as tensorflow_rename_variables
from calamari_ocr import __version__

import json
import os


class Checkpoint:
    VERSION = 3

    def __init__(self, json_path: str, auto_update=True, dry_run=False):
        self.json_path = json_path if json_path.endswith('.json') else json_path + '.json'
        self.json_path = os.path.abspath(os.path.expanduser(os.path.expandvars(self.json_path)))
        self.ckpt_path = os.path.splitext(self.json_path)[0]
        self.dry_run = dry_run

        # do not parse as proto, since some parameters might have changed
        with open(self.json_path, 'r') as f:
            self.json = json.load(f)

            self.version = self.json['version'] if 'version' in self.json else 0

        if self.version != Checkpoint.VERSION:
            if auto_update:
                self.update_checkpoint()
            else:
                raise Exception("Version of checkpoint is {} but {} is required. Please upgrade the model or "
                                "set the auto update flag.".format(self.version, Checkpoint.VERSION))

        else:
            print("Checkpoint version {} is up-to-date.".format(self.version))

    def update_checkpoint(self):
        while self.version != Checkpoint.VERSION:
            if Checkpoint.VERSION < self.version:
                raise Exception("Downgrading of models is not supported ({} to {}). Please upgrade your Calamari "
                                "instance (currently installed: {})"
                                .format(self.version, Checkpoint.VERSION, __version__))
            self._single_upgrade()

        print("Successfully upgraded checkpoint version to {}".format(Checkpoint.VERSION))

    def _single_upgrade(self):
        print('Upgrading from version {}'.format(self.version))
        if self.version == 0:
            if self.json['model']['network']['backend'].get('type', 'TENSORFLOW') == 'TENSORFLOW':
                tensorflow_rename_variables.rename(self.ckpt_path, '', '', 'cnn_lstm/',
                                                   dry_run=self.dry_run, force_prefix=False)
        elif self.version == 1:
            raise Exception(
                "Due to a update to tensorflow 2.0, the weights can not be converted yet. A retraining is required.")
        elif self.version == 2:
            from calamari_ocr.ocr.migrations.version2to3 import migrate, update_model
            # Calamari 1.3 -> Calamari 2.0
            self.json = migrate(self.json)
            update_model(self.json, self.ckpt_path)

        self.version += 1
        self._update_json_version()

    def _update_json_version(self):
        self.json['version'] = self.version

        if not self.dry_run:
            shutil.copyfile(self.json_path, self.json_path + f'_v{self.version - 1}')
            s = json.dumps(self.json, indent=2)

            with open(self.json_path, 'w') as f:
                f.write(s)


