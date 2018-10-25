from calamari_ocr.proto import CheckpointParams
import calamari_ocr.scripts.tensorflow_rename_variables as tensorflow_rename_variables

import json
from google.protobuf import json_format
import os


class Checkpoint:
    VERSION = 1

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

        with open(self.json_path, 'r') as f:
            self.checkpoint = json_format.Parse(f.read(), CheckpointParams())

    def update_checkpoint(self):
        while self.version != Checkpoint.VERSION:
            self._single_upgrade()

        print("Successfully upgraded checkpoint version to {}".format(Checkpoint.VERSION))

    def _single_upgrade(self):
        print('Upgrading from version {}'.format(self.version))
        if self.version == 0:
            if self.json['model']['network']['backend'].get('type', 'TENSORFLOW') == 'TENSORFLOW':
                tensorflow_rename_variables.rename(self.ckpt_path, '', '', 'cnn_lstm/',
                                                   dry_run=self.dry_run, force_prefix=False)

            pass

        self.version += 1
        self._update_json_version()

    def _update_json_version(self):
        self.json['version'] = self.version

        if not self.dry_run:
            s = json.dumps(self.json, indent=2)

            with open(self.json_path, 'w') as f:
                f.write(s)


