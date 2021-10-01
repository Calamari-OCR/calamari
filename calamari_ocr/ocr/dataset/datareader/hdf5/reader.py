from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from typing import Generator, List

from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import PipelineMode

import numpy as np
import h5py

from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    InputSample,
    SampleMeta,
    CalamariDataGeneratorParams,
)
from calamari_ocr.utils import split_all_ext, glob_all


@pai_dataclass
@dataclass
class Hdf5(CalamariDataGeneratorParams):
    files: List[str] = field(default_factory=list, metadata=pai_meta(required=True))
    pred_extension: str = field(
        default=".pred.h5",
        metadata=pai_meta(help="Default extension of the prediction files"),
    )

    def __len__(self):
        return len(self.files)

    def to_prediction(self):
        self.files = sorted(glob_all(self.files))
        pred = deepcopy(self)
        pred.files = [split_all_ext(f)[0] + self.pred_extension for f in self.files]
        return pred

    @staticmethod
    def cls():
        return Hdf5Generator

    def prepare_for_mode(self, mode: PipelineMode):
        self.files = sorted(glob_all(self.files))


class Hdf5Generator(CalamariDataGenerator[Hdf5]):
    def __init__(self, mode: PipelineMode, params: Hdf5):
        super(Hdf5Generator, self).__init__(mode, params)
        self.prediction = None
        if mode == PipelineMode.PREDICTION or mode == PipelineMode.EVALUATION:
            self.prediction = {}

        for filename in self.params.files:
            f = h5py.File(filename, "r")
            codec = list(map(chr, f["codec"]))
            if mode == PipelineMode.PREDICTION or mode == PipelineMode.EVALUATION:
                self.prediction[filename] = {"transcripts": [], "codec": codec}

            basename = split_all_ext(filename)[0]

            # create empty samples for id and correct dataset size
            for i, text in enumerate(f["transcripts"]):
                self.add_sample(
                    {
                        "image": None,
                        "text": "",
                        "id": f"{basename}/{i}",
                        "filename": filename,
                    }
                )

    def store_text_prediction(self, prediction, sample_id, output_dir):
        sample = self.sample_by_id(sample_id)
        codec = self.prediction[sample["filename"]]["codec"]
        self.prediction[sample["filename"]]["transcripts"].append(list(map(codec.index, prediction.sentence)))

    def store(self):
        extension = self.params.pred_extension

        for filename, data in self.prediction.items():
            texts = data["transcripts"]
            codec = data["codec"]
            basename, ext = split_all_ext(filename)
            with h5py.File(basename + extension, "w") as file:
                dt = h5py.special_dtype(vlen=np.dtype("int32"))
                file.create_dataset("transcripts", (len(texts),), dtype=dt)
                file["transcripts"][...] = texts
                file.create_dataset("codec", data=list(map(ord, codec)))

    def _sample_iterator(self):
        return self.params.files

    def _generate_epoch(self, text_only) -> Generator[InputSample, None, None]:
        filenames = list(self.params.files)
        if self.mode == PipelineMode.TRAINING:
            shuffle(filenames)

        for filename in filenames:
            basename = split_all_ext(filename)[0]
            with h5py.File(filename, "r") as f:
                codec = list(map(chr, f["codec"]))
                if text_only:
                    for i, (text, idx) in enumerate(zip(f["transcripts"], range(len(f["transcripts"])))):
                        text = "".join([codec[c] for c in text])
                        fold_id = idx % self.params.n_folds if self.params.n_folds > 0 else -1
                        yield InputSample(
                            None,
                            text,
                            SampleMeta(id=f"{basename}/{i}", fold_id=fold_id),
                        )
                else:
                    gen = zip(
                        f["images"],
                        f["images_dims"],
                        f["transcripts"],
                        range(len(f["images"])),
                    )
                    if self.mode == PipelineMode.TRAINING:
                        gen = list(gen)
                        shuffle(gen)

                    for i, (image, shape, text, idx) in enumerate(gen):
                        image = np.reshape(image, shape)
                        text = "".join([codec[c] for c in text])
                        fold_id = idx % self.params.n_folds if self.params.n_folds > 0 else -1
                        yield InputSample(
                            image,
                            text,
                            SampleMeta(id=f"{basename}/{i}", fold_id=fold_id),
                        )

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError
