from random import shuffle
from typing import Generator

from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.params import InputSample, SampleMeta
import numpy as np
import h5py

from calamari_ocr.ocr.dataset.datareader.base import DataReader
from calamari_ocr.utils import split_all_ext


class Hdf5Reader(DataReader):
    def __init__(self, mode: PipelineMode,
                 images=None, texts=None,
                 ):
        """ Create a dataset from memory

        Since this dataset already contains all data in the memory, this dataset may not be loaded

        Parameters
        ----------
        images : list of images
            the images of the dataset
        texts : list of str
            the texts of this dataset
        """
        super().__init__(mode)

        images = images if images is not None else []
        texts = texts if texts is not None else []
        self.filenames = [i for i in set(images + texts) if i is not None]

        self.prediction = None
        if mode == PipelineMode.Prediction or mode == PipelineMode.Evaluation:
            self.prediction = {}

        for filename in self.filenames:
            f = h5py.File(filename, 'r')
            codec = list(map(chr, f['codec']))
            if mode == PipelineMode.Prediction or mode == PipelineMode.Evaluation:
                self.prediction[filename] = {'transcripts': [], 'codec': codec}

            # create empty samples for id and correct dataset size
            for i, text in enumerate(f['transcripts']):
                self.add_sample({
                    "image": None,
                    "text": "",
                    "id": f"{filename}/{i}",
                    "filename": filename,
                })

    def store_text(self, sentence, sample, output_dir, extension):
        codec = self.prediction[sample['filename']]['codec']
        self.prediction[sample['filename']]['transcripts'].append(list(map(codec.index, sentence)))

    def store(self, extension):
        for filename, data in self.prediction.items():
            texts = data['transcripts']
            codec = data['codec']
            basename, ext = split_all_ext(filename)
            with h5py.File(basename + extension, 'w') as file:
                dt = h5py.special_dtype(vlen=np.dtype('int32'))
                file.create_dataset('transcripts', (len(texts),), dtype=dt)
                file['transcripts'][...] = texts
                file.create_dataset('codec', data=list(map(ord, codec)))

    def _sample_iterator(self):
        return self.filenames

    def _generate_epoch(self, text_only) -> Generator[InputSample, None, None]:
        filenames = list(self.filenames)
        if self.mode == PipelineMode.Training:
            shuffle(filenames)

        for filename in filenames:
            with h5py.File(filename, 'r') as f:
                codec = list(map(chr, f['codec']))
                if text_only:
                    for i, (text, idx) in enumerate(zip(f['transcripts'], range(len(f['transcripts'])))):
                        text = "".join([codec[c] for c in text])
                        fold_id = idx % self.n_folds if self.n_folds > 0 else -1
                        yield InputSample(None, text, SampleMeta(id=f"{filename}/{i}", fold_id=fold_id))
                else:
                    gen = zip(f['images'], f['images_dims'], f['transcripts'], range(len(f['images'])))
                    if self.mode == PipelineMode.Training:
                        gen = list(gen)
                        shuffle(gen)

                    for i, (image, shape, text, idx) in enumerate(gen):
                        image = np.reshape(image, shape)
                        text = "".join([codec[c] for c in text])
                        fold_id = idx % self.n_folds if self.n_folds > 0 else -1
                        yield InputSample(image, text, SampleMeta(id=f"{filename}/{i}", fold_id=fold_id))

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError
