from typing import Generator

from calamari_ocr.ocr.backends.dataset.data_types import InputSample, SampleMeta
from calamari_ocr.ocr.backends.dataset.datareader import DataReader
from calamari_ocr.ocr.datasets import DataSetMode
import numpy as np
import h5py
from calamari_ocr.utils import split_all_ext


class Hdf5Reader(DataReader):
    def __init__(self, mode: DataSetMode,
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
        if mode == DataSetMode.PREDICT or mode == DataSetMode.PRED_AND_EVAL:
            self.prediction = {}

        for filename in self.filenames:
            f = h5py.File(filename, 'r')
            codec = list(map(chr, f['codec']))
            if mode == DataSetMode.PREDICT or mode == DataSetMode.PRED_AND_EVAL:
                self.prediction[filename] = {'transcripts': [], 'codec': codec}

            # create empty samples for id and correct dataset size
            for i, text in enumerate(f['transcripts']):
                self.add_sample({
                    "image": None,
                    "text": "",
                    "id": str(i),
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

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        filename = sample
        f = h5py.File(filename, 'r')
        codec = list(map(chr, f['codec']))
        if text_only:
            for i, text in enumerate(f['transcripts']):
                text = "".join([codec[c] for c in text])
                yield InputSample(None, text, SampleMeta(id=f"{filename}_{i}"))
        else:
            for i, (image, shape, text) in enumerate(zip(f['images'], f['images_dims'], f['transcripts'])):
                image = np.reshape(image, shape)
                text = "".join([codec[c] for c in text])
                yield InputSample(image, text, SampleMeta(id=f"{filename}_{i}"))
