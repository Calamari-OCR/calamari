from calamari_ocr.ocr.datasets import DataSet, DataSetMode
import numpy as np
import h5py
from calamari_ocr.utils import split_all_ext


class Hdf5DataSet(DataSet):
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

        self.prediction = None
        if mode == DataSetMode.PREDICT:
            self.prediction = {}

        for filename in [i for i in images + texts if i is not None]:
            f = h5py.File(filename, 'r')
            codec = list(map(chr, f['codec']))
            if mode == DataSetMode.PREDICT:
                self.prediction[filename] = {'transcripts': [], 'codec': codec}

            if mode == DataSetMode.TRAIN:
                for i, (image, shape, text) in enumerate(zip(f['images'], f['images_dims'], f['transcripts'])):
                    image = np.reshape(image, shape)
                    text = "".join([codec[c] for c in text])
                    self.add_sample({
                        "image": image,
                        "text": text,
                        "id": str(i),
                        "filename": filename,
                    })
            elif mode == DataSetMode.PREDICT:
                for i, (image, shape) in enumerate(zip(f['images'], f['images_dims'])):
                    self.add_sample({
                        "image": np.reshape(image, shape),
                        "text": None,
                        "id": str(i),
                        "filename": filename,
                    })
            elif mode == DataSetMode.EVAL:
                for i, text in enumerate(f['transcripts']):
                    text = "".join([codec[c] for c in text])
                    self.add_sample({
                        "image": None,
                        "text": text,
                        "id": str(i),
                        "filename": filename,
                    })

        self.loaded = True

    def _load_sample(self, sample, text_only):
        if text_only:
            return None, sample['text']
        else:
            return sample['image'], sample['text']

    def __getstate__(self):
        # pickle only relevant information to load samples, drop all irrelevant
        return self.mode

    def __setstate__(self, state):
        self.mode = state

    def store_text(self, sentence, sample, output_dir, extension):
        codec = self.prediction[sample['filename']]['codec']
        self.prediction[sample['filename']]['transcripts'].append(list(map(codec.index, sentence)))

    def store(self):
        for filename, data in self.prediction.items():
            texts = data['transcripts']
            codec = data['codec']
            basename, ext = split_all_ext(filename)
            with h5py.File(basename + '.pred' + ext, 'w') as file:
                dt = h5py.special_dtype(vlen=np.dtype('int32'))
                file.create_dataset('transcripts', (len(texts),), dtype=dt)
                file['transcripts'][...] = texts
                file.create_dataset('codec', data=list(map(ord, codec)))
