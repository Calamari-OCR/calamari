from calamari_ocr.ocr.datasets import DataSet, DataSetMode
import numpy as np
import h5py


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

        for filename in images:
            f = h5py.File(filename, 'r')
            codec = list(map(chr, f['codec']))

            if mode == DataSetMode.TRAIN:
                for i, (image, shape, text) in enumerate(zip(f['images'], f['images_dims'], f['transcripts'])):
                    image = np.reshape(image, shape)
                    text = "".join([codec[c] for c in text])
                    self.add_sample({
                        "image": image,
                        "text": text,
                        "id": str(i),
                    })
            elif mode == DataSetMode.PREDICT:
                for i, (image, shape) in enumerate(zip(f['images'], f['images_dims'])):
                    self.add_sample({
                        "image": np.reshape(image, shape),
                        "text": None,
                        "id": str(i),
                    })
            elif mode == DataSetMode.EVAL:
                for i, text in enumerate(f['transcripts']):
                    text = "".join([codec[c] for c in text])
                    self.add_sample({
                        "image": None,
                        "text": text,
                        "id": str(i),
                    })

        self.loaded = True

    def _load_sample(self, sample, text_only):
        if text_only:
            return None, sample['text']
        else:
            return sample['image'], sample['text']
