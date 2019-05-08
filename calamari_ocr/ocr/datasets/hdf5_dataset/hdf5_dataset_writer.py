import numpy as np
import h5py

from calamari_ocr.ocr.datasets import DataSetType


class Hdf5DatasetWriter:
    def __init__(self, output_filename, n_max=10000):
        self.n_max = n_max
        self.data = []
        self.text = []
        self.files = []
        self.current_chunk = 0
        self.output_filename = output_filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_chunck()

    def compute_codec(self):
        codec = set()
        for text in self.text:
            codec = codec.union(set(text))

        return list(codec)

    def finish_chunck(self):
        if len(self.text) == 0:
            return

        codec = self.compute_codec()

        filename = "{}_{:03d}{}".format(self.output_filename, self.current_chunk, DataSetType.gt_extension(DataSetType.HDF5))
        self.files.append(filename)
        file = h5py.File(filename, 'w')
        dti32 = h5py.special_dtype(vlen=np.dtype('int32'))
        dtui8 = h5py.special_dtype(vlen=np.dtype('uint8'))
        file.create_dataset('transcripts', (len(self.text),), dtype=dti32, compression='gzip')
        file.create_dataset('images_dims', data=[d.shape for d in self.data], dtype=int)
        file.create_dataset('images', (len(self.text),), dtype=dtui8, compression='gzip')
        file.create_dataset('codec', data=list(map(ord, codec)))
        file['transcripts'][...] = [list(map(codec.index, d)) for d in self.text]
        file['images'][...] = [d.reshape(-1) for d in self.data]
        file.close()

        self.current_chunk += 1
        self.data = []
        self.text = []

    def write(self, data, text):
        if not data.dtype == np.uint8:
            raise TypeError("Data for hdf5 must have type np.uint8")

        if len(data.shape) != 2:
            raise TypeError("Only Gray or Binary images are supported")

        self.data.append(data)
        self.text.append(text)

        if len(self.data) >= self.n_max:
            self.finish_chunck()


if __name__ == "__main__":
    from contextlib import ExitStack

    with Hdf5DatasetWriter('test', n_max=5) as writer:
        writer.write(np.zeros((10, 10), dtype=np.uint8), "test")
        writer.write(np.zeros((10, 15), dtype=np.uint8), "asdfasd")
        writer.write(np.zeros((1, 10), dtype=np.uint8), "te345")

    l = [Hdf5DatasetWriter('test1', n_max=5), Hdf5DatasetWriter('test2', n_max=5)]
    with ExitStack() as stack:
        w = [stack.enter_context(x) for x in l]

