import numpy as np
import h5py
from tfaip import PipelineMode

class Hdf5DatasetWriter:
    def __init__(self, output_filename, n_max=10000):
        self.n_max = n_max
        self.data = []
        self.text = []
        self.dims = []
        self.file = None
        self.files = []
        self.current_chunk = 0
        self.output_filename = output_filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_chunk()
    
    def finish_chunk(self):
        if len(self.text) == 0:
            return

        filename = "{}_{:03d}{}".format(self.output_filename, self.current_chunk, ".h5")
        self.files.append(filename)
        with h5py.File(filename, "w") as file:
            dtstr = h5py.string_dtype(encoding="utf-8")
            dtui8 = h5py.vlen_dtype(np.dtype("uint8"))
            file.create_dataset("transcripts", (len(self.text),), dtype=dtstr, compression="gzip", data=self.text)
            file.create_dataset("images_dims", data=self.dims, dtype=int)
            file.create_dataset("images", (len(self.text),), dtype=dtui8, compression="gzip", data=self.data)

        self.current_chunk += 1
        self.data = []
        self.text = []
        self.dims = []

    def write(self, sample):
        if not sample.inputs.dtype == np.uint8:
            raise TypeError("Data for hdf5 must have type np.uint8")

        self.dims.append(sample.inputs.shape)
        self.text.append(sample.targets)
        self.data.append(sample.inputs.reshape(-1))

        if len(self.data) >= self.n_max:
            self.finish_chunk()



if __name__ == "__main__":
    from calamari_ocr.ocr.dataset.datareader.file import (
        FileDataParams,
    )

    dg = FileDataParams(images="calamari_ocr/test/data/uw3_50lines/train/*.png").create(PipelineMode.TRAINING)
    with Hdf5DatasetWriter("calamari_ocr/test/data/uw3_50lines/uw3-50lines.h5", n_max=1000) as writer:
        for sample in dg.generate():
            writer.write(sample.inputs, sample.targets)

    from contextlib import ExitStack

    with Hdf5DatasetWriter("test", n_max=5) as writer:
        writer.write(np.zeros((10, 10), dtype=np.uint8), "test")
        writer.write(np.zeros((10, 15), dtype=np.uint8), "asdfasd")
        writer.write(np.zeros((1, 10), dtype=np.uint8), "te345")

    l = [Hdf5DatasetWriter("test1", n_max=5), Hdf5DatasetWriter("test2", n_max=5)]
    with ExitStack() as stack:
        w = [stack.enter_context(x) for x in l]
