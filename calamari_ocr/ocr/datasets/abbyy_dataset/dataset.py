import os
import numpy as np
from PIL import Image
from .reader import XMLReader
from .writer import XMLWriter
from tqdm import tqdm

from calamari_ocr.ocr.datasets import DataSet, DataSetMode
from calamari_ocr.utils import split_all_ext


class AbbyyDataSet(DataSet):

    def __init__(self,
                 mode: DataSetMode,
                 files,
                 xmlfiles=list(),
                 skip_invalid=False,
                 remove_invalid=True,
                 binary=False,
                 non_existing_as_empty=False,
                 ):

        """ Create a dataset from a Path as String

        Parameters
         ----------
        files : [], required
            image files
        skip_invalid : bool, optional
            skip invalid files
        remove_invalid : bool, optional
            remove invalid files
        """

        super().__init__(
            mode,
            skip_invalid, remove_invalid)

        self._non_existing_as_empty = non_existing_as_empty
        if not xmlfiles or len(xmlfiles) == 0:
            xmlfiles = [split_all_ext(p)[0] + ".xml" for p in files]

        if not files or len(files) == 0:
            files = [None] * len(xmlfiles)

        self.book = XMLReader(files, xmlfiles, skip_invalid, remove_invalid).read()
        self.binary = binary

        for p, page in enumerate(self.book.pages):
            for l, line in enumerate(page.getLines()):
                for f, fo in enumerate(line.formats):
                    self.add_sample({
                        "image_path": page.imgFile,
                        "xml_path": page.xmlFile,
                        "id": "{}_{}_{}_{}".format(os.path.splitext(page.xmlFile if page.xmlFile else page.imgFile)[0], p, l, f),
                        "line": line,
                        "format": fo,
                    })

    def _load_sample(self, sample):
        image_path = sample["image_path"]
        line = sample["line"]
        text = None
        if self.mode == DataSetMode.EVAL or self.mode == DataSetMode.TRAIN:
            text = sample["format"].text

        img = None
        if self.mode == DataSetMode.TRAIN or self.mode == DataSetMode.PREDICT:
            img = np.array(Image.open(image_path))

            ly, lx = img.shape

            # Cut the Image
            img = img[line.rect.top: -ly + line.rect.bottom, line.rect.left: -lx + line.rect.right]

            # add padding as required from normal files
            img = np.pad(img, ((3, 3), (0, 0)), mode='constant', constant_values=img.max())

            """Binarize Image"""
            if self.binary:
                img = img > 0.9

        return img, text

    def store_text(self, sentence, sample, output_dir, extension):
        # an Abbyy dataset stores the predtion in one XML file
        sample["format"].text = sentence

    def store(self):
        for page in tqdm(self.book.pages, desc="Writing Abbyy files", total=len(self.book.pages)):
            XMLWriter.write(page, split_all_ext(page.xmlFile)[0] + ".pred.abbyy.xml")

