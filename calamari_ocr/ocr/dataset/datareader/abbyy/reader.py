import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Generator

import numpy as np
from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import (
    PipelineMode,
    TARGETS_PROCESSOR,
    INPUT_PROCESSOR,
)
from tqdm import tqdm

from calamari_ocr.ocr.dataset.datareader.abbyy.xml import XMLReader, XMLWriter
from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    CalamariDataGeneratorParams,
    InputSample,
    SampleMeta,
)
from calamari_ocr.utils import split_all_ext, glob_all

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class Abbyy(CalamariDataGeneratorParams):
    images: List[str] = field(default_factory=list, metadata=pai_meta(required=True))
    xml_files: List[str] = field(default_factory=list)
    gt_extension: str = field(
        default=".abbyy.xml",
        metadata=pai_meta(help="Default extension of the gt files (expected to exist in same dir)"),
    )
    binary: bool = False
    pred_extension: str = field(
        default=".abbyy.pred.xml",
        metadata=pai_meta(help="Default extension of the prediction files"),
    )

    def __len__(self):
        return len(self.images)

    def select(self, indices: List[int]):
        if self.images:
            self.images = [self.images[i] for i in indices]
        if self.xml_files:
            self.xml_files = [self.xml_files[i] for i in indices]

    def to_prediction(self):
        self.xml_files = sorted(glob_all(self.xml_files))
        pred = deepcopy(self)
        pred.xml_files = [split_all_ext(f)[0] + self.pred_extension for f in self.xml_files]
        return pred

    @staticmethod
    def cls():
        return AbbyyGenerator

    def prepare_for_mode(self, mode: PipelineMode):
        self.images = sorted(glob_all(self.images))
        self.xml_files = sorted(glob_all(self.xml_files))
        if not self.xml_files:
            self.xml_files = [split_all_ext(f)[0] + self.gt_extension for f in self.images]
        if not self.images:
            self.images = [None] * len(self.xml_files)

        if len(self.images) != len(self.xml_files):
            raise ValueError(f"Different number of image and xml files, {len(self.images)} != {len(self.xml_files)}")
        for img_path, xml_path in zip(self.images, self.xml_files):
            if img_path and xml_path:
                img_bn, xml_bn = split_all_ext(img_path)[0], split_all_ext(xml_path)[0]
                if img_bn != xml_bn:
                    logger.warning(
                        f"Filenames are not matching, got base names \n  image: {img_bn}\n  xml:   {xml_bn}\n."
                    )


class AbbyyGenerator(CalamariDataGenerator[Abbyy]):
    def __init__(
        self,
        mode: PipelineMode,
        params: Abbyy,
    ):
        super().__init__(mode, params)

        self.book = XMLReader(self.params.images, self.params.xml_files, self.params.skip_invalid).read()

        for p, page in enumerate(self.book.pages):
            for l, line in enumerate(page.getLines()):
                for f, fo in enumerate(line.formats):
                    self.add_sample(
                        {
                            "image_path": page.imgFile,
                            "xml_path": page.xmlFile,
                            "id": "{}_{}_{}_{}".format(split_all_ext(page.xmlFile or page.imgFile)[0], p, l, f),
                            "line": line,
                            "format": fo,
                        }
                    )

    def store_text_prediction(self, prediction, sample_id, output_dir):
        # an Abbyy dataset stores the prediction in one XML file
        sample = self.sample_by_id(sample_id)
        sample["format"].text = prediction.sentence

    def store(self):
        for page in tqdm(self.book.pages, desc="Writing Abbyy files", total=len(self.book.pages)):
            XMLWriter.write(page, split_all_ext(page.xmlFile)[0] + self.params.pred_extension)

    def _sample_iterator(self):
        return zip(self.params.images, self.params.xml_files)

    def _generate_epoch(self, text_only) -> Generator[InputSample, None, None]:
        fold_id = -1
        for p, page in enumerate(self.book.pages):
            if self.mode in INPUT_PROCESSOR:
                img = self._load_image(page.imgFile)
                if self.params.binary:
                    img = img > 0.9
            else:
                img = None

            for l, line in enumerate(page.getLines()):
                for f, fo in enumerate(line.formats):
                    fold_id += 1
                    sample_id = "{}_{}_{}_{}".format(split_all_ext(page.xmlFile or page.imgFile)[0], p, l, f)
                    text = None
                    if self.mode in TARGETS_PROCESSOR:
                        text = fo.text

                    if text_only:
                        yield InputSample(None, text, SampleMeta(id=sample_id, fold_id=fold_id))

                    else:
                        cut_img = None
                        if self.mode in INPUT_PROCESSOR:
                            ly, lx = img.shape

                            # Cut the Image
                            cut_img = img[
                                line.rect.top : -ly + line.rect.bottom,
                                line.rect.left : -lx + line.rect.right,
                            ]

                            # add padding as required from normal files
                            cut_img = np.pad(
                                cut_img,
                                ((3, 3), (0, 0)),
                                mode="constant",
                                constant_values=cut_img.max(),
                            )

                        yield InputSample(cut_img, text, SampleMeta(id=sample_id, fold_id=fold_id))

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError
