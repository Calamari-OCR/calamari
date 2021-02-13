import os
from dataclasses import dataclass, field
from typing import List, Generator

import numpy as np
from paiargparse import pai_dataclass, pai_meta
from tfaip.base.data.pipeline.definitions import PipelineMode, TARGETS_PROCESSOR, INPUT_PROCESSOR
from tqdm import tqdm

from calamari_ocr.ocr.dataset.datareader.abbyy.xml import XMLReader, XMLWriter
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGenerator, CalamariDataGeneratorParams, InputSample, \
    SampleMeta
from calamari_ocr.utils import split_all_ext, glob_all, keep_files_with_same_file_name
from calamari_ocr.utils.image import load_image


@pai_dataclass
@dataclass
class Abbyy(CalamariDataGeneratorParams):
    images: List[str] = field(default_factory=list, metadata=pai_meta(required=True))
    xml_files: List[str] = field(default_factory=list)
    gt_extension: str = field(default='.abbyy.xml', metadata=pai_meta(
        help="Default extension of the gt files (expected to exist in same dir)"
    ))
    binary: bool = False

    @staticmethod
    def cls():
        return AbbyyGenerator

    def prepare_for_mode(self, mode: PipelineMode):
        input_image_files = sorted(glob_all(self.images)) if self.images else None

        if not self.xml_files:
            gt_txt_files = [split_all_ext(f)[0] + self.gt_extension for f in
                            input_image_files] if self.gt_extension else None
        else:
            gt_txt_files = sorted(glob_all(self.xml_files))
            if mode in INPUT_PROCESSOR:
                input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
                for img, gt in zip(input_image_files, gt_txt_files):
                    if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                        raise Exception(f"Expected identical basenames of file: {img} and {gt}")
            else:
                input_image_files = None

        self.images = input_image_files
        self.xml_files = gt_txt_files
        if len(self.xml_files) == 0:
            self.xml_files = [split_all_ext(p)[0] + self.gt_extension for p in self.xml_files]

        if len(self.images) == 0:
            self.images = [None] * len(self.xml_files)


class AbbyyGenerator(CalamariDataGenerator[Abbyy]):
    def __init__(self,
                 mode: PipelineMode,
                 params: Abbyy,
                 ):
        super().__init__(mode, params)

        self.book = XMLReader(self.params.images, self.params.xml_files, self.params.skip_invalid).read()

        for p, page in enumerate(self.book.pages):
            for l, line in enumerate(page.getLines()):
                for f, fo in enumerate(line.formats):
                    self.add_sample({
                        "image_path": page.imgFile,
                        "xml_path": page.xmlFile,
                        "id": "{}_{}_{}_{}".format(os.path.splitext(page.xmlFile if page.xmlFile else page.imgFile)[0],
                                                   p, l, f),
                        "line": line,
                        "format": fo,
                    })

    def store_text(self, sentence, sample, output_dir, extension):
        # an Abbyy dataset stores the prediction in one XML file
        sample["format"].text = sentence

    def store(self, extension):
        for page in tqdm(self.book.pages, desc="Writing Abbyy files", total=len(self.book.pages)):
            XMLWriter.write(page, split_all_ext(page.xmlFile)[0] + extension)

    def _sample_iterator(self):
        return zip(self.params.images, self.params.xml_files)

    def _generate_epoch(self, text_only) -> Generator[InputSample, None, None]:
        fold_id = -1
        for p, page in enumerate(self.book.pages):
            if self.mode in INPUT_PROCESSOR:
                img = load_image(page.imgFile)
                if self.params.binary:
                    img = img > 0.9
            else:
                img = None

            for l, line in enumerate(page.getLines()):
                for f, fo in enumerate(line.formats):
                    fold_id += 1
                    sample_id = "{}_{}_{}_{}".format(
                        os.path.splitext(page.xmlFile if page.xmlFile else page.imgFile)[0], p, l, f)
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
                            cut_img = img[line.rect.top: -ly + line.rect.bottom, line.rect.left: -lx + line.rect.right]

                            # add padding as required from normal files
                            cut_img = np.pad(cut_img, ((3, 3), (0, 0)), mode='constant', constant_values=cut_img.max())

                        yield InputSample(cut_img, text, SampleMeta(id=sample_id, fold_id=fold_id))

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError
