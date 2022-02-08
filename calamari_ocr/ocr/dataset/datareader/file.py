import codecs
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import numpy as np
import logging

from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import PipelineMode, INPUT_PROCESSOR

from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    CalamariDataGeneratorParams,
    InputSample,
    SampleMeta,
)

from calamari_ocr.utils import split_all_ext, glob_all, keep_files_with_same_file_name

logger = logging.getLogger(__name__)


@pai_dataclass(alt="File")
@dataclass
class FileDataParams(CalamariDataGeneratorParams):
    images: List[str] = field(
        default_factory=list,
        metadata=pai_meta(
            help="List all image files that shall be processed. Ground truth files with the same "
            "base name but with '.gt.txt' as extension are required at the same location",
        ),
    )
    texts: List[str] = field(default_factory=list, metadata=pai_meta(help="List the text files"))
    gt_extension: str = field(
        default=".gt.txt",
        metadata=pai_meta(help="Extension of the gt files (expected to exist in same dir)"),
    )
    pred_extension: str = field(
        default=".pred.txt",
        metadata=pai_meta(help="Extension of prediction text files"),
    )

    @staticmethod
    def cls():
        return FileDataGenerator

    def __len__(self):
        return len(self.images) if self.images else len(self.texts)

    def to_prediction(self):
        self.texts = sorted(glob_all(self.texts))
        pred = deepcopy(self)
        pred.texts = [split_all_ext(t)[0] + self.pred_extension for t in self.texts]
        return pred

    def select(self, indices: List[int]):
        if self.images:
            self.images = [self.images[i] for i in indices]
        if self.texts:
            self.texts = [self.texts[i] for i in indices]

    def prepare_for_mode(self, mode: PipelineMode):
        logger.info("Resolving input files")
        input_image_files = sorted(glob_all(self.images))

        if not self.texts:
            gt_txt_files = [split_all_ext(f)[0] + self.gt_extension for f in input_image_files]
        else:
            gt_txt_files = sorted(glob_all(self.texts))
            if mode in INPUT_PROCESSOR:
                input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
                for img, gt in zip(input_image_files, gt_txt_files):
                    if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                        raise Exception(f"Expected identical basenames of file: {img} and {gt}")
            else:
                input_image_files = None

        if mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
            if len(set(gt_txt_files)) != len(gt_txt_files):
                logger.warning(
                    "Some ground truth text files occur more than once in the data set "
                    "(ignore this warning, if this was intended)."
                )
            if len(set(input_image_files)) != len(input_image_files):
                logger.warning(
                    "Some images occur more than once in the data set. " "This warning should usually not be ignored."
                )

        self.images = input_image_files
        self.texts = gt_txt_files


class FileDataGenerator(CalamariDataGenerator[FileDataParams]):
    def __init__(
        self,
        mode: PipelineMode,
        params: FileDataParams,
        skip_invalid=False,
        remove_invalid=True,
        non_existing_as_empty=False,
    ):
        """Create a dataset from a list of files

        Images or texts may be empty to create a dataset for prediction or evaluation only.
        """
        super().__init__(mode, params)

        if mode == PipelineMode.PREDICTION:
            texts = [None] * len(params.images)
        else:
            texts = params.texts

        if mode == PipelineMode.TARGETS:
            images = [None] * len(params.texts)
        else:
            images = params.images

        for image, text in zip(images, texts):
            try:
                if image is None and text is None:
                    raise Exception("An empty data point is not allowed. Both image and text file are None")

                img_bn, text_bn = None, None
                if image:
                    img_path, img_fn = os.path.split(image)
                    img_bn, img_ext = split_all_ext(img_fn)

                    if not self.params.non_existing_as_empty and not os.path.exists(image):
                        raise Exception("Image at '{}' must exist".format(image))

                if text:
                    if not self.params.non_existing_as_empty and not os.path.exists(text):
                        raise Exception("Text file at '{}' must exist".format(text))

                    text_path, text_fn = os.path.split(text)
                    text_bn, text_ext = split_all_ext(text_fn)

                if image and text and img_bn != text_bn:
                    raise Exception(
                        "Expected image base name equals text base name but got '{}' != '{}'".format(img_bn, text_bn)
                    )
            except Exception as e:
                logger.exception(e)
                if self.params.skip_invalid:
                    logger.warning("Exception raised. Invalid data. Skipping invalid example.")
                    continue
                else:
                    raise e

            self.add_sample(
                {
                    "image_path": image,
                    "text_path": text,
                    "id": img_bn or text_bn,
                    "base_name": img_bn or text_bn,
                }
            )

    def _load_sample(self, sample, text_only):
        if text_only:
            yield InputSample(
                None,
                self._load_gt_txt(sample["text_path"]),
                SampleMeta(sample["id"], fold_id=sample["fold_id"]),
            )
        else:
            yield InputSample(
                self._load_line(sample["image_path"]),
                self._load_gt_txt(sample["text_path"]),
                SampleMeta(sample["id"], fold_id=sample["fold_id"]),
            )

    def _load_gt_txt(self, gt_txt_path):
        if gt_txt_path is None:
            return None

        if not os.path.exists(gt_txt_path):
            if self.params.non_existing_as_empty:
                return ""
            else:
                raise Exception("Text file at '{}' does not exist".format(gt_txt_path))

        try:
            with codecs.open(gt_txt_path, "r", "utf-8") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Could not read {gt_txt_path} as text file.") from e

    def _load_line(self, image_path):
        if image_path is None:
            return None

        if not os.path.exists(image_path):
            if self.params.non_existing_as_empty:
                return np.zeros((1, 1), dtype=np.uint8)
            else:
                raise Exception("Image file at '{}' does not exist".format(image_path))

        try:
            img = self._load_image(image_path)
        except:
            return None

        return img

    def store_text_prediction(self, prediction, sample_id, output_dir):
        sample = self.sample_by_id(sample_id)
        output_dir = output_dir if output_dir else os.path.dirname(sample["image_path"])
        bn = sample.get("base_name", sample["id"])
        with codecs.open(os.path.join(output_dir, bn + self.params.pred_extension), "w", "utf-8") as f:
            f.write(prediction.sentence)
