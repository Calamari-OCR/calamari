import codecs
import os
import numpy as np
from PIL import Image
import logging
from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.params import InputSample, SampleMeta
from calamari_ocr.ocr.dataset.datareader.base import DataReader

from calamari_ocr.utils import split_all_ext
from calamari_ocr.utils.image import load_image

logger = logging.getLogger(__name__)

class FileDataReader(DataReader):
    def __init__(self,
                 mode: PipelineMode,
                 images=None, texts=None,
                 skip_invalid=False, remove_invalid=True,
                 non_existing_as_empty=False):
        """ Create a dataset from a list of files

        Images or texts may be empty to create a dataset for prediction or evaluation only.
        """
        super().__init__(mode,
                         skip_invalid=skip_invalid,
                         remove_invalid=remove_invalid)
        self._non_existing_as_empty = non_existing_as_empty

        images = [] if images is None else images
        texts = [] if texts is None else texts

        # when evaluating, only texts are set via --gt argument      --> need dummy [None] imgs
        # when predicting, only imags are set via --files  argument  --> need dummy [None] texts

        if mode == PipelineMode.Prediction:
            texts = [None] * len(images)

        if mode == PipelineMode.Targets:
            images = [None] * len(texts)

        for image, text in zip(images, texts):
            try:
                if image is None and text is None:
                    raise Exception("An empty data point is not allowed. Both image and text file are None")

                img_bn, text_bn = None, None
                if image:
                    img_path, img_fn = os.path.split(image)
                    img_bn, img_ext = split_all_ext(img_fn)

                    if not self._non_existing_as_empty and not os.path.exists(image):
                        raise Exception("Image at '{}' must exist".format(image))

                if text:
                    if not self._non_existing_as_empty and not os.path.exists(text):
                        raise Exception("Text file at '{}' must exist".format(text))

                    text_path, text_fn = os.path.split(text)
                    text_bn, text_ext = split_all_ext(text_fn)

                if image and text and img_bn != text_bn:
                    raise Exception("Expected image base name equals text base name but got '{}' != '{}'".format(
                        img_bn, text_bn
                    ))
            except Exception as e:
                logger.exception(e)
                if self.skip_invalid:
                    logger.warning("Exception raised. Invalid data. Skipping invalid example.")
                    continue
                else:
                    raise e

            self.add_sample({
                "image_path": image,
                "text_path": text,
                "id": image or text,
                "base_name": img_bn or text_bn,
            })

    def _load_sample(self, sample, text_only):
        if text_only:
            yield InputSample(None,
                              self._load_gt_txt(sample["text_path"]),
                              SampleMeta(sample['id']),
                              )
        else:
            yield InputSample(self._load_line(sample["image_path"]),
                              self._load_gt_txt(sample["text_path"]),
                              SampleMeta(sample['id']),
                              )

    def _load_gt_txt(self, gt_txt_path):
        if gt_txt_path is None:
            return None

        if not os.path.exists(gt_txt_path):
            if self._non_existing_as_empty:
                return ""
            else:
                raise Exception("Text file at '{}' does not exist".format(gt_txt_path))

        with codecs.open(gt_txt_path, 'r', 'utf-8') as f:
            return f.read()

    def _load_line(self, image_path):
        if image_path is None:
            return None

        if not os.path.exists(image_path):
            if self._non_existing_as_empty:
                return np.zeros((1, 1))
            else:
                raise Exception("Image file at '{}' does not exist".format(image_path))

        try:
            img = load_image(image_path)
        except:
            return None

        return img
