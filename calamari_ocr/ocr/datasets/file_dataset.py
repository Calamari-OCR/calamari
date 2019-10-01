import codecs
import os
from PIL import Image
import numpy as np

from calamari_ocr.utils import split_all_ext
from .dataset import DataSet, DataSetMode, DatasetGenerator


class FileDataSetGenerator(DatasetGenerator):
    def __init__(self, mp_context, output_queue, mode: DataSetMode, samples, non_existing_as_empty):
        super().__init__(mp_context, output_queue, mode, samples)
        self._non_existing_as_empty = non_existing_as_empty

    def _load_sample(self, sample, text_only):
        if text_only:
            yield None, self._load_gt_txt(sample["text_path"])
        else:
            yield self._load_line(sample["image_path"]), \
                  self._load_gt_txt(sample["text_path"])

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

        i = Image.open(image_path)
        try:
            i.load()
            img = np.array(i)
        except:
            return None
        finally:
            i.close()

        return img


class FileDataSet(DataSet):
    def __init__(self, mode: DataSetMode,
                 images=None, texts=None,
                 skip_invalid=False, remove_invalid=True,
                 non_existing_as_empty=False):
        """ Create a dataset from a list of files

        Images or texts may be empty to create a dataset for prediction or evaluation only.

        Parameters
        ----------
        images : list of str, optional
            image files
        texts : list of str, optional
            text files
        skip_invalid : bool, optional
            skip invalid files
        remove_invalid : bool, optional
            remove invalid files
        non_existing_as_empty : bool, optional
            tread non existing files as empty. This is relevant for evaluation a dataset
        """
        super().__init__(mode,
                         skip_invalid=skip_invalid,
                         remove_invalid=remove_invalid)
        self._non_existing_as_empty = non_existing_as_empty

        images = [] if images is None else images
        texts = [] if texts is None else texts

        # when evaluating, only texts are set via --gt argument      --> need dummy [None] imgs
        # when predicting, only imags are set via --files  argument  --> need dummy [None] texts
        
        if (mode == DataSetMode.PREDICT or mode == DataSetMode.PRED_AND_EVAL) and not texts:
            texts = [None] * len(images)

        if (mode == DataSetMode.EVAL or mode == DataSetMode.PRED_AND_EVAL) and not images: 
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
                if self.skip_invalid:
                    print("Invalid data: {}".format(e))
                    continue
                else:
                    raise e

            self.add_sample({
                "image_path": image,
                "text_path": text,
                "id": img_bn if image else text_bn,
            })

    def create_generator(self, mp_context, output_queue):
        return FileDataSetGenerator(mp_context, output_queue, self.mode, self.samples(), self._non_existing_as_empty)
