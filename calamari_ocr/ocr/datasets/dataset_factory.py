from enum import Enum

from .dataset import DataSet, RawDataSet, DataSetMode
from .file_dataset import FileDataSet
from .abbyy_dataset import AbbyyDataSet
from .pagexml_dataset import PageXMLDataset
from calamari_ocr.utils import keep_files_with_same_file_name


class DataSetType(Enum):
    RAW = 0
    FILE = 1
    ABBYY = 2
    PAGEXML = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DataSetType[s]
        except KeyError:
            raise ValueError()

    @staticmethod
    def files(type):
        files_meta = {
            DataSetType.RAW: False,
            DataSetType.FILE: True,
            DataSetType.ABBYY: True,
            DataSetType.PAGEXML: True,
        }

        return files_meta[type]

    @staticmethod
    def gt_extension(type):
        return {
            DataSetType.RAW: None,
            DataSetType.FILE: ".gt.txt",
            DataSetType.ABBYY: ".abbyy.xml",
            DataSetType.PAGEXML: ".xml",
        }[type]


def create_dataset(type: DataSetType,
                   mode: DataSetMode,
                   images = list(),
                   texts = list(),
                   skip_invalid=False,
                   remove_invalid=True,
                   non_existing_as_empty=False,
                   args = dict(),
                   ):
    if DataSetType.files(type):
        if images:
            images.sort()

        if texts:
            texts.sort()

        if images and texts and len(images) > 0 and len(texts) > 0:
            images, texts = keep_files_with_same_file_name(images, texts)

    if type == DataSetType.RAW:
        return RawDataSet(mode, images, texts)

    elif type == DataSetType.FILE:
        return FileDataSet(mode, images, texts,
                           skip_invalid=skip_invalid,
                           remove_invalid=remove_invalid,
                           non_existing_as_empty=non_existing_as_empty)
    elif type == DataSetType.ABBYY:
        return AbbyyDataSet(mode, images, texts,
                            skip_invalid=skip_invalid,
                            remove_invalid=remove_invalid,
                            non_existing_as_empty=non_existing_as_empty)
    elif type == DataSetType.PAGEXML:
        return PageXMLDataset(mode, images, texts,
                              skip_invalid=skip_invalid,
                              remove_invalid=remove_invalid,
                              non_existing_as_empty=non_existing_as_empty,
                              args=args)
    else:
        raise Exception("Unsuppoted dataset type {}".format(type))
