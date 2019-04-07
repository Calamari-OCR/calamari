from enum import Enum
from typing import List

from .dataset import RawDataSet, DataSetMode
from .file_dataset import FileDataSet
from .abbyy_dataset import AbbyyDataSet
from .pagexml_dataset import PageXMLDataset
from .hdf5_dataset import Hdf5DataSet
from calamari_ocr.utils import keep_files_with_same_file_name


class DataSetType(Enum):
    RAW = 0
    FILE = 1
    ABBYY = 2
    PAGEXML = 3
    HDF5 = 4
    EXTENDED_PREDICTION = 5
    GENERATED_LINE = 6

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
            DataSetType.EXTENDED_PREDICTION: True,
            DataSetType.HDF5: False,
            DataSetType.GENERATED_LINE: False,
        }

        return files_meta[type]

    @staticmethod
    def gt_extension(type):
        return {
            DataSetType.RAW: None,
            DataSetType.FILE: ".gt.txt",
            DataSetType.ABBYY: ".abbyy.xml",
            DataSetType.PAGEXML: ".xml",
            DataSetType.EXTENDED_PREDICTION: ".json",
            DataSetType.HDF5: ".h5",
            DataSetType.GENERATED_LINE: None,
        }[type]


def create_dataset(type: DataSetType,
                   mode: DataSetMode,
                   images: List[str] = None,
                   texts: List[str] = None,
                   skip_invalid=False,
                   remove_invalid=True,
                   non_existing_as_empty=False,
                   args: dict = None,
                   ):
    if images is None:
        images = []

    if texts is None:
        texts = []

    if args is None:
        args = dict()

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
    elif type == DataSetType.HDF5:
        return Hdf5DataSet(mode, images, texts)
    elif type == DataSetType.EXTENDED_PREDICTION:
        from .extended_prediction_dataset import ExtendedPredictionDataSet
        return ExtendedPredictionDataSet(texts=texts)
    elif type == DataSetType.GENERATED_LINE:
        from .generated_line_dataset import GeneratedLineDataset
        return GeneratedLineDataset(mode, args=args)
    else:
        raise Exception("Unsupported dataset type {}".format(type))
