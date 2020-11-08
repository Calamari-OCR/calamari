from typing import TYPE_CHECKING, List

from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.backends.dataset.datareader.base import DataReader
from calamari_ocr.ocr.datasets.datasetype import DataSetType
from calamari_ocr.utils import keep_files_with_same_file_name


def create_data_reader(type: DataSetType,
                       mode: PipelineMode,
                       images: List[str] = None,
                       texts: List[str] = None,
                       skip_invalid=False,
                       remove_invalid=True,
                       non_existing_as_empty=False,
                       args: dict = None,
                       ) -> DataReader:
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
        from calamari_ocr.ocr.backends.dataset.datareader.raw import RawDataReader
        return RawDataReader(mode, images, texts)

    elif type == DataSetType.FILE:
        from calamari_ocr.ocr.backends.dataset.datareader.file import FileDataReader
        return FileDataReader(mode, images, texts,
                              skip_invalid=skip_invalid,
                              remove_invalid=remove_invalid,
                              non_existing_as_empty=non_existing_as_empty)
    elif type == DataSetType.ABBYY:
        from calamari_ocr.ocr.backends.dataset.datareader.abbyy import AbbyyReader
        return AbbyyReader(mode, images, texts,
                           skip_invalid=skip_invalid,
                           remove_invalid=remove_invalid,
                           non_existing_as_empty=non_existing_as_empty)
    elif type == DataSetType.PAGEXML:
        from calamari_ocr.ocr.backends.dataset.datareader.pagexml.reader import PageXMLReader
        return PageXMLReader(mode, images, texts,
                             skip_invalid=skip_invalid,
                             remove_invalid=remove_invalid,
                             non_existing_as_empty=non_existing_as_empty,
                             args=args)
    elif type == DataSetType.HDF5:
        from calamari_ocr.ocr.backends.dataset.datareader.hdf5 import Hdf5Reader
        return Hdf5Reader(mode, images, texts)
    elif type == DataSetType.EXTENDED_PREDICTION:
        from .extended_prediction_dataset import ExtendedPredictionDataSet
        return ExtendedPredictionDataSet(texts=texts)
    elif type == DataSetType.GENERATED_LINE:
        from .generated_line_dataset import GeneratedLineDataset
        return GeneratedLineDataset(mode, args=args)
    else:
        raise Exception("Unsupported dataset type {}".format(type))
