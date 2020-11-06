import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING, Type
import logging

from dataclasses_json import dataclass_json

from calamari_ocr.ocr.datasets.dataset_factory import DataSetMode
from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.proto.params import LineGeneratorParams, TextGeneratorParams

from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
if TYPE_CHECKING:
    from calamari_ocr.ocr.backends.dataset.datareader import DataReader

logger = logging.getLogger(__name__)


class DataReaderFactory(ABC):
    factories: Dict[str, Type['DataReaderFactory']] = {}

    @classmethod
    def register(cls, factory: Type['DataReaderFactory']):
        cls.factories[factory.__name__] = factory

    @classmethod
    def from_dict(cls, d) -> 'DataReaderFactory':
        df_cls = DataReaderFactory.factories[d['type']]
        return df_cls(**d['params'])

    def to_dict(self):
        return {
            'type': self.__class__.__name__,
            'params': self._to_dict(),
        }

    @abstractmethod
    def _to_dict(self):
        raise NotImplementedError

    @abstractmethod
    def create(self) -> 'DataReader':
        raise NotImplementedError


class RawDataReaderFactory(DataReaderFactory):
    def _to_dict(self):
        logger.waring("A raw data factory can not be serialized")
        return {}

    def __init__(self, reader: 'DataReader'):
        self.reader = reader

    def create(self) -> 'DataReader':
        return self.reader


@dataclass_json
@dataclass
class FileDataReaderArgs:
    line_generator_params: LineGeneratorParams = field(default_factory=LineGeneratorParams)
    text_generator_params: TextGeneratorParams = field(default_factory=TextGeneratorParams)
    pad: int = 0
    text_index: int = 0


class FileDataReaderFactory(DataReaderFactory):
    def _to_dict(self):
        return {
            'files': self.files,
            'text_files': self.text_files,
            'data_set_type': self.data_set_type,
            'data_set_mode': self.data_set_mode,
            'gt_extension': self.gt_extension,
            'skip_invalid': self.skip_invalid,
            'data_reader_args': self.data_reader_args.to_dict() if self.data_reader_args else None,
        }

    @classmethod
    def from_dict(cls, d):
        if d['data_reader_args']:
            d['data_reader_args'] = FileDataReaderArgs.from_dict(d['data_reader_args'])
        super(FileDataReaderFactory, cls).from_dict(d)


    def __init__(self,
                 data_set_type: DataSetType,
                 data_set_mode: DataSetMode,
                 files: List[str],
                 text_files: Optional[List[str]] = None,
                 skip_invalid: bool = True,
                 gt_extension: Optional[str] = None,
                 data_reader_args: Optional[FileDataReaderArgs] = None,
                 ):
        self.data_set_type = data_set_type
        self.data_set_mode = data_set_mode
        self.files = files
        self.text_files = text_files
        self.skip_invalid = skip_invalid
        self.gt_extension = gt_extension if gt_extension is not None else DataSetType.gt_extension(data_set_type)
        self.data_reader_args = data_reader_args

    def create(self) -> 'DataReader':
        from calamari_ocr.ocr.datasets.dataset_factory import create_data_reader
        # Training dataset
        logger.info("Resolving input files")
        input_image_files = sorted(glob_all(self.files))
        if not self.text_files:
            if self.gt_extension:
                gt_txt_files = [split_all_ext(f)[0] + self.gt_extension for f in input_image_files]
            else:
                gt_txt_files = [None] * len(input_image_files)
        else:
            gt_txt_files = sorted(glob_all(self.text_files))
            input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
            for img, gt in zip(input_image_files, gt_txt_files):
                if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                    raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

        if len(set(gt_txt_files)) != len(gt_txt_files):
            raise Exception("Some image are occurring more than once in the data set.")

        dataset = create_data_reader(
            self.data_set_type,
            self.data_set_mode,
            images=input_image_files,
            texts=gt_txt_files,
            skip_invalid=self.skip_invalid,
            args=self.data_reader_args if self.data_reader_args else FileDataReaderArgs()
        )
        logger.info(f"Found {len(dataset)} files in the dataset")
        return dataset


DataReaderFactory.register(FileDataReaderFactory)
