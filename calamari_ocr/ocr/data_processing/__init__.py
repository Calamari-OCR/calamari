from inspect import isclass
from typing import List, Type
import logging

from tfaip.util.enum import StrEnum

from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor, NoopDataPreprocessor, MultiDataProcessor
from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation
from calamari_ocr.ocr.data_processing.scale_to_height_processor import ScaleToHeightProcessor


logger = logging.getLogger(__name__)


DataPreprocessors = StrEnum('DataPreprocessors',
                            {
                                **{k: k for k, v in globals().items() if isclass(v) and issubclass(v, DataPreprocessor) and v != DataPreprocessor},
                                **{'DefaultDataPreprocessor': 'DefaultDataPreprocessor'},
                             })


def data_processor_cls(s: str):
    return globals()[s]


def data_processor_from_dict(d: dict):
    return DataPreprocessor.from_dict(d)


def default_data_preprocessors() -> List[DataPreprocessors]:
    return list(map(DataPreprocessors, map(lambda x: x.__name__, [DataRangeNormalizer, CenterNormalizer, FinalPreparation])))


def data_processor_from_list(line_height: int, pad: int, processors: List[DataPreprocessors] = None) -> DataPreprocessor:
    if processors is None:
        processors = default_data_preprocessors()
    else:
        processors = [data_processor_cls(p.value) for p in processors]

    mp = MultiDataProcessor()
    for p in processors:
        if p == DataRangeNormalizer:
            mp.sub_processors.append(p())
        elif p == CenterNormalizer:
            mp.sub_processors.append(p(line_height))
        elif p == FinalPreparation:
            mp.sub_processors.append(p(True, True, True, pad, 0))
        else:
            logger.warning(f"Unknown processor {p.__name__}. Creation might fail.")
            mp.sub_processors.append(p())

    return mp
