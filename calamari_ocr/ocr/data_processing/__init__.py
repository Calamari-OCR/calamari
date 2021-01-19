from inspect import isclass

from tfaip.util.enum import StrEnum

from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor, NoopDataPreprocessor, MultiDataProcessor
from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation
from calamari_ocr.ocr.data_processing.scale_to_height_processor import ScaleToHeightProcessor

from calamari_ocr.ocr.data_processing.default_data_preprocessor import DefaultDataPreprocessor

from calamari_ocr.proto import DataPreprocessorParams

DataPreprocessors = StrEnum('DataPreprocessors',
                            {
                                **{k: k for k, v in globals().items() if isclass(v) and issubclass(v, DataPreprocessor) and v != DataPreprocessor},
                                **{'DefaultDataPreprocessor': 'DefaultDataPreprocessor'},
                             })


def data_processor_cls(s: str):
    return globals()[s]


def data_processor_from_dict(d: dict):
    return DataPreprocessor.from_dict(d)


def data_processor_from_proto(data_preprocessor_params):
    if len(data_preprocessor_params.children) > 0 and data_preprocessor_params.type != DataPreprocessorParams.MULTI_NORMALIZER:
        raise ValueError("Only a MULTI_NORMALIZER may have children, however got {}".format(
            DataPreprocessorParams.Type.Name(data_preprocessor_params.type)))

    if data_preprocessor_params.type == DataPreprocessorParams.MULTI_NORMALIZER:
        return MultiDataProcessor(
            [data_processor_from_proto(c) for c in data_preprocessor_params.children]
        )
    elif data_preprocessor_params.type == DataPreprocessorParams.DEFAULT_NORMALIZER:
        return DefaultDataPreprocessor(data_preprocessor_params.line_height, data_preprocessor_params.pad)
    elif data_preprocessor_params.type == DataPreprocessorParams.NOOP_NORMALIZER:
        return NoopDataPreprocessor()
    elif data_preprocessor_params.type == DataPreprocessorParams.RANGE_NORMALIZER:
        return DataRangeNormalizer()
    elif data_preprocessor_params.type == DataPreprocessorParams.CENTER_NORMALIZER:
        return CenterNormalizer(data_preprocessor_params.line_height)
    elif data_preprocessor_params.type == DataPreprocessorParams.FINAL_PREPARATION:
        return FinalPreparation(data_preprocessor_params)
    elif data_preprocessor_params.type == DataPreprocessorParams.SCALE_TO_HEIGHT:
        return ScaleToHeightProcessor(data_preprocessor_params)

    raise Exception("Unknown proto type {} of an data processor".format(data_preprocessor_params.type))

