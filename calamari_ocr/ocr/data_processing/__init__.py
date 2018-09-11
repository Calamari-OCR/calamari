from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import DataPreprocessor, NoopDataPreprocessor, MultiDataProcessor
from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation

from calamari_ocr.ocr.data_processing.default_data_preprocessor import DefaultDataPreprocessor

from calamari_ocr.proto import DataPreprocessorParams


def data_processor_from_proto(data_preprocessor_params):
    if data_preprocessor_params.type == DataPreprocessorParams.MULTI_NORMALIZER:
        return MultiDataProcessor(
            [data_processor_from_proto(c) for c in data_preprocessor_params.children]
        )
    elif data_preprocessor_params.type == DataPreprocessorParams.DEFAULT_NORMALIZER:
        return DefaultDataPreprocessor(data_preprocessor_params)
    elif data_preprocessor_params.type == DataPreprocessorParams.NOOP_NORMALIZER:
        return NoopDataPreprocessor()
    elif data_preprocessor_params.type == DataPreprocessorParams.RANGE_NORMALIZER:
        return DataRangeNormalizer()
    elif data_preprocessor_params.type == DataPreprocessorParams.CENTER_NORMALIZER:
        return CenterNormalizer(data_preprocessor_params)
    elif data_preprocessor_params.type == DataPreprocessorParams.FINAL_PREPARATION:
        return FinalPreparation(data_preprocessor_params)

    raise Exception("Unknown proto type {} of an data processor".format(data_preprocessor_params.type))

