from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import MultiDataProcessor
from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation


class DefaultDataPreprocessor(MultiDataProcessor):
    def __init__(self, params):
        super().__init__(
            [
                DataRangeNormalizer(),
                CenterNormalizer(params),
                FinalPreparation(params),
            ]
        )
