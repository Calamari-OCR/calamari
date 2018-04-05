from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import MultiDataProcessor
from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation


class DefaultDataPreprocessor(MultiDataProcessor):
    def __init__(self, line_height, pad=0):
        super().__init__(
            [
                DataRangeNormalizer(),
                CenterNormalizer(target_height=line_height),
                FinalPreparation(pad=pad),
            ]
        )
