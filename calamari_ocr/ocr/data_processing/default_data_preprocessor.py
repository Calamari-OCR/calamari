from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.data_preprocessor import MultiDataProcessor
from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation


def DefaultDataPreprocessor(line_height: int, pad: int):
    return MultiDataProcessor(
            [
                DataRangeNormalizer(),
                CenterNormalizer(line_height),
                FinalPreparation(
                    True,
                    True,
                    True,
                    pad,
                    0,
                ),
            ]
        )
