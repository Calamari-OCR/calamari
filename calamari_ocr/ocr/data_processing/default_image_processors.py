from typing import List

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, inputs_pipeline_modes

from calamari_ocr.ocr.data_processing.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.data_processing.data_range_normalizer import DataRangeNormalizer
from calamari_ocr.ocr.data_processing.final_preparation import FinalPreparation


def default_image_processors() -> List[DataProcessorFactoryParams]:
    return [
        DataProcessorFactoryParams(DataRangeNormalizer.__name__, inputs_pipeline_modes),
        DataProcessorFactoryParams(CenterNormalizer.__name__, inputs_pipeline_modes),
        DataProcessorFactoryParams(FinalPreparation.__name__, inputs_pipeline_modes),
    ]
