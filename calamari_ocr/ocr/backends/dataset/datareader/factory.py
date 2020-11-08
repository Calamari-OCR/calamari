import os
from typing import TYPE_CHECKING
import logging

from tfaip.base.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.backends.dataset.data_types import CalamariPipelineParams, FileDataReaderArgs
from calamari_ocr.ocr.backends.dataset.datareader.base import DataReader

from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name

logger = logging.getLogger(__name__)


def data_reader_from_params(mode: PipelineMode, params: CalamariPipelineParams) -> DataReader:
    from calamari_ocr.ocr.datasets.dataset_factory import create_data_reader
    # Training dataset
    logger.info("Resolving input files")
    input_image_files = sorted(glob_all(params.files))
    if not params.text_files:
        if params.gt_extension:
            gt_txt_files = [split_all_ext(f)[0] + params.gt_extension for f in input_image_files]
        else:
            gt_txt_files = [None] * len(input_image_files)
    else:
        gt_txt_files = sorted(glob_all(params.text_files))
        input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
        for img, gt in zip(input_image_files, gt_txt_files):
            if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

    if mode in {PipelineMode.Training, PipelineMode.Evaluation}:
        if len(set(gt_txt_files)) != len(gt_txt_files):
            raise Exception("Some image are occurring more than once in the data set.")

        dataset = create_data_reader(
            params.type,
            mode,
            images=input_image_files,
            texts=gt_txt_files,
            skip_invalid=params.skip_invalid,
            args=params.data_reader_args if params.data_reader_args else FileDataReaderArgs(),
        )
        logger.info(f"Found {len(dataset)} files in the dataset")
        return dataset
