from tfaip.base.data.pipeline.processor.pipeline import DataProcessorPipeline

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


class OutputToInputTransformer:
    def __init__(self, data_processing: DataProcessorPipeline):
        self.data_processing = data_processing

    def local_to_global(self, x, model_factor, data_proc_params):
        assert(model_factor >= 1)  # Should never be < 0, this would mean, that the network increases the size
        x *= model_factor
        if self.data_processing:
            # reverse
            for processor in self.data_processing[0].processors[::-1]:
                if isinstance(processor, ImageProcessor):
                    x = processor.local_to_global_pos(x, data_proc_params)
        return x
