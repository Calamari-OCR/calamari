from tfaip.data.pipeline.processor.dataprocessor import SequenceProcessor
from tfaip.data.pipeline.processor.pipeline import DataProcessorPipeline

from calamari_ocr.ocr.dataset.imageprocessors.data_preprocessor import ImageProcessor


class OutputToInputTransformer:
    def __init__(self, data_processing: DataProcessorPipeline):
        self.data_processing = data_processing
        self.processors = []
        for p in self.data_processing.pipeline:
            proc = p.create_processor_fn()
            if isinstance(proc, SequenceProcessor):
                self.processors.extend(proc.processors)
            else:
                self.processors.extend(proc)

        self.processors = list(filter(lambda p: isinstance(p, ImageProcessor), self.processors))

    def local_to_global(self, x, model_factor, data_proc_params):
        assert model_factor >= 1  # Should never be < 0, this would mean, that the network increases the size
        x *= model_factor
        # reverse
        for processor in self.processors:
            x = processor.local_to_global_pos(x, data_proc_params)
        return x
