from typing import Iterable

from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator
from tfaip.base.data.pipeline.definitions import Sample, PipelineMode

from calamari_ocr.ocr.dataset.datareader.factory import data_reader_from_params


class CalamariPipeline(DataPipeline):
    def __init__(self,
                 mode: PipelineMode,
                 data_base,
                 generator_params,
                 input_processors=None,
                 output_processors=None,
                 ):
        super(CalamariPipeline, self).__init__(mode, data_base, generator_params, input_processors, output_processors)
        self._reader = None
        self._output_processors.run_parallel = False  # TODO: parallel support, but currently in voter this makes one prediction per pipeline, mega slow

    def reader(self):
        if self._reader is None:
            self._reader = data_reader_from_params(self.mode, self.generator_params)

        return self._reader

    def create_data_generator(self) -> DataGenerator:
        reader = self.reader()

        class Gen(DataGenerator):
            def __len__(self):
                return len(reader)

            def generate(self) -> Iterable[Sample]:
                # Depending on the mode, do not produce images or targets (force it for the future pipeline)
                if self.mode == PipelineMode.Prediction:
                    return map(lambda s: Sample(s.first, None, s.meta), reader.generate())
                elif self.mode == PipelineMode.Targets:
                    return map(lambda s: Sample(None, s.second, s.meta), reader.generate())

                return reader.generate()

        return Gen(self.mode, self.generator_params)
