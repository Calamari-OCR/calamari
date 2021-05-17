from typing import Iterable

from tfaip.data.pipeline.datapipeline import DataPipeline, DataGenerator
from tfaip.data.pipeline.definitions import Sample, PipelineMode

from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams


class CalamariPipeline(DataPipeline):
    def __init__(
        self,
        pipeline_params,
        data_base,
        generator_params,
        input_processors=None,
        output_processors=None,
    ):
        super(CalamariPipeline, self).__init__(
            pipeline_params,
            data_base,
            generator_params,
            input_processors,
            output_processors,
        )
        if generator_params and isinstance(generator_params, CalamariDataGeneratorParams):
            generator_params.n_folds = data_base.params.ensemble
        self._reader = None
        self._output_processors.run_parallel = (
            False  # TODO: parallel support, but currently in voter this makes one prediction per pipeline, mega slow
        )

    def reader(self):
        if self._reader is None:
            self._reader = self.generator_params.create(self.mode)

        return self._reader

    def create_data_generator(self) -> DataGenerator:
        reader = self.reader()

        class Gen(DataGenerator):
            def __len__(self):
                return len(reader)

            def generate(self) -> Iterable[Sample]:
                # Depending on the mode, do not produce images or targets (force it for the future pipeline)
                if self.mode == PipelineMode.PREDICTION:
                    return map(
                        lambda s: Sample(inputs=s.inputs, meta=s.meta),
                        reader.generate(),
                    )
                elif self.mode == PipelineMode.TARGETS:
                    return map(
                        lambda s: Sample(targets=s.targets, meta=s.meta),
                        reader.generate(),
                    )

                return reader.generate()

        return Gen(self.mode, self.generator_params)
