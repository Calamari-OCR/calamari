import logging
import random
from multiprocessing import Process, Queue, Manager

import numpy as np
from tfaip.data.pipeline.definitions import PipelineMode

from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    InputSample,
    SampleMeta,
)
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset.params import (
    GeneratedLineDatasetParams,
)
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset.line_generator import (
    LineGenerator,
)
from calamari_ocr.ocr.dataset.datareader.generated_line_dataset.text_generation.text_generator import (
    TextGenerator,
)

logger = logging.getLogger(__name__)


class LineGeneratorProcess(Process):
    def __init__(self, output_queue: Queue, text_generator, line_generator, name=-1):
        super().__init__(daemon=True)
        self.text_generator = TextGenerator(text_generator)
        self.line_generator = LineGenerator(line_generator)
        self.output_queue = output_queue
        self.text_only = False
        self.name = "{}".format(name)

    def _handle(self):
        try:
            words = self.text_generator.generate()
            image = self.line_generator.draw(words) if not self.text_only else None
            self.output_queue.put((image, TextGenerator.words_to_unformatted_text(words)))
        except ValueError as e:
            logger.exception(e)
            raise

    def run(self):
        random.seed()
        np.random.seed()
        try:
            while True:
                self._handle()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            # queue closed, stop the process
            return


class GeneratedLineDataset(CalamariDataGenerator[GeneratedLineDatasetParams]):
    def __init__(
        self,
        mode: PipelineMode,
        params: GeneratedLineDatasetParams,
    ):
        """Create a dataset from memory
        Since this dataset already contains all data in the memory, this dataset may not be loaded
        Parameters
        ----------
        """
        super().__init__(mode, params)

        self._samples = [{"id": "{}".format(i)} for i in range(self.params.lines_per_epoch)]
        self.text_generator_params = self.params.text_generator
        self.line_generator_params = self.params.line_generator
        self.manager = Manager()
        self.data_queue = self.manager.Queue(50)
        self.data_generators = [
            LineGeneratorProcess(
                self.data_queue,
                self.text_generator_params,
                self.line_generator_params,
                "{}".format(i),
            )
            for i in range(8)
        ]
        for d in self.data_generators:
            d.start()

    def store_text_prediction(self, prediction, sample_id, output_dir):
        pass

    def _load_sample(self, sample, text_only):
        image, text = self.data_queue.get()
        fold_id = -1 if self.params.n_folds <= 0 else np.random.randint(self.params.n_folds)
        yield InputSample(image, text, SampleMeta(id=sample["id"], fold_id=fold_id))


if __name__ == "__main__":
    from calamari_ocr.ocr.dataset.datareader.generated_line_dataset.params import (
        TextGeneratorParams,
        LineGeneratorParams,
    )

    params = TextGeneratorParams()
    params.word_length_mean = 11
    params.word_length_sigma = 3
    params.number_of_words_mean = 7
    params.number_of_words_mean = 4
    params.word_separator = " "
    params.sub_script_p = 0.2
    params.super_script_p = 0.2
    params.letter_spacing_p = 0.5
    params.letter_spacing_mean = 0.5
    params.letter_spacing_sigma = 0.05
    params.bold_p = 0.5
    params.italic_p = 0.5
    params.charset = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]()_-.;:'\""
        "éèíìóòúù"
        "ăȁĕȅĭŏőŭű"
        "āĀǟǞēĒěīĪōŌȫȪūŪǖǕ"
        "ẹḲḳ"
        "αβγδεζηθικλμνξοπρστυφχψω"
        "½"
        "—"
        "–"
        "℔"
        "šŠ"
        "„“"
        "†"
    )

    dataset = GeneratedLineDatasetParams(
        lines_per_epoch=10,
        line_generator=LineGeneratorParams(
            font_size=48,
            min_script_offset=-0.5,
            max_script_offset=0.5,
            fonts=["Junicode.ttf", "DejaVuSerif.ttf"],
        ),
        text_generator=params,
    ).create(PipelineMode.TRAINING)

    import matplotlib.pyplot as plt

    for sample in dataset.generate():
        plt.imshow(sample.inputs)
        plt.title(sample.outputs)
        plt.show()
