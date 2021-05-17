import logging
from dataclasses import dataclass, field

import numpy as np
from paiargparse import PAIArgumentParser, pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import PipelineMode
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

from calamari_ocr import __version__
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class Args:
    data: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(choices=DATA_GENERATOR_CHOICES, mode="flat"),
    )


def main(args=None):
    parser = PAIArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s v" + __version__)
    parser.add_root_argument("args", Args)
    parser.add_argument("--line_height", type=int, default=48, help="The line height")
    parser.add_argument("--pad", type=int, default=16, help="Padding (left right) of the line")

    args = parser.parse_args(args=args)

    data: CalamariDataGeneratorParams = args.args.data
    gen = data.create(PipelineMode.EVALUATION)

    logger.info(f"Loading {len(data)} files")
    images, texts, metas = list(
        zip(
            *map(
                lambda s: (s.inputs, s.targets, s.meta),
                tqdm_wrapper(gen.generate(), progress_bar=True, total=len(gen)),
            )
        )
    )
    statistics = {
        "n_lines": len(images),
        "chars": [len(c) for c in texts],
        "widths": [
            img.shape[1] / img.shape[0] * args.line_height + 2 * args.pad
            for img in images
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0
        ],
        "total_line_width": 0,
        "char_counts": {},
    }

    for image, text in zip(images, texts):
        for c in text:
            if c in statistics["char_counts"]:
                statistics["char_counts"][c] += 1
            else:
                statistics["char_counts"][c] = 1

    statistics["av_line_width"] = np.average(statistics["widths"])
    statistics["max_line_width"] = np.max(statistics["widths"])
    statistics["min_line_width"] = np.min(statistics["widths"])
    statistics["total_line_width"] = np.sum(statistics["widths"])

    statistics["av_chars"] = np.average(statistics["chars"])
    statistics["max_chars"] = np.max(statistics["chars"])
    statistics["min_chars"] = np.min(statistics["chars"])
    statistics["total_chars"] = np.sum(statistics["chars"])

    statistics["av_px_per_char"] = statistics["av_line_width"] / statistics["av_chars"]
    statistics["codec_size"] = len(statistics["char_counts"])

    del statistics["chars"]
    del statistics["widths"]

    print(statistics)
    return statistics


if __name__ == "__main__":
    main()
