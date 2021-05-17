from dataclasses import dataclass, field
from typing import Type

import bidi.algorithm as bidi_algorithm
from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams
from tfaip.util.enum import StrEnum

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor


@pai_dataclass
@dataclass
class StripTextProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["TextProcessor"]:
        return StripTextProcessor


class StripTextProcessor(TextProcessor[StripTextProcessorParams]):
    def _apply_single(self, txt, meta):
        if isinstance(txt, str):
            return txt.strip()

        elif isinstance(txt, list):
            while txt[0].isspace():
                del txt[0]

            while txt[-1].isspace():
                del txt[-1]

            return txt

        else:
            raise TypeError()


class BidiDirection(StrEnum):
    LTR = "L"
    RTL = "R"
    AUTO = "auto"


@pai_dataclass
@dataclass
class BidiTextProcessorParams(DataProcessorParams):
    bidi_direction: BidiDirection = field(
        default=BidiDirection.AUTO,
        metadata=pai_meta(
            help="The default text direction when preprocessing bidirectional text. Supported values "
            "are 'auto' to automatically detect the direction, 'ltr' and 'rtl' for left-to-right and "
            "right-to-left, respectively"
        ),
    )

    @staticmethod
    def cls() -> Type["TextProcessor"]:
        return BidiTextProcessor


class BidiTextProcessor(TextProcessor[BidiTextProcessorParams]):
    def _apply_single(self, txt, meta):
        # To support arabic text
        return bidi_algorithm.get_display(
            txt,
            base_dir=self.params.bidi_direction.value if self.params.bidi_direction != BidiDirection.AUTO else None,
        )
