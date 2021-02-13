import unicodedata
from dataclasses import dataclass, field
from typing import Type

from paiargparse import pai_dataclass, pai_meta
from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor


@pai_dataclass
@dataclass
class TextNormalizer(DataProcessorParams):
    unicode_normalization: str = field(default="NFC", metadata=pai_meta(
        help="Unicode text normalization to apply. Defaults to NFC"))

    @staticmethod
    def cls() -> Type['TextProcessor']:
        return Impl


class Impl(TextProcessor[TextNormalizer]):
    def _apply_single(self, txt, meta):
        txt = unicodedata.normalize(
            self.params.unicode_normalization,
            txt
        )

        return txt
