from dataclasses import dataclass, field
from typing import List, Type

from paiargparse import pai_dataclass
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams

from calamari_ocr.ocr.dataset.textprocessors import TextProcessor


@pai_dataclass
@dataclass
class StrToCharListProcessorParams(DataProcessorParams):
    chars: List[str] = field(default_factory=list)

    @staticmethod
    def cls() -> Type["TextProcessor"]:
        return StrToCharListProcessor


class StrToCharListProcessor(TextProcessor[StrToCharListProcessorParams]):
    def _apply_single(self, txt, meta):
        index = 0
        out = []
        while index < len(txt):
            found = False
            for char in self.params.chars:
                if len(char) == 0:
                    continue  # blank
                if txt[index : index + len(char)] == char:
                    out.append(char)
                    index += len(char)
                    found = True
                    break

            if found:
                continue

            else:
                raise Exception("Could not parse remainder '{}' of '{}'".format(txt[index:], txt))

        return out
