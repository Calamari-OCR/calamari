from dataclasses import dataclass, field
from typing import Callable, List, Optional
from math import ceil
import numpy as np

import tfaip as tfaip
from dataclasses_json import dataclass_json
from paiargparse import pai_meta
from tfaip.data.pipeline.definitions import Sample


@dataclass_json
@dataclass
class PredictionCharacter:
    char: str = ""
    label: int = 0
    probability: float = 0

    def __post_init__(self):
        self.probability = float(self.probability)
        self.label = int(self.label)


@dataclass_json
@dataclass
class PredictionPosition:
    chars: List[PredictionCharacter] = field(default_factory=list)
    local_start: int = 0
    local_end: int = 0
    global_start: int = 0
    global_end: int = 0
    global_start_ext: int = 0
    global_end_ext: int = 0


@dataclass_json
@dataclass
class Prediction:
    id: str = ""
    sentence: str = ""
    labels: List[int] = field(default_factory=list)
    positions: List[PredictionPosition] = field(default_factory=list)
    logits: Optional[np.array] = field(default=None)
    total_probability: float = 0
    avg_char_probability: float = 0
    is_voted_result: bool = False
    line_path: str = ""
    voter_predictions: Optional[List["Prediction"]] = None


@dataclass_json
@dataclass
class Predictions:
    predictions: List[Prediction] = field(default_factory=list)
    line_path: str = ""


@dataclass_json
@dataclass
class PredictorParams(tfaip.PredictorParams):
    # override defaults
    silent: bool = field(default=True, metadata=pai_meta(mode="ignore"))


class PredictionResult:
    def __init__(
        self,
        prediction,
        codec,
        text_postproc,
        out_to_in_trans: Callable[[int], int],
        ground_truth=None,
    ):
        """The output of a networks prediction (PredictionProto) with additional information

        It stores all required information for decoding (`codec`) and interpreting the output.

        Parameters
        ----------
        prediction : PredictionProto
            prediction the DNN
        codec : Codec
            codec required to decode the `prediction`
        text_postproc : TextPostprocessor
            text processor to apply to the decodec `prediction` to receive the actual prediction sentence
        """
        self.prediction = prediction
        self.logits = prediction.logits
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply_on_sample(Sample(inputs="", outputs="".join(self.chars))).outputs
        self.prediction.sentence = self.sentence
        self.ground_truth = ground_truth

        self.prediction.avg_char_probability = 0

        last_p = None
        for n, p in enumerate(self.prediction.positions):
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(out_to_in_trans(p.local_start))
            p.global_end = ceil(out_to_in_trans(p.local_end))

            p_len = max(1, p.global_end - p.global_start)
            if n == 0:
                p.global_start_ext = max(0, (p.global_start - p_len) // 2)
            else:
                p.global_start_ext = (p.global_start + last_p.global_end) // 2
                last_p.global_end_ext = p.global_start_ext

            if n == len(self.prediction.positions) - 1:
                line_len = out_to_in_trans(self.logits.shape[0])
                p.global_end_ext = min(line_len - 1, ceil((line_len + p.global_end + p_len) / 2))

            if len(p.chars) > 0:
                self.prediction.avg_char_probability += p.chars[0].probability

            last_p = p

        self.prediction.avg_char_probability /= (
            len(self.prediction.positions) if len(self.prediction.positions) > 0 else 1
        )
