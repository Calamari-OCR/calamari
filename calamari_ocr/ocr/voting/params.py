from dataclasses import dataclass

from dataclasses_json import dataclass_json
from tfaip.util.enum import StrEnum


class VoterType(StrEnum):
    SequenceVoter = "sequence_voter"
    ConfidenceVoterDefaultCTC = "confidence_voter_default_ctc"


@dataclass_json
@dataclass
class VoterParams:
    type: VoterType = VoterType.ConfidenceVoterDefaultCTC
    blank_index: int = 0
