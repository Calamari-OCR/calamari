from calamari_ocr.ocr.voting.params import VoterParams, VoterType
from calamari_ocr.ocr.voting.sequence_voter import SequenceVoter, Voter
from calamari_ocr.ocr.voting.confidence_voter import ConfidenceVoter


def voter_from_params(voter_params: VoterParams) -> Voter:
    if voter_params.type == VoterType.SequenceVoter:
        return SequenceVoter()
    elif voter_params.type == VoterType.ConfidenceVoterDefaultCTC:
        return ConfidenceVoter(blank_index=voter_params.blank_index)
    else:
        raise Exception("Unknown voter type '{}'".format(voter_params.type))
