from calamari_ocr.ocr.voting.sequence_voter import SequenceVoter
from calamari_ocr.ocr.voting.confidence_voter import ConfidenceVoter

from calamari_ocr.proto import VoterParams


def voter_from_proto(voter_params):
    if voter_params.type == VoterParams.SEQUENCE_VOTER:
        return SequenceVoter()
    elif voter_params.type == VoterParams.CONFIDENCE_VOTER_FUZZY_CTC:
        return ConfidenceVoter(fuzzy_ctc=True, blank_index=voter_params.blank_index)
    elif voter_params.type == VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC:
        return ConfidenceVoter(fuzzy_ctc=False, blank_index=voter_params.blank_index)
    else:
        raise Exception("Unknown voter type '{}'".format(voter_params.type))
