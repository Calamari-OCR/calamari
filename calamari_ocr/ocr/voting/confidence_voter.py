import numpy as np
import operator

from calamari_ocr.ocr.voting.voter import Voter
from calamari_ocr.ocr.text_processing.text_synchronizer import synchronize


def add_llocs(s, new):
    for char in new.keys():
        if char in s:
            s[char] += new[char]
        else:
            s[char] = new[char]


def find_voters_with_most_frequent_length(sync, voters):
    lengths = {}

    for i, voter in enumerate(voters):
        length = sync.length(i)

        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1

    most_freq = max(lengths.items(), key=operator.itemgetter(1))[0]

    return [i for i, voter in enumerate(voters) if sync.length(i) == most_freq], most_freq


class MergeableCharacter:
    def __init__(self, char, p, start, stop):
        self.char = char
        self.p = p
        self.start = start
        self.stop = stop

    def merge(self, char, p, start, stop):
        assert(self.char == char)
        self.p += p
        self.start = min(start, self.start)
        self.stop = max(stop, self.stop)


def perform_conf_vote(voters):
    results = [voter['sequence'] for voter in voters]
    synclist = synchronize(results)

    final_result = []

    for sync in synclist:
        actual_voters, most_freq_length = find_voters_with_most_frequent_length(sync, voters)

        # set of all characters (check if all say the same, then the set size is one)
        s = []
        for r in [voters[voter_id]['sequence'][sync.start(voter_id):sync.stop(voter_id)+1] for voter_id in actual_voters]:
            if r not in s:
                s.append(r)

        def add_char(actual_voters, i):
            c_p = {}
            for voter_id in actual_voters:
                idx = i + sync.start(voter_id)
                alts = voters[voter_id]['alternatives'][idx]
                pos = voters[voter_id]['positions'][idx]
                for k, p in alts.items():
                    if k in c_p:
                        c_p[k].merge(k, p / len(actual_voters), pos.global_start, pos.global_end)
                    else:
                        c_p[k] = MergeableCharacter(k, p / len(actual_voters), pos.global_start, pos.global_end)

            chars = sorted(c_p.values(), key=lambda v: -v.p)
            final_result.append(chars)

        if len(s) == 1:
            sentence = s.pop()
            for i in range(len(sentence)):
                add_char(actual_voters, i)

            continue

        if len(actual_voters) == 1:
            voter_id = actual_voters[0]
            for i in range(sync.length(voter_id)):
                add_char([voter_id], i)
        else:
            for i in range(most_freq_length):
                add_char(actual_voters, i)

    return final_result


class ConfidenceVoter(Voter):
    def __init__(self, blank_index=0, fuzzy_ctc=False, blank_threshold=0.7):
        super().__init__()
        self.fuzzy_ctc = fuzzy_ctc
        self.blank_threshold = blank_threshold
        self.blank_index = blank_index
        self.min_candidate_probability = 1e-4

    def _apply_vote(self, predictions, prediction_out):
        def extract_data(prediction):
            # We need to vote by chars not labels, because the labels
            # can be different for different predictions, but labels are universal
            alternatives = []
            for pos in prediction.prediction.positions:
                d = {}
                alternatives.append(d)
                for c in pos.chars:
                    d[c.char] = c.probability

            return {"sequence": prediction.chars,
                    "alternatives": alternatives,
                    "positions": prediction.prediction.positions, }

        voted = perform_conf_vote([extract_data(p) for p in predictions])

        sentence = ""

        for voted_pos in voted:
            pos = prediction_out.positions.add()
            for character in voted_pos:
                char = pos.chars.add()
                char.char = character.char
                char.probability = character.p

            if len(voted_pos) > 0:
                pos.global_start = voted_pos[0].start
                pos.global_end = voted_pos[0].stop
                sentence += voted_pos[0].char

        prediction_out.sentence = sentence


