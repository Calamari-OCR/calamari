import numpy as np

from calamari_ocr.ocr.voting.voter import Voter
from calamari_ocr.ocr.text_processing.text_synchronizer import synchronize


def add_llocs(s, new):
    for char in new.keys():
        if char in s:
            s[char] += new[char]
        else:
            s[char] = new[char]


def get_most_likely_char(s):
    return max(s, key=lambda key: s[key])


def find_voters_with_most_frequent_legth(sync, voters):
    lengths = {}

    for i, voter in enumerate(voters):
        length = sync.length(i)

        if length in lengths:
            lengths[length] += 1
        else:
            lengths[length] = 1

    most_freq_length = -1
    occurrences = 0

    for length in lengths.keys():
        if lengths[length] < occurrences:
            continue
        elif lengths[length] == occurrences:
            if length < most_freq_length:
                most_freq_length = length
                occurrences = lengths[length]
        else:
            most_freq_length = length
            occurrences = lengths[length]

    actual_voters = []

    for i, voter in enumerate(voters):
        if sync.length(i) == most_freq_length:
            actual_voters.append(i)

    return actual_voters, most_freq_length


def perform_conf_vote(voters):
    results = [voter[1] for voter in voters]

    synclist = synchronize(results)

    final_result = ""

    for sync in synclist:
        actual_voters, most_freq_length = find_voters_with_most_frequent_legth(sync, voters)

        # check if all voters agree
        if len(actual_voters) == len(voters):
            l = []

            for i in range(len(voters)):
                l.append(voters[i][1][sync.start(i):sync.stop(i) + 1])

            s = set(l)

            if len(s) == 1:
                final_result += s.pop()
                continue

        if len(actual_voters) == 1:
            final_result += voters[actual_voters[0]][1][sync.start(actual_voters[0]):sync.stop(actual_voters[0]) + 1]
        else:
            for i in range(most_freq_length):
                s = {}

                for idx, voter in enumerate(actual_voters):
                    new_llocs = voters[voter][2][sync.start(voter) + i:sync.start(voter) + i + 1]

                    if len(new_llocs) == 0:
                        continue

                    add_llocs(s, new_llocs[0])

                if len(s) > 0:
                    final_result += get_most_likely_char(s)

    return final_result


class ConfidenceVoter(Voter):
    def __init__(self, blank_index=0, fuzzy_ctc=False, blank_threshold=0.7):
        super().__init__()
        self.fuzzy_ctc = fuzzy_ctc
        self.blank_threshold = blank_threshold
        self.blank_index = blank_index
        self.min_candidate_probability = 1e-4

    def _apply_vote(self, prediction_result_tuple):
        if self.fuzzy_ctc:
            return self._apply_fuzzy_vote(prediction_result_tuple)
        else:
            return self._apply_default_vote(prediction_result_tuple)

    def _apply_fuzzy_vote(self, prediction_result_tuple):
        voter_sequences = []
        voter_alternatives = []
        for prediction_result in prediction_result_tuple:
            pred, codec = prediction_result.logits, prediction_result.codec
            sequence = []
            alternatives = []

            # This is an implementation of the ctc greedy decoder, but we track the probabilities
            # of all alternatives here
            current_best_index = -1
            current_best_p = -1

            current_best_char_p = np.zeros(pred.shape[1], dtype=np.float32)
            for t in range(len(pred)):
                p_blank = pred[t, self.blank_index]
                if p_blank > self.blank_threshold:
                    if current_best_index >= 0:
                        if current_best_index != self.blank_index:
                            sequence.append(current_best_index)
                            alternatives.append(current_best_char_p.copy())

                        current_best_char_p.fill(0)
                        current_best_index = -1
                        current_best_p = -1
                else:
                    current_best_char_p = np.maximum(current_best_char_p, pred[t])
                    # alternatively, add all p's

                    idx = np.argmax(pred[t])
                    p = pred[t, idx]
                    if p > current_best_p:
                        current_best_index = idx
                        current_best_p = p

            if current_best_index >= 0:
                if current_best_index != self.blank_index:
                    sequence.append(current_best_index)
                    alternatives.append(current_best_char_p.copy())

            # end of ctc greedy fuzzy decoder

            voter_sequences.append("".join([codec.code2char[s] for s in sequence]))
            alternative_dicts = []
            for alternative in alternatives:
                alternatives_dict = {}
                for i, p in enumerate(alternative):
                    if p < self.min_candidate_probability:
                        continue

                    if i == self.blank_index:
                        continue

                    alternatives_dict[codec.code2char[i]] = p

                alternative_dicts.append(alternatives_dict)

            voter_alternatives.append(alternative_dicts)

        for seq, alt in zip(voter_sequences, voter_alternatives):
            assert (len(seq) == len(alt))

        return perform_conf_vote(list(zip(range(len(voter_sequences)), voter_sequences, voter_alternatives)))

    def _apply_default_vote(self, prediction_result_tuple):
        voter_sequences = []
        voter_alternatives = []
        for prediction_result in prediction_result_tuple:
            pred, codec = prediction_result.logits, prediction_result.codec
            sequence = []
            alternatives = []

            best_chars = np.argmax(pred, axis=1)

            # This is the default ctc greedy decoder, but we store the probabilities as-well
            current_best_char_p = np.zeros(pred.shape[1], dtype=np.float32)
            last_char = self.blank_index
            for t in range(len(pred)):
                best_char = best_chars[t]
                if best_char != last_char:
                    if last_char != self.blank_index:
                        sequence.append(last_char)
                        alternatives.append(current_best_char_p.copy())
                        current_best_char_p.fill(0)

                    last_char = best_char

                if best_char != self.blank_index:
                    current_best_char_p = np.maximum(current_best_char_p, pred[t])

            if last_char != self.blank_index:
                sequence.append(last_char)
                alternatives.append(current_best_char_p.copy())

            # End of ctc greedy decoder

            voter_sequences.append("".join([codec.code2char[s] for s in sequence]))
            alternative_dicts = []
            for alternative in alternatives:
                alternatives_dict = {}
                for i, p in enumerate(alternative):
                    if p < self.min_candidate_probability:
                        continue

                    if i == self.blank_index:
                        continue

                    alternatives_dict[codec.code2char[i]] = p

                alternative_dicts.append(alternatives_dict)

            voter_alternatives.append(alternative_dicts)

        for seq, alt in zip(voter_sequences, voter_alternatives):
            assert (len(seq) == len(alt))

        return perform_conf_vote(list(zip(range(len(voter_sequences)), voter_sequences, voter_alternatives)))

