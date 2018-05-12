import numpy as np

from calamari_ocr.ocr.voting.voter import Voter
from calamari_ocr.ocr.text_processing.text_synchronizer import synchronize


class SequenceVoter(Voter):
    def __init__(self, optimize=False, n_best=3):
        super().__init__()
        self.optimize = optimize
        self.n_best = n_best

    def _apply_vote(self, predictions, prediction_out):
        texts = [prediction_result.sentence for prediction_result in predictions]
        out = self.process_text(texts)

        # TODO:
        prediction_out.sentence = "".join([c for c, p in out])

    def process_text(self, texts):
        voters = SequenceVoter.text_to_voters(texts)

        if self.optimize:
            SequenceVoter.select_voters(voters)

            if self.n_best > 0:
                actual_voters = voters[:self.n_best]
            else:
                actual_voters = voters

        else:
            actual_voters = voters

        inputs = [voter.text for voter in actual_voters]

        synclist = synchronize(inputs)
        return SequenceVoter.perform_vote(inputs, synclist, actual_voters)

    @staticmethod
    def perform_vote(inputs, synclist, voters):
        num_candidates = 0
        candidates = [{"char": None, "num_votes": 0} for _ in voters]
        output = []

        def place_vote(c, num_candidates, num_votes=1):
            index = 0
            if c is not None:
                while index < num_candidates and (candidates[index]['char'] is None or candidates[index]["char"] != c):
                    index += 1
            else:
                while index < num_candidates and (candidates[index]['char'] is not None):
                    index += 1

            if index < num_candidates:
                candidates[index]['num_votes'] += num_votes
                return num_candidates
            else:
                candidates[i]["char"] = c
                candidates[i]["num_votes"] = num_votes
                return num_candidates + 1

        def winner(num_candidates):
            if num_candidates == 0:
                return True, "", 0

            leader = 0
            total_votes = candidates[0]['num_votes']
            for i in range(1, num_candidates):
                total_votes += candidates[i]['num_votes']
                if candidates[i]['num_votes'] > candidates[leader]['num_votes']:
                    leader = i

            if candidates[leader]["char"] is None:
                return False, "", 0

            return True, candidates[leader]["char"], candidates[leader]['num_votes'] / total_votes

        for sync in synclist:
            r = True
            while r:
                for i, voter in enumerate(voters):
                    if sync.start(i) <= sync.stop(i):
                        num_candidates = place_vote(inputs[i][sync.start(i)], num_candidates)
                        sync.set_start(i, sync.start(i) + 1)
                    else:
                        num_candidates = place_vote(None, num_candidates)

                r, out, p = winner(num_candidates)
                if len(out) > 0:
                    output.append((out, p))
                num_candidates = 0

        return output

    class Voter:
        def __init__(self, text, distance=0, argnum=-1, filename=None):
            self.text = text
            self.distance = distance

        def __str__(self):
            return "Voter: {%f, %s}" % (self.distance, self.text)

        def compute_distance(self, index, sequences):
            for sequence in sequences:
                diff = np.abs(sequence.count[index] - sequence.median)

                self.distance += diff

    class Sequence:
        def __init__(self, key, count, median):
            self.key = key
            self.count = count
            self.median = median

        def __str__(self):
            return "Sequence: {%s, %s, %f}" % (self.key, self.count, self.median)

        def compute_median(self):
            self.median = np.median(self.count)

    @staticmethod
    def add_sequence(sequences, key, reject, index, number_of_voters):
        if key in sequences:
            sequence = sequences[key]
        else:
            sequence = SequenceVoter.Sequence(key, [0] * number_of_voters, 0 if reject else 1)
            sequences[key] = sequence

        sequence.count[index] += 1

    @staticmethod
    def count_sequences(sequences, index, voters):
        voter = voters[index]
        for start in range(len(voter.text)):
            SequenceVoter.add_sequence(sequences, voter.text[start:start + 2], False, index, len(voters))

    @staticmethod
    def select_voters(voters):
        sequences_dict = {}
        for i, voter in enumerate(voters):
            SequenceVoter.count_sequences(sequences_dict, i, voters)

        sequences = sequences_dict.values()

        for sequence in sequences:
            sequence.compute_median()

        for i, voter in enumerate(voters):
            voter.compute_distance(i, sequences)

        voters.sort(key=lambda v: v.distance)

    @staticmethod
    def clean_text(text):
        return text.strip()

    @staticmethod
    def text_to_voters(texts):
        return [SequenceVoter.Voter(SequenceVoter.clean_text(t)) for t in texts]

