import numpy as np
from difflib import SequenceMatcher

class Sync:
    def __init__(self, texts, substr=None):
        self.texts = texts

        if substr is not None:
            assert substr.shape[0] == len(self.texts)
            self.substr = substr
        else:
            self.substr = np.zeros((len(texts), 2), dtype=int)

    def __str__(self):
        return str(self.substr)

    def get_text(self):
        return [self.texts[i][start : start + length] for i, (start, length) in enumerate(self.substr)]

    def is_valid(self):
        return np.any(self.substr[:, 1] > 0)

    def lengths(self):
        return self.substr[:, 1]

    def start(self, idx):
        return self.substr[idx, 0]

    def stop(self, idx):
        return self.start(idx) + self.length(idx) - 1

    def length(self, idx):
        return self.substr[idx, 1]

    def set_start(self, idx, v):
        self.substr[idx, 0] = v

    def set_length(self, idx, v):
        self.substr[idx, 1] = v

    def set_all(self, idx, v):
        self.substr[idx, :] = v


def synchronize(texts):
    n_texts = len(texts)
    
    def init():
        sync = Sync(texts, substr=np.asarray([(0, len(t)) for t in texts], dtype=int))
        if sync.is_valid():
            return [sync]
        return []

    def longest_match(c1, start1, l1, c2, start2, l2):
        m = SequenceMatcher(None, c1[start1:start1+l1], c2[start2:start2+l2]).find_longest_match()
        return m.size, start1 + m.a, start2 + m.b
    
    def save_match(synclist, sync, start, length):
        left, right = Sync(texts), Sync(texts)
        for i in range(n_texts):
            left.set_all(i,  [sync.start(i), start[i] - sync.start(i)])
            stop = start[i] + length - 1
            right.set_all(i, [stop + 1,      sync.stop(i) - stop])
            sync.set_all(i,  [start[i],      length])

        if left.is_valid():
            synclist.insert(synclist.index(sync), left)

        if right.is_valid():
            synclist.insert(synclist.index(sync) + 1, right)

    def recursive_sync(synclist, texts, start_index):
        sync = synclist[start_index]
        if np.any(sync.lengths() == 0):
            return

        start = np.zeros(n_texts, dtype=int)
        start[0] = sync.start(0)
        length = sync.length(0)
        for i, text in enumerate(texts[1:], 1):
            length, new_start, start[i] = longest_match(
                texts[0],
                start[0],
                length,
                text,
                sync.start(i),
                sync.length(i),
            )

            if length == 0:
                return

            change = new_start - start[0]
            if change > 0:
                for j in range(i):
                    start[j] += change

        save_match(synclist, sync, start, length)

        start_index = synclist.index(sync)
        if start_index - 1 >= 0:
            recursive_sync(synclist, texts, start_index - 1)

        start_index = synclist.index(sync)
        if start_index + 1 < len(synclist):
            recursive_sync(synclist, texts, start_index + 1)

        return

    synclist = init()

    if len(synclist) > 0:
        recursive_sync(synclist, texts, 0)

    return synclist


if __name__ == "__main__":
    synclist = synchronize(["AbcdEfG", "cdEFG"])
    print([s.get_text() for s in synclist])
