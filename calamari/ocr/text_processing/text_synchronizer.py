import numpy as np


class Sync:
    def __init__(self, texts, substr=None, match=None):
        self.texts = texts

        if substr:
            assert(substr.shape[0] == len(self.texts))
            self.substr = substr
        else:
            self.substr = np.zeros((len(texts), 3), dtype=int)

        self.match = match

    def __str__(self):
        return str(self.substr)

    def get_text(self):
        return [self.texts[i][start:start+length] for i, (start, end, length) in enumerate(self.substr)]

    def is_valid(self):
        return np.any(self.substr[:, 2] > 0)

    def lengths(self):
        return self.substr[:, 2]

    def start(self, idx):
        return self.substr[idx, 0]

    def stop(self, idx):
        return self.substr[idx, 1]

    def length(self, idx):
        return self.substr[idx, 2]

    def set_start(self, idx, v):
        self.substr[idx, 0] = v

    def set_stop(self, idx, v):
        self.substr[idx, 1] = v

    def set_length(self, idx, v):
        self.substr[idx, 2] = v

    def set_all(self, idx, v):
        self.substr[idx, :] = v


def synchronize(texts):
    num_text = len(texts)

    def init():
        sync = Sync(texts)
        for i, text in enumerate(texts):
            sync.set_all(i, [0, len(text) - 1, len(text)])

        if sync.is_valid():
            return [sync]

        return []

    def longest_match(maxlen, c1, start1, stop1, c2, start2, stop2):
        mstart1 = 0
        mstart2 = 0
        s1limit = stop1 - maxlen
        s2limit = stop2 - maxlen
        for s1 in range(start1, s1limit + 1):
            for s2 in range(start2, s2limit + 1):
                if c1[s1] == c2[s2]:
                    i1 = s1 + 1
                    i2 = s2 + 1
                    while i1 <= stop1 and i2 <= stop2 and c1[i1] == c2[i2]:
                        i1 += 1
                        i2 += 1

                    increase = i1 - s1 - maxlen
                    if increase > 0:
                        s1limit -= increase
                        s2limit -= increase
                        maxlen += increase
                        mstart1 = s1
                        mstart2 = s2

        return maxlen, mstart1, mstart2

    def save_match(synclist, num_text, sync, start, length, match):
        left, right = Sync(texts), Sync(texts)
        for i in range(num_text):
            stop = start[i] + length - 1
            left.set_all(i, [sync.start(i), start[i] - 1, start[i] - sync.start(i)])
            right.set_all(i, [stop + 1, sync.stop(i), sync.stop(i) - stop])
            sync.set_all(i, [start[i], stop, length])

        sync.match = match
        if left.is_valid():
            synclist.insert(synclist.index(sync), left)

        if right.is_valid():
            synclist.insert(synclist.index(sync) + 1, right)

    def recursive_sync(synclist, texts, start_index):
        sync = synclist[start_index]
        if np.any(sync.lengths() == 0):
            return

        start = np.zeros(len(texts), dtype=int)
        start[0] = sync.start(0)
        length = sync.length(0)
        for i, text in enumerate(texts[1:], 1):
            length, new_start, start[i] = longest_match(0, texts[0], start[0], start[0] + length - 1,
                                                        text, sync.start(i), sync.stop(i))

            if length == 0:
                return

            change = new_start - start[0]
            if change > 0:
                for j in range(i):
                    start[j] += change

        save_match(synclist, len(texts), sync, start, length, True)

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
