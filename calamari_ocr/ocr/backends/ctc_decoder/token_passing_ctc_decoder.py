# Based on: https://github.com/githubharald/CTCDecoder
# Using the algorithm of Graves

import math
import numpy as np
from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoder


class TokenPassingCTCDecoder(CTCDecoder):
    def __init__(self, params, codec):
        super().__init__(params, codec)

    def decode(self, probabilities):
        if self.params.blank_index == 0:
            probabilities = np.roll(probabilities, -1, axis=1)
        r = ctcTokenPassing(probabilities, self.codec.charset, self.params.dictionary)
        return self._prediction_from_string(probabilities, r)


def extendByBlanks(seq, b):
    """extends a label seq. by adding blanks at the beginning, end and in between each label"""
    res = [b]
    for s in seq:
        res.append(s)
        res.append(b)
    return res


def wordToLabelSeq(w, classes):
    """map a word to a sequence of labels (indices)"""
    try:
        res = [classes.index(c) for c in w]
        return res
    except ValueError:
        return None


class Token:
    """token for token passing algorithm. Each token contains a score and a history of visited words."""
    def __init__(self, score=float('-inf'), history=None):
        self.score = score
        self.history = history if history else []

    def __str__(self):
        res = 'class Token: '+str(self.score)+'; '
        for w in self.history:
            res += w+'; '
        return res


class TokenList:
    """this class simplifies getting/setting tokens"""
    def __init__(self):
        self.tokens = {}

    def set(self, w, s, t, tok):
        self.tokens[(w, s, t)] = tok

    def get(self, w, s, t):
        return self.tokens[(w, s, t)]

    def dump(self, s, t):
        for (k, v) in self.tokens.items():
            if k[1] == s and k[2] == t:
                print(k, v)


def outputIndices(toks, words, s, t):
    """argmax_w tok(w,s,t)"""
    res = []
    for (wIdx, _) in enumerate(words):
        res.append(toks.get(wIdx, s, t))

    idx = [i[0] for i in sorted(enumerate(res), key=lambda x: x[1].score)]
    return idx


def log(val):
    """return -inf for log(0) instead of throwing error like python implementation does it"""
    if val > 0:
        return math.log(val)
    return float('-inf')


def ctcTokenPassing(mat, classes, charWords):
    """implements CTC Token Passing Algorithm as shown by Graves (Dissertation, p67-69)"""
    blankIdx = len(classes)
    maxT, _ = mat.shape

    # special s index for beginning and end of word
    beg = 0
    end = -1

    # map characters to labels for each word
    words = [wordToLabelSeq(w, classes) for w in charWords]
    words = [w for w in words if w]

    # w' in paper: word with blanks in front, back and between labels: for -> _f_o_r_
    primeWords = [extendByBlanks(w, blankIdx) for w in words]

    # data structure holding all tokens
    toks = TokenList()

    # Initialisation: 1-9
    for (wIdx, w) in enumerate(words):
        w = words[wIdx]
        wPrime = primeWords[wIdx]

        #set all toks(w,s,t) to init state
        for s in range(len(wPrime)):
            for t in range(maxT):
                toks.set(wIdx, s+1, t+1, Token())
                toks.set(wIdx, beg, t, Token())
                toks.set(wIdx, end, t, Token())

        toks.set(wIdx, 1, 1, Token(log(mat[1 - 1, blankIdx]), [wIdx]))
        cIdx = w[1 - 1]
        toks.set(wIdx, 2, 1, Token(log(mat[1 - 1, cIdx]), [wIdx]))

        if len(w) == 1:
            toks.set(wIdx, end, 1, toks.get(wIdx, 2, 1))

    # Algorithm: 11-24
    t = 2
    while t <= maxT:

        sortedWordIdx = outputIndices(toks, words, end, t-1)

        for wIdx in sortedWordIdx:
            wPrime = primeWords[wIdx]
            w = words[wIdx]

            # 15-17
            # if bigrams should be used, these lines have to be adapted
            bestOutputTok = toks.get(sortedWordIdx[-1], end, t-1)
            toks.set(wIdx, beg, t, Token(bestOutputTok.score, bestOutputTok.history+[wIdx]))

            # 18-24
            s = 1
            while s <= len(wPrime):
                P = [toks.get(wIdx, s, t-1), toks.get(wIdx, s - 1, t - 1)]
                if wPrime[s-1] != blankIdx and s > 2 and wPrime[s - 2 - 1] != wPrime[s - 1]:
                    tok = toks.get(wIdx, s - 2, t - 1)
                    P.append(Token(tok.score, tok.history))

                maxTok = sorted(P, key=lambda x: x.score)[-1]
                cIdx = wPrime[s-1]

                score = maxTok.score+log(mat[t-1, cIdx])
                history = maxTok.history

                toks.set(wIdx, s, t, Token(score, history))
                s += 1

            maxTok = sorted([toks.get(wIdx, len(wPrime), t), toks.get(wIdx, len(wPrime)-1, t)], key=lambda x: x.score, reverse=True)[0]
            toks.set(wIdx, end, t, maxTok)

        t += 1

    # Termination: 26-28
    bestWIdx = outputIndices(toks, words, end, maxT)[-1]
    return str(' ').join([charWords[i] for i in toks.get(bestWIdx, end, maxT).history])


if __name__ == '__main__':
    """test decoder"""
    classes = 'ab'
    mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
    print('Test token passing')
    expected = 'a'
    actual = ctcTokenPassing(mat, classes, ['a', 'b', 'ab', 'ba'])
    print('Expected: "'+expected+'"')
    print('Actual: "'+actual+'"')
    print('OK' if expected == actual else 'ERROR')
