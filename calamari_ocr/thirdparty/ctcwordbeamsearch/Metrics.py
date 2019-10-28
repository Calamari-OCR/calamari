from __future__ import division
from __future__ import print_function

import re

import editdistance


class Metrics:
    "CER and WER"

    def __init__(self, wordChars=r'\w'):
        self.numWords = 0
        self.numChars = 0

        self.edWords = 0
        self.edChars = 0

        self.pattern = '[' + wordChars + ']+'

    def getWordIDStrings(self, s1, s2):
        # get words in string 1 and string 2
        words1 = re.findall(self.pattern, s1)
        words2 = re.findall(self.pattern, s2)

        # find unique words
        allWords = list(set(words1 + words2))

        # list of word ids for string 1
        idStr1 = []
        for w in words1:
            idStr1.append(allWords.index(w))

        # list of word ids for string 2
        idStr2 = []
        for w in words2:
            idStr2.append(allWords.index(w))

        return (idStr1, idStr2)

    def addSample(self, gt, rec):
        "insert result and ground truth for next sample"
        # chars
        self.edChars += editdistance.eval(gt, rec)
        self.numChars += len(gt)

        # words
        (idStrGt, idStrRec) = self.getWordIDStrings(gt, rec)
        self.edWords += editdistance.eval(idStrGt, idStrRec)
        self.numWords += len(idStrGt)

    def getCER(self):
        "get character error rate"
        return self.edChars / self.numChars

    def getWER(self):
        "get word error rate"
        return self.edWords / self.numWords


if __name__ == '__main__':
    m = Metrics()
    m.addSample('hxllo world', 'hello world')
    m.addSample('yes we cxn', 'yes we can')
    print('CER:', m.getCER())
    print('WER:', m.getWER())
