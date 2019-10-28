from __future__ import division
from __future__ import print_function

import codecs
import os.path

import numpy as np

from LanguageModel import LanguageModel


def softmax(mat):
    "calc softmax such that labels per time-step form probability distribution"
    # dim0=t, dim1=c
    maxT, _ = mat.shape
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        maxValue = np.max(y)
        e = np.exp(y - maxValue)
        s = np.sum(e)
        res[t, :] = e / s

    return res


def loadFromCSV(fn):
    "load matrix from csv file. Last entry in row terminated by semicolon."
    mat = np.genfromtxt(fn, delimiter=';')[:, :-1]
    mat = softmax(mat)
    return mat


class Data:
    "holds matrix, ground truth and filenames of a sample"

    def __init__(self):
        self.gt = ''
        self.mat = None
        self.fn = ''


class DataLoader:
    "load data from a given directory"

    def __init__(self, dataset, sampleEach=1):
        self.path = '../data/' + dataset + '/'
        self.chars = codecs.open(self.path + 'chars.txt', 'r', 'utf8').read()
        self.wordChars = codecs.open(self.path + 'wordChars.txt', 'r', 'utf8').read()
        self.lm = LanguageModel(codecs.open(self.path + 'corpus.txt', 'r', 'utf8').read(), self.chars, self.wordChars)
        self.mats = []
        self.gts = []
        self.fns = []

        i = 0
        while True:
            fnMat = self.path + 'mat_' + str(i) + '.csv'
            fnGT = self.path + 'gt_' + str(i) + '.txt'
            i += 1

            # file not found
            if (not os.path.isfile(fnMat)) or (not os.path.isfile(fnGT)):
                break

            # ignore this sample
            if (i - 1) % sampleEach != 0:
                continue

            # put into result
            self.mats.append(fnMat)
            self.gts.append(fnGT)
            self.fns.append(fnMat + '|' + fnGT)

        self.currIdx = 0

    def getNumSamples(self):
        return len(self.mats)

    def __next__(self):
        if self.currIdx >= len(self.mats):
            raise StopIteration()

        data = Data()
        data.mat = loadFromCSV(self.mats[self.currIdx])
        data.gt = codecs.open(self.gts[self.currIdx], 'r', 'utf8').read()
        data.fn = self.fns[self.currIdx]

        self.currIdx += 1
        return data

    # python2 needs next, not __next__
    next = __next__

    def __iter__(self):
        return self
