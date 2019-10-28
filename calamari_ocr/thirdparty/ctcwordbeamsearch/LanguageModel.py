from __future__ import division
from __future__ import print_function

import re

from .PrefixTree import PrefixTree


class LanguageModel:
    "unigram/bigram LM, add-k smoothing"

    def __init__(self, corpus, chars, wordChars):
        "read text from filename, specify chars which are contained in dataset, specify chars which form words"
        # read from file
        self.wordCharPattern = '[' + re.escape(wordChars) + ']'
        self.wordPattern = self.wordCharPattern + '+'
        words = re.findall(self.wordPattern, corpus)
        uniqueWords = list(set(words))  # make unique
        self.numWords = len(words)
        self.numUniqueWords = len(uniqueWords)
        self.smoothing = True
        self.addK = 1.0 if self.smoothing else 0.0

        # create unigrams
        self.unigrams = {}
        for w in words:
            w = w.lower()
            if w not in self.unigrams:
                self.unigrams[w] = 0
            self.unigrams[w] += 1 / self.numWords

        # create unnormalized bigrams
        bigrams = {}
        for i in range(len(words) - 1):
            w1 = words[i].lower()
            w2 = words[i + 1].lower()
            if w1 not in bigrams:
                bigrams[w1] = {}
            if w2 not in bigrams[w1]:
                bigrams[w1][w2] = self.addK  # add-K
            bigrams[w1][w2] += 1

        # normalize bigrams
        for w1 in bigrams.keys():
            # sum up
            probSum = self.numUniqueWords * self.addK  # add-K smoothing
            for w2 in bigrams[w1].keys():
                probSum += bigrams[w1][w2]
            # and divide
            for w2 in bigrams[w1].keys():
                bigrams[w1][w2] /= probSum
        self.bigrams = bigrams

        # create prefix tree
        self.tree = PrefixTree()  # create empty tree
        self.tree.addWords(uniqueWords)  # add all unique words to tree

        # list of all chars, word chars and nonword chars
        self.allChars = chars
        self.wordChars = wordChars
        self.nonWordChars = str().join(
            set(chars) - set(re.findall(self.wordCharPattern, chars)))  # else calculate those chars

    def getNextWords(self, text):
        "text must be prefix of a word"
        return self.tree.getNextWords(text)

    def getNextChars(self, text):
        "text must be prefix of a word"
        nextChars = str().join(self.tree.getNextChars(text))

        # if in between two words or if word ends, add non-word chars
        if (text == '') or (self.isWord(text)):
            nextChars += self.getNonWordChars()

        return nextChars

    def getWordChars(self):
        return self.wordChars

    def getNonWordChars(self):
        return self.nonWordChars

    def getAllChars(self):
        return self.allChars

    def isWord(self, text):
        return self.tree.isWord(text)

    def getUnigramProb(self, w):
        "prob of seeing word w."
        w = w.lower()
        val = self.unigrams.get(w)
        if val != None:
            return val
        return 0

    def getBigramProb(self, w1, w2):
        "prob of seeing words w1 w2 next to each other."
        w1 = w1.lower()
        w2 = w2.lower()
        val1 = self.bigrams.get(w1)
        if val1 != None:
            val2 = val1.get(w2)
            if val2 != None:
                return val2
            return self.addK / (self.getUnigramProb(w1) * self.numUniqueWords + self.numUniqueWords)
        return 0


if __name__ == '__main__':
    lm = LanguageModel('12 1 13 12 15 234 2526', ' ,.:0123456789', '0123456789')
    prefix = '1'
    print('getNextChars:', lm.getNextChars(prefix))
    print('getNonWordChars:', lm.getNonWordChars())
    print('getNextWords:', lm.getNextWords(prefix))
    print('isWord:', lm.isWord(prefix))
    print('getBigramProb:', lm.getBigramProb('12', '15'))
