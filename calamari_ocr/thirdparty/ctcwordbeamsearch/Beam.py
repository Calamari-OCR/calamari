from __future__ import division
from __future__ import print_function

import copy


class Optical:
    "optical score of beam"

    def __init__(self, prBlank=0, prNonBlank=0):
        self.prBlank = prBlank  # prob of ending with a blank
        self.prNonBlank = prNonBlank  # prob of ending with a non-blank


class Textual:
    "textual score of beam"

    def __init__(self, text=''):
        self.text = text
        self.wordHist = []  # history of words so far
        self.wordDev = ''  # developing word
        self.prUnnormalized = 1.0
        self.prTotal = 1.0


class Beam:
    "beam with text, optical and textual score"

    def __init__(self, lm, useNGrams):
        "creates genesis beam"
        self.optical = Optical(1.0, 0.0)
        self.textual = Textual('')
        self.lm = lm
        self.useNGrams = useNGrams

    def mergeBeam(self, beam):
        "merge probabilities of two beams with same text"

        if self.getText() != beam.getText():
            raise Exception('mergeBeam: texts differ')

        self.optical.prNonBlank += beam.getPrNonBlank()
        self.optical.prBlank += beam.getPrBlank()

    def getText(self):
        return self.textual.text

    def getPrBlank(self):
        return self.optical.prBlank

    def getPrNonBlank(self):
        return self.optical.prNonBlank

    def getPrTotal(self):
        return self.getPrBlank() + self.getPrNonBlank()

    def getPrTextual(self):
        return self.textual.prTotal

    def getNextChars(self):
        return self.lm.getNextChars(self.textual.wordDev)

    def createChildBeam(self, newChar, prBlank, prNonBlank):
        "extend beam by new character and set optical score"
        beam = Beam(self.lm, self.useNGrams)

        # copy textual information
        beam.textual = copy.deepcopy(self.textual)
        beam.textual.text += newChar

        # do textual calculations only if beam gets extended
        if newChar != '':
            if self.useNGrams:  # use unigrams and bigrams

                # if new char occurs inside a word
                if newChar in beam.lm.getWordChars():
                    beam.textual.wordDev += newChar
                    nextWords = beam.lm.getNextWords(beam.textual.wordDev)

                    # no complete word in text, then use unigram of all possible next words
                    numWords = len(beam.textual.wordHist)
                    prSum = 0
                    if numWords == 0:
                        for w in nextWords:
                            prSum += beam.lm.getUnigramProb(w)
                    # take last complete word and sum up bigrams of all possible next words
                    else:
                        lastWord = beam.textual.wordHist[-1]
                        for w in nextWords:
                            prSum += beam.lm.getBigramProb(lastWord, w)
                    beam.textual.prTotal = beam.textual.prUnnormalized * prSum
                    beam.textual.prTotal = beam.textual.prTotal ** (
                            1 / (numWords + 1)) if numWords >= 1 else beam.textual.prTotal

                # if new char does not occur inside a word
                else:
                    # if current word is not empty, add it to history
                    if beam.textual.wordDev != '':
                        beam.textual.wordHist.append(beam.textual.wordDev)
                        beam.textual.wordDev = ''

                        # score with unigram (first word) or bigram (all other words) probability
                        numWords = len(beam.textual.wordHist)
                        if numWords == 1:
                            beam.textual.prUnnormalized *= beam.lm.getUnigramProb(beam.textual.wordHist[-1])
                            beam.textual.prTotal = beam.textual.prUnnormalized
                        elif numWords >= 2:
                            beam.textual.prUnnormalized *= beam.lm.getBigramProb(beam.textual.wordHist[-2],
                                                                                 beam.textual.wordHist[-1])
                            beam.textual.prTotal = beam.textual.prUnnormalized ** (1 / numWords)

            else:  # don't use unigrams and bigrams, just keep wordDev up to date
                if newChar in beam.lm.getWordChars():
                    beam.textual.wordDev += newChar
                else:
                    beam.textual.wordDev = ''

        # set optical information
        beam.optical.prBlank = prBlank
        beam.optical.prNonBlank = prNonBlank
        return beam

    def __str__(self):
        return '"' + self.getText() + '"' + ';' + str(self.getPrTotal()) + ';' + str(self.getPrTextual()) + ';' + str(
            self.textual.prUnnormalized)


class BeamList:
    "list of beams at specific time-step"

    def __init__(self):
        self.beams = {}

    def addBeam(self, beam):
        "add or merge new beam into list"
        # add if text not yet known
        if beam.getText() not in self.beams:
            self.beams[beam.getText()] = beam
        # otherwise merge with existing beam
        else:
            self.beams[beam.getText()].mergeBeam(beam)

    def getBestBeams(self, num):
        "return best beams, specify the max. number of beams to be returned (beam width)"
        u = [v for (_, v) in self.beams.items()]
        lmWeight = 1
        return sorted(u, reverse=True, key=lambda x: x.getPrTotal() * (x.getPrTextual() ** lmWeight))[:num]

    def deletePartialBeams(self, lm):
        "delete beams for which last word is not finished"
        for (k, v) in self.beams.items():
            lastWord = v.textual.wordDev
            if (lastWord != '') and (not lm.isWord(lastWord)):
                del self.beams[k]

    def completeBeams(self, lm):
        "complete beams such that last word is complete word"
        for (_, v) in self.beams.items():
            lastPrefix = v.textual.wordDev
            if lastPrefix == '' or lm.isWord(lastPrefix):
                continue

            # get word candidates for this prefix
            words = lm.getNextWords(lastPrefix)
            # if there is just one candidate, then the last prefix can be extended to
            if len(words) == 1:
                word = words[0]
                v.textual.text += word[len(lastPrefix) - len(word):]

    def dump(self):
        for k in self.beams.keys():
            print(unicode(self.beams[k]).encode('ascii', 'replace'))  # map to ascii if possible (for py2 and windows)
