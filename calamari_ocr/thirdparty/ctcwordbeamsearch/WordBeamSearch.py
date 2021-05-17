import numpy as np

from .Beam import Beam, BeamList
from .LanguageModel import LanguageModel


def wordBeamSearch(mat, beamWidth, lm, useNGrams, allowWordToWordTransition=False):
    "decode matrix, use given beam width and language model"
    chars = lm.getAllChars()
    blankIdx = len(chars)
    maxT, _ = mat.shape  # shape of RNN output: TxC

    genesisBeam = Beam(lm, useNGrams)  # empty string
    last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.addBeam(genesisBeam)  # start with genesis beam

    startChars = set(lm.tree.getNextChars(""))

    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  # get best beams
        for beam in bestBeams:
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.getText() != "":
                # char at time-step t must also occur at t-1
                labelIdx = chars.index(beam.getText()[-1])
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]

            # calc probability that beam ends with blank
            prBlank = beam.getPrTotal() * mat[t, blankIdx]

            # save result
            curr.addBeam(beam.createChildBeam("", prBlank, prNonBlank))

            def getNonBlank(c):
                labelIdx = chars.index(c)
                if beam.getText() != "" and beam.getText()[-1] == c:
                    return mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                else:
                    return mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours

            # extend current beam with characters according to language model
            for c in beam.getNextChars():
                curr.addBeam(beam.createChildBeam(c, 0, getNonBlank(c)))

            # allow words to directly follow words without a space (or any other sign)
            if lm.isWord(beam.textual.wordDev) and allowWordToWordTransition:
                for c in startChars:
                    b = beam.createChildBeam(c, 0, getNonBlank(c))
                    b.textual.wordDev = c
                    curr.addBeam(b)

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.completeBeams(lm)
    bestBeams = last.getBestBeams(1)  # sort by probability
    return bestBeams[0].getText()


if __name__ == "__main__":
    testLM = LanguageModel("a b aa ab ba bb", "ab ", "ab")
    testMat = np.array([[0.3, 0.1, 0, 0.6], [0.3, 0.1, 0, 0.6]])
    testBW = 25
    res = wordBeamSearch(testMat, testBW, testLM, False)
    print('Result: "' + res + '"')
