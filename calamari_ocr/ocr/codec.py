class Codec:
    @staticmethod
    def from_texts(texts):
        chars = set()

        for text in texts:
            for c in text:
                chars.add(c)

        return Codec(sorted(list(chars)))

    def __init__(self, charset):
        if len(charset) == 0:
            raise Exception("Got empty charset")

        if charset[0] != "":
            self.charset = [""] + charset  # blank is label 0
        else:
            self.charset = charset

        self.code2char = {}
        self.char2code = {}
        for code, char in enumerate(self.charset):
            self.code2char[code] = char
            self.char2code[char] = code

    def __len__(self):
        return len(self.charset)

    def size(self):
        return len(self.charset)

    def encode(self, s):
        return [self.char2code[c] for c in s]

    def decode(self, l):
        return [self.code2char[c] for c in l]

    def extend(self, codec):
        charset = self.code2char.values()
        size = self.size()
        counter = 0
        for c in codec.code2char.values():
            if c not in charset:  # append chars that doesn't appear in the codec
                self.code2char[size] = c
                self.char2code[c] = size
                size += 1
                counter += 1

        return counter

    def shrink(self, codec):
        deleted_positions = []
        positions = []
        for number, char in self.code2char.iteritems():
            if char not in codec.char2code:
                deleted_positions.append(number)
            else:
                positions.append(number)

        self.charset = [self.code2char[c] for c in sorted(positions)]
        self.code2char = {}
        self.char2code = {}
        for code, char in enumerate(self.charset):
            self.code2char[code] = char
            self.char2code[char] = code

        return deleted_positions


def ascii_codec():
    ascii_labels = ["", " ", "~"] + [chr(x) for x in range(33, 126)]
    return Codec(ascii_labels)

