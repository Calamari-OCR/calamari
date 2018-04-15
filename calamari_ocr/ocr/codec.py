class Codec:
    @staticmethod
    def from_texts(texts, whitelist=[]):
        chars = set(whitelist)

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
        size = self.size()
        added = []
        for c in codec.code2char.values():
            if c not in self.charset:  # append chars that don't appear in the codec
                self.code2char[size] = c
                self.char2code[c] = size
                self.charset.append(c)
                added.append(size)
                size += 1

        return added

    def shrink(self, codec):
        deleted_positions = []
        positions = []
        for number, char in self.code2char.items():
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

    def align(self, codec):
        deleted_positions = self.shrink(codec)
        added_positions = self.extend(codec)
        return deleted_positions, added_positions


def ascii_codec():
    ascii_labels = ["", " ", "~"] + [chr(x) for x in range(33, 126)]
    return Codec(ascii_labels)

