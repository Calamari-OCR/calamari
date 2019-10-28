from __future__ import division
from __future__ import print_function


class Node:
    "class representing nodes in a prefix tree"

    def __init__(self):
        self.children = {}  # all child elements beginning with current prefix
        self.isWord = False  # does this prefix represent a word

    def __str__(self):
        s = ''
        for k in self.children.keys():
            s += k
        return 'isWord: ' + str(self.isWord) + '; children: ' + s


class PrefixTree:
    "prefix tree"

    def __init__(self):
        self.root = Node()

    def addWord(self, text):
        "add word to prefix tree"
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
            isLast = (i + 1 == len(text))
            if isLast:
                node.isWord = True

    def addWords(self, words):
        for w in words:
            self.addWord(w)

    def getNode(self, text):
        "get node representing given text"
        node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def isWord(self, text):
        node = self.getNode(text)
        if node:
            return node.isWord
        return False

    def getNextChars(self, text):
        "get all characters which may directly follow given text"
        chars = []
        node = self.getNode(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars

    def getNextWords(self, text):
        "get all words of which given text is a prefix (including the text itself, it is a word)"
        words = []
        node = self.getNode(text)
        if node:
            nodes = [node]
            prefixes = [text]
            while len(nodes) > 0:
                # put all children into list
                for k, v in nodes[0].children.items():
                    nodes.append(v)
                    prefixes.append(prefixes[0] + k)

                # is current node a word
                if nodes[0].isWord:
                    words.append(prefixes[0])

                # remove current node
                del nodes[0]
                del prefixes[0]

        return words

    def dump(self):
        nodes = [self.root]
        while len(nodes) > 0:
            # put all children into list
            for v in nodes[0].children.values():
                nodes.append(v)

            # dump current node
            print(nodes[0])

            # remove from list
            del nodes[0]


if __name__ == '__main__':
    t = PrefixTree()  # create tree
    t.addWords(['this', 'that'])  # add words
    print(t.getNextChars('th'))  # chars following 'th'
    print(t.getNextWords('tha'))  # all words of which 'th' is prefix
    t.dump()  # dump all nodes
