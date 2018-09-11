import os
from lxml import etree as ET
from .data import *
from .exceptions import *
from tqdm import tqdm
from collections import defaultdict


class XMLReader:
    """
    This class can read Abbyy documents out of a directory
    """

    def __init__(self, imgfiles: [], abbyfiles: [], skip_invalid: bool, remove_invalid: bool):

        """
        Constructs an XMLReader class with the :param directory

        :param directory: Absolute or relative path of the directory there the abbyy documents are located
        """
        assert(len(imgfiles) == len(abbyfiles))
        self.imgfiles = imgfiles
        self.xmlfiles = abbyfiles
        self.skip_invalid = skip_invalid
        self.remove_invalid = remove_invalid

    def read(self) -> Book:

        """
        Start trying to read the data from the directory :var self.directory

        :return: a Data.Book class with all the readed data from :var self.directory
        :exception WrongFileStructureException: Is raised then files are missing in the directory
                    (e.g.: no image file for an xml file which is named equally)
        :exception XMLParseError: Is raised then there are errors in a xml file
        """

        book = Book()
        toremove = []

        # Searching for the xml abbyy files and handling Errors in the data structure
        for i, (imgfile, xmlfile) in tqdm(enumerate(zip(self.imgfiles, self.xmlfiles)),
                                          desc="Loading abby files", total=len(self.imgfiles)):
            if xmlfile:
                if not os.path.exists(xmlfile):
                    if not self.skip_invalid:
                        raise XMLParseError('The abbyy xml file {} does not exist'.format(xmlfile))
                    else:
                        toremove.append(i)
                        continue

            if imgfile:
                if not os.path.exists(imgfile):
                    if not self.skip_invalid:
                        raise XMLParseError('The image file {} does not exist'.format(imgfile))
                    else:
                        toremove.append(i)
                        continue

            try:
                book.pages += list(self.parseXMLfile(imgfile, xmlfile))
            except XMLParseError as e:
                print(e)
                if self.skip_invalid:
                    toremove.append(i)
                    continue
                else:
                    raise e

        for i in reversed(toremove):
            del self.imgfiles[i]
            del self.xmlfiles[i]

        return book

    @staticmethod
    def parseRect(node, required=True) -> Rect:
        try:
            a = XMLReader.requireAttr(node, ['l', 't', 'r', 'b'])

            rect = Rect(int(a['l']), int(a['t']), int(a['r']), int(a['b']))

        except Exception as e:
            if required:
                raise e
            else:
                return None

        return rect

    @staticmethod
    def requireAttr(node, attrs):
        a = {}
        for attr in attrs:
            a[attr] = node.get(attr)
            if a[attr] is None:
                raise XMLParseError('Missing required attribute {} on node {}'.format(attr, node))

        return a

    def parseXMLfile(self, imgfile, xmlfile):
        # Reads the xml file with the xml.etree.ElementTree package
        try:
            tree = ET.parse(xmlfile)
        except ET.ParseError as e:
            raise XMLParseError('The xml file \'' + xmlfile + '\' couldn\'t be read because of a '
                                                              'syntax error in the xml file. ' + e.msg)

        root = tree.getroot()

        if root is None:
            raise XMLParseError('The xml file \'' + xmlfile + '\' is empty.')

        for pagecount, pageNode in enumerate(root):
            a = XMLReader.requireAttr(pageNode, ['width', 'height', 'resolution', 'originalCoords'])
            page = Page(a['width'], a['height'], a['resolution'], a['originalCoords'], imgfile, xmlfile)

            for blockcount, blockNode in enumerate(pageNode):

                # Checks if the blockType is text, ignoring all other types
                type = blockNode.get('blockType')
                if type is not None and type == 'Text':

                    # Reads rectangle data and controls if they are empty
                    name = blockNode.get('blockName')

                    block: Block = Block(type, name, XMLReader.parseRect(blockNode, required=False))

                    for textNode in blockNode:

                        # Again only text nodes will be considered

                        if textNode.tag == '{http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml}text':
                            for parNode in textNode:
                                align = parNode.get('align')
                                startIndent = parNode.get('startIndent')
                                lineSpacing = parNode.get('lineSpacing')

                                par: Par = Par(align, startIndent, lineSpacing)

                                for linecount, lineNode in enumerate(parNode):
                                    baseline = lineNode.get('baseline')

                                    line: Line = Line(baseline, XMLReader.parseRect(lineNode))

                                    lang = None
                                    text = ""
                                    maxCount = 0
                                    for formNode in lineNode:
                                        countChars = 0
                                        if formNode.text is None or formNode.text == "\n" or formNode.text == "":
                                            for charNode in formNode:
                                                text += str(charNode.text)
                                                countChars = countChars + 1
                                            if countChars > maxCount:
                                                maxCount = countChars
                                                lang = formNode.get('lang')


                                        else:
                                            lang = formNode.get('lang')
                                            text = str(formNode.text)

                                    format: Format = Format(lang, text)
                                    line.formats.append(format)
                                    par.lines.append(line)

                                block.pars.append(par)

                    page.blocks.append(block)

            yield page

