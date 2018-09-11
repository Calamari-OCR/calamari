class Rect:

    """
    Class defines the rectangle of an element in the Abbyy document
    """

    def __init__(self, l: int, t: int, r: int, b: int):

        """
        Constructs a Rect class
        :param l: length
        :param t: top
        :param r: right
        :param b: bottom
        """

        self.left = l
        self.top = t
        self.right = r
        self.bottom = b
        self.height = self.bottom - self.top
        self.width = self.right - self.left

    def __str__(self):
        return 'Rect:[l=\'' + self.left.__str__() + '\', t=\'' + self.top.__str__() + '\', r=\'' + \
               self.right.__str__() + '\', b=\'' + self.bottom.__str__() + '\']'


class Book:

    """
    Main class; contains all subelements: book -> page -> block -> par -> line -> format
    """

    def __init__(self):
        self.pages = []

    def __str__(self):
        """
        Writes all information of the book element and all sub elments into the python console.
        :return: None
        """

        s = ""

        for page in self.pages:
            s += page
            for block in page.blocks:
                s += ('     '+block.__str__())
                for par in block.pars:
                    s += ('         '+par.__str__())
                    for line in par.lines:
                        s += '              '+line.__str__()
                        for format in line.formats:
                            s += '                  ' + format.__str__()

    def getBlocks(self)->[]:

        """
        :return: All the blocks of this book
        """

        blocks = []

        for page in self.pages:
            for block in page.blocks:
                blocks.append(block)

        return blocks

    def getPars(self)->[]:

        """
        :return: All the paragraphs of this book
        """

        pars = []

        for page in self.pages:
            for block in page.blocks:
                for par in block.pars:
                    pars.append(par)

        return pars

    def getLines(self)->[]:

        """
        :return: All the lines of this book
        """

        lines = []

        for page in self.pages:
            for block in page.blocks:
                for par in block.pars:
                    for line in par.lines:
                        lines.append(line)

        return lines

    def getFormats(self)->[]:

        """
        :return: All the chars of this book
        """

        formats = []

        for page in self.pages:
            for block in page.blocks:
                for par in block.pars:
                    for line in par.lines:
                        for format in line.formats:
                            formats.append(format)

        return formats


class Page:

    """
    Subelement of the book class; contains a list with the subelement block
    """

    def __init__(self, width: int, height: int, resolution: int, originalCoords: int, imgFile: str, xmlFile: str):

        """
        Construct a page class with an empty block list
        :param width: The width of the page (in pixel)
        :param height: The height of the page (in pixel)
        :param resolution: The resolution of the page (in dpi ???)
        :param originalCoords: ???
        :param imgFile: The name of the image file
        :param xmlFile: The name of the xml file
        """

        self.width = width
        self.height = height
        self.resolution = resolution
        self.originalCoords = originalCoords
        self.imgFile = imgFile
        self.xmlFile = xmlFile
        self.blocks = []

    def __str__(self):
        return 'Page:[ImageFile=\''+self.imgFile +\
                '\', XMLFile=\''+self.xmlFile +\
                '\', width=\''+self.width.__str__() +\
                '\', height=\''+self.height.__str__() +\
                '\', resolution=\''+self.resolution.__str__() +\
                '\', originalCoords=\''+self.originalCoords.__str__()+'\']'

    def getPars(self) -> []:

        """
        :return: All the pars of this page
        """

        pars = []

        for block in self.blocks:
            for par in block.pars:
                pars.append(par)

        return pars


    def getLines(self) -> []:

        """
        :return: All the lines of this page
        """

        lines = []

        for block in self.blocks:
            for par in block.pars:
                for line in par.lines:
                    lines.append(line)

        return lines

    def getFormats(self) -> []:

        """
        :return: All the Format Tags of this page
        """

        formats = []

        for block in self.blocks:
            for par in block.pars:
                for line in par.lines:
                    for format in line.formats:
                        formats.append(format)

        return formats


class Block:

    """
    Subelement of the page class; contains a list with the subelement par
    """

    def __init__(self, blockType: str, blockName: str, rect: Rect):

        """
        Construct a block class with an empty line list
        :param blockType: The type of a block (further information in the abbyy doc)
        :param rect: The rectangle of this element
        """

        self.blockType = blockType
        self.blockName = blockName
        self.rect = rect
        self.pars = []

    def __str__(self):
        return 'Block:[BlockType={}, rect={}]'.format(self.blockType, self.rect)

    def getLines(self) -> []:

        """
        :return: All the lines of this block
        """

        lines = []

        for par in self.pars:
            for line in par.lines:
                lines.append(line)

        return lines

    def getFormats(self) -> []:

        """
        :return: All the Format Tags of this block
        """

        formats = []

        for par in self.pars:
            for line in par.lines:
                for format in line.formats:
                    formats.append(format)

        return formats


class Par:
    """"
    Subelement of the block class; contains a list with the subelement line
    """

    def __init__(self, align: str, startIndent: int, lineSpacing: int):

        """
        Construct a Paragraph Class with an empty line list
        :param align:
        :param startIndent:
        :param lineSpacing:
        """

        self.align = align
        self.startIndent = startIndent
        self.lineSpacing = lineSpacing
        self.lines = []

    def __str__(self):
        return 'Paragraph:[Align=\''+self.align.__str__()+\
                '\', startIndent=\''+self.startIndent.__str__()+\
                '\', lineSpacing=\''+self.lineSpacing.__str__()+'\']'

    def getFormats(self) -> []:

        """
        :return: All the Format Tags of the Paragraph
        """

        formats = []

        for line in self.lines:
            for format in line.formats:
                formats.append(format)

        return formats


class Line:

    """"
    Subelement of the par class; contains a list with the subelement format
    """

    def __init__(self, baseline: int, rect: Rect):

        """
        Construct a line class with an empty char list
        :param baseline: ???
        :param rect: The rectangle of this element
        """

        self.baseline = baseline
        self.rect = rect
        self.formats = []

    def __str__(self):
        return 'Line:[baseline=\''+self.baseline.__str__() +\
                '\', '+self.rect.__str__()+']'


class Format:

    def __init__(self, lang: str, text: str):

        self.lang = lang
        self.text = text

    def __str__(self):
        return 'Format:[lang=\''+self.lang.__str__() + \
               '\', text=\''+self.text+'\']'