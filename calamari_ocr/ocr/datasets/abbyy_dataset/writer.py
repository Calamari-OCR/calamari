from lxml import etree as ET
import os, stat
from .data import Book, Page
from shutil import copy


class XMLWriter:
    @staticmethod
    def write(page: Page, filename: str):
        self = XMLWriter

        root = ET.Element('document')
        tree = ET.ElementTree(root)

        self._addElement(root, "xmlns", "http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml")
        self._addElement(root, "version", "1.0")
        self._addElement(root, "producer", "Calamari")
        self._addElement(root, "languages", "")
        NS_XSI = "{http://www.w3.org/2001/XMLSchema-instance}"
        root.set(NS_XSI + "schemaLocation",
                 "http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml")

        pageNode = ET.SubElement(root, "page")
        self._addElement(pageNode, "width", page.width)
        self._addElement(pageNode, "height", page.height)
        self._addElement(pageNode, "resolution", page.resolution)
        self._addElement(pageNode, "originalCoords", page.resolution)

        for block in page.blocks:

            blockNode = ET.SubElement(pageNode, "block")
            self._addElement(blockNode, "blockType", block.blockType)
            self._addElement(blockNode, "blockName", block.blockName)
            if block.rect:
                self._addElement(blockNode, "l", block.rect.left.__str__())
                self._addElement(blockNode, "t", block.rect.top.__str__())
                self._addElement(blockNode, "r", block.rect.right.__str__())
                self._addElement(blockNode, "b", block.rect.bottom.__str__())

            textNode = ET.SubElement(blockNode, "text")

            for par in block.pars:

                parNode = ET.SubElement(textNode, "par")
                self._addElement(parNode, "align", par.align)
                self._addElement(parNode, "startIndent", par.startIndent)
                self._addElement(parNode, "lineSpacing", par.lineSpacing)

                for line in par.lines:

                    lineNode = ET.SubElement(parNode, "line")
                    self._addElement(lineNode, "baseline", line.baseline)
                    self._addElement(lineNode, "l", line.rect.left.__str__())
                    self._addElement(lineNode, "t", line.rect.top.__str__())
                    self._addElement(lineNode, "r", line.rect.right.__str__())
                    self._addElement(lineNode, "b", line.rect.bottom.__str__())

                    for fo in line.formats:

                        foNode = ET.SubElement(lineNode, "formatting")
                        self._addElement(foNode, "lang", fo.lang)
                        foNode.text = fo.text

        tree.write(open(filename, 'wb'), encoding='utf-8', xml_declaration=True,
                   pretty_print=True)

    @staticmethod
    def _addElement(element, key, value):

        """
        Only add attributes to an tag if the key is not None

        :param element: the tag element of the xml tree
        :param key: the key of the attribute
        :param value: the value of the attribute
        :return:
        """

        if value is not None:
            element.set(key, value)

