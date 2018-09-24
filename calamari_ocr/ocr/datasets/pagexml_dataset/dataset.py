import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from lxml import etree, html
from lxml.etree import Element, ElementTree
from skimage.draw import polygon

from calamari_ocr.ocr.datasets import DataSet, DataSetMode
from calamari_ocr.utils import split_all_ext


class PageXMLDataset(DataSet):

    def __init__(self,
                 mode: DataSetMode,
                 files,
                 xmlfiles=list(),
                 skip_invalid=False,
                 remove_invalid=True,
                 non_existing_as_empty=False,
                 args=dict(),
                 ):

        """ Create a dataset from a Path as String

        Parameters
         ----------
        files : [], required
            image files
        skip_invalid : bool, optional
            skip invalid files
        remove_invalid : bool, optional
            remove invalid files
        """

        super().__init__(
            mode,
            skip_invalid, remove_invalid,
        )

        self.text_index = args.get('text_index', 0)

        self._non_existing_as_empty = non_existing_as_empty
        if not xmlfiles or len(xmlfiles) == 0:
            xmlfiles = [split_all_ext(p)[0] + ".xml" for p in files]

        if not files or len(files) == 0:
            files = [None] * len(xmlfiles)

        self.files = files
        self.xmlfiles = xmlfiles

        self.pages = [self.read_page_xml(img, xml) for img, xml in zip(files, xmlfiles)]

    @staticmethod
    def cutout(pageimg, coordstring, scale=1, rect=False):
        coords = [p.split(",") for p in coordstring.split()]
        coords = np.array([(int(scale * int(c[1])), int(scale * int(c[0])))
                           for c in coords])
        if rect:
            return pageimg[min(c[0] for c in coords):max(c[0] for c in coords),
                   min(c[1] for c in coords):max(c[1] for c in coords)]
        rr, cc = polygon(coords[:, 0], coords[:, 1], pageimg.shape)
        offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
        box = np.ones(
            (max([x[0] for x in coords]) - offset[0],
             max([x[1] for x in coords]) - offset[1]),
            dtype=pageimg.dtype) * 255
        box[rr - offset[0], cc - offset[1]] = pageimg[rr, cc]
        return box

    def _load_sample(self, sample):
        image_path = sample["imgfile"]
        text = sample["text"]

        img = None
        if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
            img = np.array(Image.open(image_path))

            ly, lx = img.shape
            
            img = PageXMLDataset.cutout(img, sample['coords'], lx / sample['img_width'])

            # add padding as required from normal files
            # img = np.pad(img, ((3, 3), (0, 0)), mode='constant', constant_values=img.max())

        return img, text

    def store_text(self, sentence, sample, output_dir, extension):
        ns = sample['ns']
        line: Element = sample['xml_element']
        textequivxml = line.find('./ns:TextEquiv[@index="{}"]'.format(self.text_index),
                                    namespaces=ns)
        if textequivxml is None:
            textequivxml = etree.SubElement(line, "TextEquiv", attrib={"index": str(self.text_index)})

        u_xml = textequivxml.find('./ns:Unicode', namespaces=ns)
        if u_xml is None:
            u_xml = etree.SubElement(textequivxml, "Unicode")

        u_xml.text = sentence

    def store(self):
        for xml, page in tqdm(zip(self.xmlfiles, self.pages), desc="Writing PageXML files", total=len(self.xmlfiles)):
            with open(split_all_ext(xml)[0] + ".pred.xml", 'w') as f:
                f.write(etree.tounicode(page.getroottree()))

    def read_page_xml(self, img, xml, skipcommented=True):
        if not os.path.exists(xml):
            if self._non_existing_as_empty:
                return None
            else:
                raise Exception("File '{}' does not exist.".format(xml))

        root = etree.parse(xml).getroot()

        if self.mode == DataSetMode.TRAIN or self.mode == DataSetMode.EVAL:
            self._samples_gt_from_book(root, img, skipcommented)
        else:
            self._samples_from_book(root, img)

        return root

    def _samples_gt_from_book(self, root, img,
                              skipcommented=True):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if self.mode == DataSetMode.TRAIN and not img.endswith(imgfile):
            raise Exception("Mapping of image file to xml file invalid: {} vs {}".format(img, imgfile))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        tequivs = root.xpath('//ns:TextEquiv[@index="{}"]'.format(self.text_index),
                             namespaces=ns)
        for l in tequivs:
            parat = l.getparent().attrib
            if skipcommented and "comments" in parat and parat["comments"]:
                continue

            text = l.xpath('./ns:Unicode', namespaces=ns).pop().text
            if not text:
                if self.skip_invalid:
                    continue
                elif self._non_existing_as_empty:
                    text = ""
                else:
                    raise Exception("Empty text field")

            self.add_sample({
                'ns': ns,
                "rtype": l.xpath('../../@type', namespaces=ns).pop(),
                'xml_element': l,
                "imgfile": img,
                "id": l.xpath('../@id', namespaces=ns).pop(),
                "text": text,
                "coords": l.xpath('../ns:Coords/@points',
                                  namespaces=ns).pop(),
                "img_width": img_w
            })

    def _samples_from_book(self, root, img):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if not img.endswith(imgfile):
            raise Exception("Mapping of image file to xml file invalid: {} vs {}".format(img, imgfile))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        for l in root.xpath('//ns:TextLine', namespaces=ns):
            self.add_sample({
                'ns': ns,
                "rtype": l.xpath('../@type', namespaces=ns).pop(),
                'xml_element': l,
                "imgfile": img,
                "id": l.xpath('./@id', namespaces=ns).pop(),
                "coords": l.xpath('./ns:Coords/@points',
                                  namespaces=ns).pop(),
                "img_width": img_w,
                "text": None,
            })
