import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from lxml import etree
from skimage.draw import polygon
from typing import List

from calamari_ocr.ocr.datasets import DataSet, DataSetMode, DatasetGenerator
from calamari_ocr.utils import split_all_ext, filename


class PageXMLDatasetGenerator(DatasetGenerator):
    def __init__(self, output_queue, mode: DataSetMode, images, xml_files, text_only, epochs, non_existing_as_empty, text_index, skip_invalid):
        super().__init__(output_queue, mode, list(zip(images, xml_files)), text_only, epochs)
        self._non_existing_as_empty = non_existing_as_empty
        self.text_index = text_index
        self.skip_invalid = skip_invalid

    def _load_sample(self, sample, text_only):
        loader = PageXMLDatasetLoader(self.mode, self._non_existing_as_empty, self.text_index, self.skip_invalid)
        image_path, xml_path = sample

        img = None
        if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
            img = np.array(Image.open(image_path))

        for sample in loader.load(image_path, xml_path):
            text = sample["text"]

            if not text_only and (self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN):
                ly, lx = img.shape

                line_img = PageXMLDataset.cutout(img, sample['coords'], lx / sample['img_width'])

                # add padding as required from normal files
                # img = np.pad(img, ((3, 3), (0, 0)), mode='constant', constant_values=img.max())
            else:
                line_img = None

            yield line_img, text


class PageXMLDatasetLoader:
    def __init__(self, mode: DataSetMode, non_existing_as_empty: bool, text_index: int, skip_invalid: bool=True):
        self.mode = mode
        self._non_existing_as_empty = non_existing_as_empty
        self.root = None
        self.text_index = text_index
        self.skip_invalid = skip_invalid

    def load(self, img, xml, skip_commented=True):
        if not os.path.exists(xml):
            if self._non_existing_as_empty:
                return None
            else:
                raise Exception("File '{}' does not exist.".format(xml))

        root = etree.parse(xml).getroot()
        self.root = root

        if self.mode == DataSetMode.TRAIN or self.mode == DataSetMode.EVAL:
            return self._samples_gt_from_book(root, img, skip_commented)
        else:
            return self._samples_from_book(root, img)


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

            yield {
                'ns': ns,
                "rtype": l.xpath('../../@type', namespaces=ns).pop(),
                'xml_element': l,
                "image_path": img,
                "id": l.xpath('../@id', namespaces=ns).pop(),
                "text": text,
                "coords": l.xpath('../ns:Coords/@points',
                                  namespaces=ns).pop(),
                "img_width": img_w
            }

    def _samples_from_book(self, root, img):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if not img.endswith(imgfile):
            raise Exception("Mapping of image file to xml file invalid: {} vs {}".format(img, imgfile))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        for l in root.xpath('//ns:TextLine', namespaces=ns):
            yield {
                'ns': ns,
                "rtype": l.xpath('../@type', namespaces=ns).pop(),
                'xml_element': l,
                "image_path": img,
                "id": l.xpath('./@id', namespaces=ns).pop(),
                "coords": l.xpath('./ns:Coords/@points',
                                  namespaces=ns).pop(),
                "img_width": img_w,
                "text": None,
            }


class PageXMLDataset(DataSet):

    def __init__(self,
                 mode: DataSetMode,
                 files,
                 xmlfiles: List[str] = None,
                 skip_invalid=False,
                 remove_invalid=True,
                 non_existing_as_empty=False,
                 args: dict = None,
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

        if xmlfiles is None:
            xmlfiles = []

        if args is None:
            args = []

        self.text_index = args.get('text_index', 0)

        self._non_existing_as_empty = non_existing_as_empty
        if len(xmlfiles) == 0:
            xmlfiles = [split_all_ext(p)[0] + ".xml" for p in files]

        if len(files) == 0:
            files = [None] * len(xmlfiles)

        self.files = files
        self.xmlfiles = xmlfiles
        self.pages = []
        for img, xml in zip(files, xmlfiles):
            loader = PageXMLDatasetLoader(self.mode, self._non_existing_as_empty, self.text_index, self.skip_invalid)
            for sample in loader.load(img, xml):
                self.add_sample(sample)

            self.pages.append(loader.root)

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

    def _load_sample(self, sample, text_only):
        image_path = sample["image_path"]
        text = sample["text"]
        img = None

        if text_only:
            return img, text

        if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
            img = np.array(Image.open(image_path))

            ly, lx = img.shape
            
            img = PageXMLDataset.cutout(img, sample['coords'], lx / sample['img_width'])

            # add padding as required from normal files
            # img = np.pad(img, ((3, 3), (0, 0)), mode='constant', constant_values=img.max())

        return img, text

    def store_text(self, sentence, sample, output_dir, extension):
        ns = sample['ns']
        line = sample['xml_element']
        textequivxml = line.find('./ns:TextEquiv[@index="{}"]'.format(self.text_index),
                                    namespaces=ns)
        if textequivxml is None:
            textequivxml = etree.SubElement(line, "TextEquiv", attrib={"index": str(self.text_index)})

        u_xml = textequivxml.find('./ns:Unicode', namespaces=ns)
        if u_xml is None:
            u_xml = etree.SubElement(textequivxml, "Unicode")

        u_xml.text = sentence

    def store_extended_prediction(self, data, sample, output_dir, extension):
        output_dir = os.path.join(output_dir, filename(sample['image_path']))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        super().store_extended_prediction(data, sample, output_dir, extension)

    def store(self):
        for xml, page in tqdm(zip(self.xmlfiles, self.pages), desc="Writing PageXML files", total=len(self.xmlfiles)):
            with open(split_all_ext(xml)[0] + ".pred.xml", 'w') as f:
                f.write(etree.tounicode(page.getroottree()))

    def create_generator(self, output_queue, epochs, text_only) -> DatasetGenerator:
        return PageXMLDatasetGenerator(output_queue, self.mode, self.files, self.xmlfiles, text_only, epochs, self._non_existing_as_empty, self.text_index, self.skip_invalid)
