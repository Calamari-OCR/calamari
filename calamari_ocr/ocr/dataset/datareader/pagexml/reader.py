import os
import numpy as np
from tfaip.base.data.pipeline.definitions import PipelineMode, INPUT_PROCESSOR, TARGETS_PROCESSOR
from tqdm import tqdm
from lxml import etree
from skimage.transform import rotate
import cv2 as cv
from typing import List, Generator

from calamari_ocr.ocr.dataset.params import InputSample, SampleMeta
from calamari_ocr.ocr.dataset.datareader.base import DataReader
from calamari_ocr.ocr.dataset.datareader.factory import FileDataReaderArgs
from calamari_ocr.utils import split_all_ext, filename

import logging

from calamari_ocr.utils.image import load_image

logger = logging.getLogger(__name__)


def xml_attr(elem, ns, label, default=None):
    try:
        return elem.xpath(label, namespaces=ns).pop()
    except IndexError as e:
        if default is None:
            raise e

        return default


class PageXMLDatasetLoader:
    def __init__(self, mode: PipelineMode, non_existing_as_empty: bool, text_index: int, skip_invalid: bool=True):
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

        if self.mode in TARGETS_PROCESSOR:
            return self._samples_gt_from_book(root, img, xml, skip_commented)
        else:
            return self._samples_from_book(root, img, xml)

    def _samples_gt_from_book(self, root, img, page_id,
                              skipcommented=True):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if (self.mode in {PipelineMode.Training, PipelineMode.Evaluation}) and not split_all_ext(img)[0].endswith(split_all_ext(imgfile)[0]):
            raise Exception("Mapping of image file to xml file invalid: {} vs {} (comparing basename {} vs {})".format(
                img, imgfile, split_all_ext(img)[0], split_all_ext(imgfile)[0]))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        textlines = root.xpath('//ns:TextLine', namespaces=ns)

        for textline in textlines:
            tequivs = textline.xpath('./ns:TextEquiv[@index="{}"]'.format(self.text_index),
                                     namespaces=ns)

            if not tequivs:
                tequivs = textline.xpath('./ns:TextEquiv[not(@index)]', namespaces=ns)

            if len(tequivs) > 1:
                logger.warning("PageXML is invalid: TextLine includes TextEquivs with non unique ids")

            parat = textline.attrib
            if skipcommented and "comments" in parat and parat["comments"]:
                continue

            if tequivs is not None and len(tequivs) > 0:
                l = tequivs[0]
                text = l.xpath('./ns:Unicode', namespaces=ns).pop().text
                if text is None:
                    # Handle empty tag as empty string not as "not existing"
                    text = ''
            else:
                l = None
                text = None

            if text is None:
                if self.skip_invalid:
                    continue
                elif self._non_existing_as_empty:
                    text = ""
                else:
                    raise Exception("Empty text field")

            try:
                orientation = float(textline.xpath('../@orientation', namespaces=ns).pop())
            except (ValueError, IndexError):
                orientation = 0

            if self.mode in {PipelineMode.Training, PipelineMode.Evaluation}:
                if len(text) == 0:
                    # Empty lines cannot be used for training (CTC-loss can not be computed)
                    continue

            yield {
                'page_id': page_id,
                'ns': ns,
                "rtype": xml_attr(textline, ns, '../@type', ''),
                'xml_element': l,
                "image_path": img,
                "id": "{}/{}".format(page_id, xml_attr(textline, ns, './@id')),
                "text": text,
                "coords": xml_attr(textline, ns, './ns:Coords/@points'),
                "orientation": orientation,
                "img_width": img_w
            }

    def _samples_from_book(self, root, img, page_id):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if not split_all_ext(img)[0].endswith(split_all_ext(imgfile)[0]):
            raise Exception("Mapping of image file to xml file invalid: {} vs {} (comparing basename {} vs {})".format(
                img, imgfile, split_all_ext(img)[0], split_all_ext(imgfile)[0]))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        for l in root.xpath('//ns:TextLine', namespaces=ns):
            try:
                orientation = float(l.xpath('../@orientation', namespaces=ns).pop())
            except (ValueError, IndexError):
                orientation = 0

            yield {
                'page_id': page_id,
                'ns': ns,
                "rtype": xml_attr(l, ns, '../@type', ''),
                'xml_element': l,
                "image_path": img,
                "id": "{}/{}".format(page_id, xml_attr(l, ns, './@id')),
                "coords": xml_attr(l, ns, './ns:Coords/@points'),
                "orientation": orientation,
                "img_width": img_w,
                "text": None,
            }


class PageXMLReader(DataReader):
    def __init__(self,
                 mode: PipelineMode,
                 files,
                 xmlfiles: List[str] = None,
                 skip_invalid=False,
                 remove_invalid=True,
                 non_existing_as_empty=False,
                 args: FileDataReaderArgs = None,
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
            args = {}

        self.args = args

        self.text_index = args.text_index

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

        # store which pagexml was stored last, to check when a file is ready to be written during sequential prediction
        self._last_page_id = None

    @staticmethod
    def cutout(pageimg, coordstring, scale=1, rect=False, rrect=False):
        """ Cut region from image
        Parameters
        ----------
        pageimg : image (numpy array)
        coordstring : coordinates from PAGE as one string
        scale : factor to scale the coordinates with
        rect : cut out rectangle instead of polygons
        rrect : cut minimum enclosing rectangle instead of polygons
        """
        coords = [p.split(",") for p in coordstring.split()]
        coords = [(int(scale*int(c[0])), int(scale*int(c[1]))) for c in coords]
        coords = np.array(coords, np.int32)
        min_x, max_x = min(c[1] for c in coords), max(c[1] for c in coords)
        min_y, max_y = min(c[0] for c in coords), max(c[0] for c in coords)
        cut = pageimg[min_x:max_x+1, min_y:max_y+1]
        if rect and not rrect:
            return cut
        coords = coords - [min_y, min_x]
        if rrect:
            rect_ma = cv.minAreaRect(coords)
            rect = cv.boxPoints(rect_ma)
            coords = rect.astype(int)
            shift = np.clip(np.min(coords, 0), None, 0)
            coords = coords - shift
            max_d = np.max(coords, 0)
            lu = [min_y, min_x] + shift
            cut = pageimg[lu[1]:lu[1] + max_d[1] + 1,
                          lu[0]:lu[0] + max_d[0] + 1]

        if cut.ndim == 2:
            cval = np.amax(cut)
        else:
            x, y = np.unravel_index(np.argmax(np.mean(cut, axis=2)), cut.shape[:2])
            cval = cut[x, y, :]
        box = np.ones(cut.shape, dtype=cut.dtype) * cval
        mask = np.zeros(cut.shape[:2], dtype=cut.dtype)
        mask = cv.fillPoly(mask, [coords], color=1).astype(np.bool)
        box[mask] = cut[mask]
        return box

    def prepare_store(self):
        self._last_page_id = None

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

        # check if page can be stored, this requires that (standard in prediction) the pages are passed sequentially
        if self._last_page_id != sample['page_id']:
            if self._last_page_id:
                self._store_page(extension, self._last_page_id)
            self._last_page_id = sample['page_id']

    def store_extended_prediction(self, data, sample, output_dir, extension):
        output_dir = os.path.join(output_dir, filename(sample['image_path']))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        super().store_extended_prediction(data, sample, output_dir, extension)

    def store(self, extension):
        if self._last_page_id:
            self._store_page(extension, self._last_page_id)
            self._last_page_id = None
        else:
            for xml, page in tqdm(zip(self.xmlfiles, self.pages), desc="Writing PageXML files", total=len(self.xmlfiles)):
                with open(split_all_ext(xml)[0] + extension, 'w') as f:
                    f.write(etree.tounicode(page.getroottree()))

    def _store_page(self, extension, page_id):
        page = self.pages[self.xmlfiles.index(page_id)]
        with open(split_all_ext(page_id)[0] + extension, 'w') as f:
            f.write(etree.tounicode(page.getroottree()))

    def _sample_iterator(self):
        return zip(self.files, self.xmlfiles, range(len(self.files)))

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        loader = PageXMLDatasetLoader(self.mode, self._non_existing_as_empty, self.text_index, self.skip_invalid)
        image_path, xml_path, idx = sample

        img = None
        if self.mode in INPUT_PROCESSOR:
            img = load_image(image_path)

        for i, sample in enumerate(loader.load(image_path, xml_path)):
            fold_id = (idx + i) % self.n_folds if self.n_folds > 0 else -1
            text = sample["text"]
            orientation = sample["orientation"]

            if not text_only and self.mode in INPUT_PROCESSOR:
                ly, lx = img.shape[:2]

                line_img = PageXMLReader.cutout(img, sample['coords'], lx / sample['img_width'])

                # rotate by orientation angle in clockwise direction to correct present skew
                # (skimage rotates in counter-clockwise direction)
                if orientation and orientation % 360 != 0:
                    line_img = rotate(line_img, orientation*-1, resize=True, mode='constant', cval=line_img.max(), preserve_range=True).astype(np.uint8)

                # add padding as required from normal files
                if self.args.pad:
                    pad = self.args.pad
                    img = np.pad(img, pad, mode='constant', constant_values=img.max())
            else:
                line_img = None

            yield InputSample(line_img, text, SampleMeta(id=sample['id'], fold_id=fold_id))
