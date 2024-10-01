import os
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
import math

import numpy as np
from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import (
    PipelineMode,
    INPUT_PROCESSOR,
    TARGETS_PROCESSOR,
)
from tqdm import tqdm
from lxml import etree
import cv2 as cv
from typing import List, Generator, Optional, Iterable, Dict, Any, Tuple
from enum import IntEnum
from calamari_ocr.ocr.dataset.datareader.base import (
    CalamariDataGenerator,
    CalamariDataGeneratorParams,
    InputSample,
    SampleMeta,
)
from calamari_ocr.utils import split_all_ext, filename, glob_all
from calamari_ocr.ocr.predict.params import Prediction

import logging


logger = logging.getLogger(__name__)


class CutMode(IntEnum):
    BOX = 0
    POLYGON = 1
    MBR = 2


class PageXMLDatasetLoader:
    def __init__(
        self,
        mode: PipelineMode,
        non_existing_as_empty: bool,
        text_index: int,
        skip_invalid: bool = True,
        skip_commented=False,
    ):
        self.mode = mode
        self._non_existing_as_empty = non_existing_as_empty
        self.root = None
        self.text_index = text_index
        self.skip_invalid = skip_invalid
        self.skip_commented = skip_commented

    def load(self, img, xml) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(xml):
            if self.skip_invalid:
                logger.warning(f"File '{xml}' does not exist. Skipping since `skip_invalid=True`.")
                return []
            else:
                raise FileNotFoundError(f"File '{xml}' does not exist.")

        # remove_blank_text=True is needed so we can add tags to the tree without the pretty printer breaking
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(xml, parser).getroot()
        self.root = root

        page_id = split_all_ext(xml)[0]
        if self.mode in TARGETS_PROCESSOR:
            return self._samples_gt_from_book(root, img, page_id)
        else:
            return self._samples_from_book(root, img, page_id)

    def _samples_gt_from_book(self, root, img, page_id) -> Iterable[Dict[str, Any]]:
        ns = {"ns": root.nsmap[root.prefix]}
        page = root.find(".//ns:Page", namespaces=ns)
        imgfile = page.attrib.get("imageFilename")
        if (self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}) and not split_all_ext(img)[0].endswith(
            split_all_ext(imgfile)[0]
        ):
            logger.warning(
                "Mapping of image file to xml file invalid: {} vs {} (comparing basename {} vs {})".format(
                    img, imgfile, split_all_ext(img)[0], split_all_ext(imgfile)[0]
                )
            )

        img_w = int(page.attrib.get("imageWidth"))
        textlines = root.findall(".//ns:TextLine", namespaces=ns)

        for textline in textlines:
            tequivs = textline.findall('./ns:TextEquiv[@index="{}"]'.format(self.text_index), namespaces=ns)

            if not tequivs:
                tequivs = [te for te in textline.findall("./ns:TextEquiv", namespaces=ns) if "index" not in te.attrib]

            if len(tequivs) > 1:
                logger.warning("PageXML is invalid: TextLine includes TextEquivs with non unique ids")

            if self.skip_commented and len(textline.attrib.get("comments", "")):
                continue

            if tequivs is not None and len(tequivs) > 0:
                l = tequivs[0]
                uc = l.find("./ns:Unicode", namespaces=ns)
                text = uc.text if uc is not None else ""
                if text is None:
                    # Handle empty tag as empty string not as "not existing"
                    text = ""
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

            orientation = float(textline.getparent().attrib.get("orientation", default=0))

            if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
                if len(text) == 0:
                    # Empty lines cannot be used for training (CTC-loss can not be computed)
                    continue

            yield {
                "page_id": page_id,
                "ns": ns,
                "rtype": textline.getparent().attrib.get("type", default=""),
                "xml_element": textline,
                "image_path": img,
                "id": "{}/{}".format(page_id, textline.attrib.get("id")),
                "base_name": textline.attrib.get("id"),
                "text": text,
                "coords": textline.find("./ns:Coords", namespaces=ns).attrib.get("points"),
                "orientation": orientation,
                "img_width": img_w,
            }

    def _samples_from_book(self, root, img, page_id) -> Iterable[Dict[str, Any]]:
        ns = {"ns": root.nsmap[root.prefix]}
        page = root.find(".//ns:Page", namespaces=ns)
        imgfile = page.attrib.get("imageFilename")
        if not split_all_ext(img)[0].endswith(split_all_ext(imgfile)[0]):
            logger.warning(
                "Mapping of image file to xml file invalid: {} vs {} (comparing basename {} vs {})".format(
                    img, imgfile, split_all_ext(img)[0], split_all_ext(imgfile)[0]
                )
            )

        img_w = int(page.attrib.get("imageWidth"))
        for textline in root.findall(".//ns:TextLine", namespaces=ns):
            if self.skip_commented and len(textline.attrib.get("comments", "")):
                continue
            orientation = float(textline.getparent().attrib.get("orientation", default=0))

            yield {
                "page_id": page_id,
                "ns": ns,
                "rtype": textline.getparent().attrib.get("type", default=""),
                "xml_element": textline,
                "image_path": img,
                "id": "{}/{}".format(page_id, textline.attrib.get("id")),
                "base_name": textline.attrib.get("id"),
                "coords": textline.find("./ns:Coords", namespaces=ns).attrib.get("points"),
                "orientation": orientation,
                "img_width": img_w,
                "text": None,
            }


@pai_dataclass
@dataclass
class PageXML(CalamariDataGeneratorParams):
    images: List[str] = field(default_factory=list)
    xml_files: List[str] = field(default_factory=list)
    gt_extension: str = field(
        default=".xml",
        metadata=pai_meta(help="Default extension of the gt files (expected to exist in same dir)"),
    )
    text_index: int = 0
    pad: Optional[List[int]] = field(
        default=None,
        metadata=pai_meta(help="Additional padding after lines were cut out."),
    )
    pred_extension: str = field(
        default=".pred.xml",
        metadata=pai_meta(help="Default extension of the prediction files"),
    )
    skip_commented: bool = field(default=False, metadata=pai_meta(help='Skip lines with "comments" attribute.'))
    cut_mode: CutMode = field(default=CutMode.POLYGON, metadata=pai_meta(help="Mode for cutting out the lines."))
    output_confidences: bool = field(
        default=False, metadata=pai_meta(help="Write prediction confidences into the output.")
    )
    output_glyphs: bool = field(
        default=False, metadata=pai_meta(help="Output the words and glyphs each line is made up of.")
    )
    max_glyph_alternatives: int = field(
        default=1,
        metadata=pai_meta(
            help="When output_glyphs is True, determines the maximum amount of glyph alternatives to output."
        ),
    )
    delete_old_words: bool = field(
        default=True,
        metadata=pai_meta(
            help="If there are already words in the input, "
            + "delete them instead of writing the new ones alongside them."
        ),
    )

    def __len__(self):
        return len(self.images)

    def select(self, indices: List[int]):
        if self.images:
            self.images = [self.images[i] for i in indices]
        if self.xml_files:
            self.xml_files = [self.xml_files[i] for i in indices]

    def to_prediction(self):
        self.xml_files = sorted(glob_all(self.xml_files))
        pred = deepcopy(self)
        pred.xml_files = [split_all_ext(f)[0] + self.pred_extension for f in self.xml_files]
        return pred

    @staticmethod
    def cls():
        return PageXMLReader

    def prepare_for_mode(self, mode: PipelineMode):
        self.images = sorted(glob_all(self.images))
        self.xml_files = sorted(glob_all(self.xml_files))
        if not self.xml_files:
            self.xml_files = [split_all_ext(f)[0] + self.gt_extension for f in self.images]
        if not self.images:
            self.images = [None] * len(self.xml_files)

        if len(self.images) != len(self.xml_files):
            raise ValueError(f"Different number of image and xml files, {len(self.images)} != {len(self.xml_files)}")
        for img_path, xml_path in zip(self.images, self.xml_files):
            if img_path and xml_path:
                img_bn, xml_bn = split_all_ext(img_path)[0], split_all_ext(xml_path)[0]
                if img_bn != xml_bn:
                    logger.warning(
                        f"Filenames are not matching, got base names \n" f"  image: {img_bn}\n" f"  xml:   {xml_bn}\n."
                    )


class PageXMLReader(CalamariDataGenerator[PageXML]):
    def __init__(
        self,
        mode: PipelineMode,
        params: PageXML,
    ):
        super().__init__(mode, params)
        self.pages = {}
        for img, xml in zip(params.images, params.xml_files):
            loader = PageXMLDatasetLoader(
                self.mode,
                params.non_existing_as_empty,
                params.text_index,
                params.skip_invalid,
                params.skip_commented,
            )
            for sample in loader.load(img, xml):
                self.add_sample(sample)

            self.pages[split_all_ext(xml)[0]] = loader.root

        # store which pagexml was stored last, to check when a file is ready to be written during sequential prediction
        self._last_page_id = None

        # counter for word tag ids
        self._next_word_id = 0

    @staticmethod
    def cutout(
        pageimg: np.array,
        coordstring: str,
        mode: CutMode,
        angle=0,
        max_auto_angle=0,
        cval=None,
        scale=1,
    ):
        """Cut region from image
        Parameters
        ----------
        pageimg : page image
        coordstring : coordinates from PAGE in the form "c1_1,c_2 c2_1,c2_2 ..."
        mode :
            CutMode.BOX : cut straight rectangle around coordinates
            CutMode.POLYGON : cut polygon around coordinates
            CutMode.MBR : cut minimum bounding rectangle around coordinates
        angle :
            float : rotate angle in clockwise direction
            None : guess angle from minimum bounding rectangle
        max_auto_angle :
            float : if angle is None, try to guess angle up to boundary
        cval :
            colour : mask and fill empty regions with
            None : calculate via maximum pixel
        scale : factor to scale the coordinates with
        """

        coords = [p.split(",") for p in coordstring.split()]
        if not coords:
            return pageimg[0:0, 0:0]
        coords = [(int(scale * int(c[1])), int(scale * int(c[0]))) for c in coords]
        coords = np.array(coords, np.int32).reshape((-1, 1, 2))
        maxX, maxY = np.amax(coords, 0).squeeze()
        minX, minY = np.amin(coords, 0).squeeze()
        cut = pageimg[minX : maxX + 1, minY : maxY + 1]
        if cut.size == 0:
            return cut  # empty image
        coords -= (minX, minY)
        maxX, maxY = (maxX - minX, maxY - minY)
        minX, minY = (0, 0)

        # calculate angle if needed
        if angle is None:
            if max_auto_angle > 0:
                mbr = cv.minAreaRect(coords)
                angle = mbr[2] - 90 if mbr[2] > 45 else mbr[2]
                if abs(angle) > max_auto_angle:
                    angle = 0
            else:
                angle = 0

        # set cval if needed
        if cval is None:
            if cut.ndim == 2:
                cval = np.amax(cut).item()
            else:
                x, y = np.unravel_index(np.argmax(np.mean(cut, axis=2)), cut.shape[:2])
                cval = cut[x, y, :].tolist()

        # rotate cut
        if angle:
            (h, w) = cut.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            # compute the new bounding dimensions of the image
            nW = np.ceil((h * sin) + (w * cos)).astype(int)
            nH = np.ceil((h * cos) + (w * sin)).astype(int)
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            # rotate coords
            coords = cv.transform(coords[..., ::-1], M)
            minX, minY = np.amin(coords, 0).squeeze()
            maxX, maxY = np.amax(coords, 0).squeeze()
            # rotate image
            cut = cv.warpAffine(
                cut,
                M,
                (nW, nH),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=cval,
            )
        else:
            coords = coords[..., ::-1]
            minX, minY = minY, minX
            maxX, maxY = maxY, maxX

        # simplify coordinates with MBR
        if mode is CutMode.MBR:
            mbr = cv.minAreaRect(coords)
            coords = cv.boxPoints(mbr).astype(int).reshape(-1, 1, 2)

        # mask pixels outside coords
        if mode in (CutMode.POLYGON, CutMode.MBR):
            box = (np.ones(cut.shape) * cval).astype(cut.dtype)
            mask = np.zeros(cut.shape, dtype=np.uint8)
            mask = cv.fillPoly(mask, [coords], color=[255] * cut.ndim)
            mask_inv = cv.bitwise_not(mask)
            fg = cv.bitwise_and(cut, mask)
            bg = cv.bitwise_and(box, mask_inv)
            cut = cv.add(fg, bg)

        return cut[minY : maxY + 1, minX : maxX + 1]

    def prepare_store(self):
        self._last_page_id = None
        self._next_word_id = 0
        self._output_dir = dict()

    def store_text_prediction(self, prediction, sample_id, output_dir):
        sentence = prediction.sentence
        sample = self.sample_by_id(sample_id)
        output_dir = output_dir or os.path.dirname(sample["page_id"])
        self._output_dir[sample["page_id"]] = output_dir
        ns = sample["ns"]
        line = sample["xml_element"]
        textequivxml = line.find('./ns:TextEquiv[@index="{}"]'.format(self.params.text_index), namespaces=ns)
        if textequivxml is None:
            textequivxml = etree.SubElement(line, "TextEquiv", attrib={"index": str(self.params.text_index)})

        u_xml = textequivxml.find("./ns:Unicode", namespaces=ns)
        if u_xml is None:
            u_xml = etree.SubElement(textequivxml, "Unicode")

        u_xml.text = sentence

        if self.params.output_glyphs:
            words = self._words_from_prediction(prediction)

            # delete or rename old words before writing the new ones
            self._store_old_words(line, ns)
            self._store_words(words, line, self._parse_coords(sample["coords"]), ns)

        if self.params.output_confidences:
            textequivxml.set("conf", str(prediction.avg_char_probability))

        # check if page can be stored, this requires that (standard in prediction) the pages are passed sequentially
        if self._last_page_id != sample["page_id"]:
            if self._last_page_id:
                self._store_page(self.params.pred_extension, self._last_page_id)
            self._last_page_id = sample["page_id"]

    def store_extended_prediction(self, data, sample, output_dir, extension):
        output_dir = os.path.join(output_dir, filename(sample["image_path"]))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        super().store_extended_prediction(data, sample, output_dir, extension)

    def store(self):
        extension = self.params.pred_extension
        if self._last_page_id:
            self._store_page(extension, self._last_page_id)
            self._last_page_id = None
        else:
            for xml in tqdm(
                self.params.xmlfiles,
                desc="Writing PageXML files",
                total=len(self.params.xmlfiles),
            ):
                page_id = split_all_ext(xml)[0]
                page = self.pages(page_id)
                path = os.path.join(self._output_dir[page_id], filename(xml) + extension)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(etree.tounicode(page.getroottree(), pretty_print=True))

    @staticmethod
    def _parse_coords(coords: str) -> List[Tuple[int, int]]:
        points = []

        for coord in coords.split():
            x, y = coord.split(sep=",", maxsplit=2)
            points.append((int(x), int(y)))

        return points

    @staticmethod
    def _bounding_rect_from_points(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_x, min_y = math.inf, math.inf
        max_x, max_y = -math.inf, -math.inf

        for x, y in points:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        width, height = max_x - min_x, max_y - min_y
        return min_x, min_y, width, height

    @staticmethod
    def _coords_for_rectangle(x, y, width, height):
        return f"{int(x)},{int(y)} {int(x+width)},{int(y)} {int(x+width)},{int(y+height)} {int(x)},{int(y+height)}"

    @staticmethod
    def _make_subelement(parent, tag, attrib=None):
        tag = "{" + parent.nsmap.get(parent.prefix, "") + "}" + tag
        return etree.SubElement(parent, tag, attrib=attrib, nsmap=parent.nsmap)

    def _store_old_word(self, word_xml, ns):
        word_xml.set("id", f"{word_xml.get('id')}_old")

        for glyph_xml in word_xml.findall("./ns:Glyph", namespaces=ns):
            glyph_xml.set("id", f"{glyph_xml.get('id')}_old")

    def _store_old_words(self, line_xml, ns):
        for word_xml in line_xml.findall("./ns:Word", namespaces=ns):
            if self.params.delete_old_words:
                # or rather, don't store old words
                line_xml.remove(word_xml)
            else:
                # append _old suffix to old words and their glyphs
                self._store_old_word(word_xml, ns)

    def _store_glyph(self, glyph, word_id, word_xml, line_x, line_y, line_height, glyph_counter, ns):
        glyph_id = f"{word_id}g{str(glyph_counter)}"

        glyph_xml = word_xml.find(f'./ns:Glyph[@id="{glyph_id}"]', namespaces=ns)
        if glyph_xml is None:
            glyph_xml = self._make_subelement(word_xml, "Glyph", attrib={"id": glyph_id})

        coords_xml = glyph_xml.find("./ns:Coords", namespaces=ns)
        if coords_xml is None:
            coords_xml = self._make_subelement(glyph_xml, "Coords")

        glyph_x, glyph_y = glyph.global_start + line_x, line_y
        glyph_width, glyph_height = glyph.global_end - glyph.global_start, line_height
        coords_xml.set("points", self._coords_for_rectangle(glyph_x, glyph_y, glyph_width, glyph_height))

        for index in range(min(len(glyph.chars), self.params.max_glyph_alternatives)):
            char, confidence = glyph.chars[index].char, glyph.chars[index].probability

            glyph_index = self.params.text_index + index

            textequiv_xml = glyph_xml.find(f'./ns:TextEquiv[@index="{glyph_index}"]', namespaces=ns)
            if textequiv_xml is None:
                textequiv_xml = self._make_subelement(glyph_xml, "TextEquiv")
                textequiv_xml.set("index", str(glyph_index))

            if self.params.output_confidences:
                textequiv_xml.set("conf", str(confidence))

            u_xml = textequiv_xml.find("./ns:Unicode", namespaces=ns)
            if u_xml is None:
                u_xml = self._make_subelement(textequiv_xml, "Unicode")
            u_xml.text = char

    def _store_words(self, words, line_xml, line_coords, ns) -> float:
        # page schema requires that word tags are directly after coords (and baseline, if present)
        line_coords_xml = line_xml.find("./ns:Coords", namespaces=ns)
        line_baseline_xml = line_xml.find("./ns:Baseline", namespaces=ns)

        if line_baseline_xml is not None:
            # if there is a baseline element, insert after it
            insert_index = line_xml.index(line_baseline_xml) + 1
        elif line_coords_xml is not None:
            # if there is a coords element, but no baseline element, insert after it
            insert_index = line_xml.index(line_coords_xml) + 1
        else:
            # otherwise, insert at the start
            insert_index = 0

        line_x, line_y, _, line_height = self._bounding_rect_from_points(line_coords)

        for word in words:
            if not word:
                # ignore empty words
                continue

            word_id = "w" + str(self._next_word_id)

            self._next_word_id += 1

            # find if we already have words with this id and overwrite them
            word_xml = line_xml.find(f'./ns:Word[@id="{word_id}"]', namespaces=ns)
            if word_xml is None:
                # no word with this id, create a new word element
                word_xml = self._make_subelement(line_xml, "Word", attrib={"id": word_id})

            coords_xml = word_xml.find("./ns:Coords", namespaces=ns)
            if coords_xml is None:
                coords_xml = self._make_subelement(word_xml, "Coords")

            word_text = ""
            word_confidence = 1
            glyph_counter = 0

            for glyph in word:
                word_text += glyph.chars[0].char
                word_confidence *= glyph.chars[0].probability

                self._store_glyph(glyph, word_id, word_xml, line_x, line_y, line_height, glyph_counter, ns)
                glyph_counter += 1

            # check if a TextEquiv with this index already exists

            textequiv_xml = word_xml.find(f'./ns:TextEquiv[@index="{self.params.text_index}"]', namespaces=ns)
            if textequiv_xml is None:
                textequiv_xml = self._make_subelement(word_xml, "TextEquiv")
                textequiv_xml.set("index", str(self.params.text_index))

            if self.params.output_confidences:
                textequiv_xml.set("conf", str(word_confidence))

            u_xml = textequiv_xml.find("./ns:Unicode", namespaces=ns)
            if u_xml is None:
                u_xml = self._make_subelement(textequiv_xml, "Unicode")
            u_xml.text = word_text

            word_x, word_y = word[0].global_start + line_x, line_y
            word_width, word_height = word[-1].global_end - word[0].global_start, line_height
            coords_xml.set("points", self._coords_for_rectangle(word_x, word_y, word_width, word_height))

            line_xml.insert(insert_index, word_xml)
            insert_index += 1

    # groups prediction positions by word, removing spaces
    @staticmethod
    def _words_from_prediction(prediction: Prediction) -> list:
        words = []
        current_word = []

        for pos in prediction.positions:
            char = pos.chars[0].char
            if char == " ":
                words.append(current_word)
                current_word = []
                continue
            current_word.append(pos)

        if current_word:
            words.append(current_word)

        return words

    def _store_page(self, extension, page_id):
        page = self.pages[page_id]
        path = os.path.join(self._output_dir[page_id], filename(page_id) + extension)
        with open(path, "w", encoding="utf-8") as f:
            f.write(etree.tounicode(page.getroottree(), pretty_print=True))

    def _sample_iterator(self):
        all_samples = zip(self.params.images, self.params.xml_files, range(len(self.params.images)))
        if self.mode == PipelineMode.TRAINING:
            all_samples = list(all_samples)
            shuffle(all_samples)
        return all_samples

    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        loader = PageXMLDatasetLoader(
            self.mode,
            self.params.non_existing_as_empty,
            self.params.text_index,
            self.params.skip_invalid,
            self.params.skip_commented,
        )
        image_path, xml_path, idx = sample

        img = None
        if self.mode in INPUT_PROCESSOR:
            img = self._load_image(image_path)

        for i, sample in enumerate(loader.load(image_path, xml_path)):
            fold_id = (idx + i) % self.params.n_folds if self.params.n_folds > 0 else -1
            text = sample["text"]
            orientation = sample["orientation"]

            if not text_only and self.mode in INPUT_PROCESSOR:
                ly, lx = img.shape[:2]

                # rotate by orientation angle in clockwise direction to correct present skew
                angle = orientation if orientation and orientation % 360 != 0 else 0

                line_img = PageXMLReader.cutout(
                    img,
                    sample["coords"],
                    mode=self.params.cut_mode,
                    angle=angle,
                    cval=None,
                    scale=lx / sample["img_width"],
                )

                # add padding as required from normal files
                if self.params.pad:
                    img = np.pad(
                        img,
                        self.params.pad,
                        mode="constant",
                        constant_values=img.max(initial=0),
                    )
            else:
                line_img = None

            yield InputSample(line_img, text, SampleMeta(id=sample["id"], fold_id=fold_id))
