from calamari_ocr.proto import LineGeneratorParameters
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import random
from skimage.transform import rescale
from enum import Enum, IntEnum
from collections import namedtuple
from typing import List


class Script(IntEnum):
    NORMAL = 0
    SUB = -1
    SUPER = 1


Word = namedtuple('Word', ('text', 'script', 'spacing', 'variant'))


def _extendToCanvas(canvas, add, point):
    x, y = point
    new_shape = tuple(np.max(
        [np.add(canvas.shape, -np.min([[y, x], [0, 0]], axis=0)),
         np.add(add.shape, np.max([[y, x], [0, 0]], axis=0))],
        axis=0))
    out = np.full(new_shape, 255, dtype=np.uint8)
    c_start = 0 if y >= 0 else -y
    out[c_start:c_start + canvas.shape[0], :canvas.shape[1]] = canvas
    y += c_start
    out[y:y+add.shape[0], x:x+add.shape[1]] = add
    return out


class FontVariantType(IntEnum):
    NORMAL = 0
    BOLD = 1
    ITALIC = 2
    BOLD_ITALICS = 3


class FontVariants:
    def __init__(self, base_font_ttf: str, font_size: int):
        self.font_name = base_font_ttf if not base_font_ttf.endswith('.ttf') else base_font_ttf[:-4]

        self.default_font = Font(ImageFont.truetype(self.font_name + '.ttf', size=font_size))

        def font_or_default(ttf):
            try:
                return Font(ImageFont.truetype(ttf, size=font_size))
            except OSError:
                print("Error loading font {}. Using default font.".format(ttf))
                return self.default_font

        self.variants = [
            self.default_font,
            font_or_default(self.font_name + '-Bold.ttf'),
            font_or_default(self.font_name + '-Italic.ttf'),
            font_or_default(self.font_name + '-BoldItalic.ttf'),
        ]


class Font:
    def __init__(self, font: ImageFont):
        self.font = font
        self.offset = 0
        self.font = font
        self.baseline = 0
        self.topline = 0
        self.char_width = 0
        self.char_height = 0
        self.center = 0

        # compute top offset
        test_text = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]()_-.;:'\""
            "éèíìóòúù"
            "ăȁĕȅĭŏőŭű"
            "āĀǟǞēĒěīĪōŌȫȪūŪǖǕ"
            "ẹḲḳ"
            "αβγδεζηθικλμνξοπρστυφχψω"
            "½"
            "—"
            "–"
            "℔"
            "šŠ"
            "„“"
            "†")

        image = self.draw(test_text)
        sums = image.mean(axis=1)
        while sums[self.offset + 1] == 255:
            self.offset += 1

        image = self.draw('ABCD')
        self.baseline = image.shape[0]

        image = self.draw('n')
        sums = image.mean(axis=1)
        while sums[self.topline + 1] == 255:
            self.topline += 1

        self.center = (self.baseline + self.topline) // 2
        self.char_height = (self.baseline - self.topline) // 2

        image = self.draw(' ')
        self.char_width = image.shape[1]

    def draw(self, text, scale=1.0, spacing=0):
        if len(text) == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        try:
            spacing = max(0, spacing)
            image = Image.new('L', tuple(np.add(self.font.getsize(text), [int(self.char_width * spacing * len(text) * 2), 0])), 255)
            draw = ImageDraw.Draw(image)
            if spacing == 0 or len(text.strip()) == 0:
                draw.text((0, 0), text, font=self.font)
                image = np.array(image)[self.offset:, :]

            else:
                x = 0
                for i, c in enumerate(text):
                    draw.text((x, 0), c, font=self.font)
                    w, h = self.font.getsize(c)
                    x += int(spacing * self.char_width + w)
                image = np.array(image)[self.offset:, :]

                sums = np.mean(image, axis=0)
                if np.mean(sums) >= 254:
                    # empty image
                    return np.zeros((0, 0), dtype=np.uint8)

                end = len(sums)
                while sums[end - 1] >= 254:
                    end -= 1
                image = image[:, :end]

            if scale != 1:
                image = rescale(image, float(scale), preserve_range=True)

            return image
        except Exception as e:
            print(e)
            print(text, spacing, scale, len(text.strip()))
            raise e


class LineGenerator:
    def __init__(self, line_generator_params: LineGeneratorParameters):
        self.params = line_generator_params
        self.fonts = [FontVariants(f, self.params.font_size) for f in self.params.fonts]

    def draw(self, words: List[Word]):
        font_variants = random.choice(self.fonts)
        canvas = np.zeros((0, 0), dtype=np.uint8)
        offset = int(font_variants.default_font.char_height // 2 + max(abs(self.params.min_script_offset), abs(self.params.max_script_offset)) * font_variants.default_font.char_height)
        x = 0
        for word in words:
            font = font_variants.variants[word.variant]
            scale = [1, 0.5, 0.5][word.script]
            if self.params.min_script_offset == self.params.max_script_offset:
                script_offset = 0
            else:
                script_offset = int(np.random.uniform(self.params.min_script_offset, self.params.max_script_offset) * font.char_height * scale)
            o = [0, font.topline - font.char_height - script_offset, font.baseline - font.char_height - script_offset][word.script]
            img = font.draw(word.text, scale, word.spacing)
            canvas = _extendToCanvas(canvas, img, (x, offset + o))
            x += img.shape[1]

        sums = np.mean(canvas, axis=1)
        cut_top = 0
        while cut_top < len(sums) - 1 and sums[cut_top + 1] == 255:
            cut_top += 1

        canvas = canvas[cut_top:, :]
        return np.pad(canvas, ((3, 3), (3, 3)), 'constant', constant_values=255)

    def _draw(self, text, font: Font = None):
        if font is None:
            font = random.choice(self.fonts)

        return font.draw(text, 0.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    params = LineGeneratorParameters()
    params.font_size = 32
    params.min_script_offset = -0.5
    params.max_script_offset = 0.5
    params.fonts.extend(['Junicode.ttf', 'DejaVuSerif.ttf'])
    line_gen = LineGenerator(params)
    plt.imshow(line_gen.draw(
        [
            Word('test', Script.NORMAL, 0, FontVariantType.NORMAL),
            Word('12345', Script.SUB, 0, FontVariantType.NORMAL),
            Word(' ', Script.NORMAL, 0, FontVariantType.NORMAL),
            Word('norm', Script.NORMAL, 0, FontVariantType.BOLD),
            Word('top', Script.SUPER, 0, FontVariantType.BOLD),
            Word('sub', Script.SUB, 0, FontVariantType.BOLD_ITALICS),
            Word('norm', Script.NORMAL, 1, FontVariantType.BOLD),
            Word('top', Script.SUPER, 1, FontVariantType.BOLD),
            Word('sub', Script.SUB, 1, FontVariantType.BOLD),
            Word('norm', Script.NORMAL, 1, FontVariantType.ITALIC),
            Word('top', Script.SUPER, 1, FontVariantType.ITALIC),
            Word('sub', Script.SUB, 1, FontVariantType.ITALIC),
            Word('norm', Script.NORMAL, 1, FontVariantType.BOLD_ITALICS),
            Word('top', Script.SUPER, 1, FontVariantType.BOLD_ITALICS),
            Word('sub', Script.SUB, 1, FontVariantType.BOLD_ITALICS),
         ]
    ))
    plt.show()

