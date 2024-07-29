from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Iterator, Set

from paiargparse import pai_dataclass, pai_meta
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tfaip.data.pipeline.definitions import PipelineMode

from calamari_ocr.utils import glob_all

if TYPE_CHECKING:
    from tfaip.data.pipeline.datapipeline import DataPipeline


@pai_dataclass
@dataclass
class CodecConstructionParams:
    keep_loaded: bool = field(
        default=True,
        metadata=pai_meta(help="Fully include the codec of the loaded model to the new codec"),
    )
    auto_compute: bool = field(
        default=True,
        metadata=pai_meta(help="Compute the codec automatically. See also include."),
    )
    include: List[str] = field(
        default_factory=list,
        metadata=pai_meta(
            help="Whitelist of characters that may not be removed on restoring a model. "
            "For large dataset you can use this to skip the automatic codec computation "
            "(see auto_compute)"
        ),
    )
    include_files: List[str] = field(
        default_factory=list,
        metadata=pai_meta(help="Whitelist of txt files that may not be removed on restoring a model"),
    )

    resolved_include_chars: Set[str] = field(default_factory=set, metadata=pai_meta(mode="ignore"))

    def __post_init__(self):
        # parse whitelist
        if len(self.include) == 1:
            include = set(self.include[0])
        else:
            include = set(self.include)

        for f in glob_all(self.include_files):
            with open(f) as txt:
                include = include.union(txt.read())

        self.resolved_include_chars = include


@pai_dataclass(no_assign_to_unknown=False)
@dataclass
class Codec:
    charset: List[str]  # this filed will be used to store and load a the Codec from json

    @staticmethod
    def from_input_dataset(
        data_pipelines: Iterator["DataPipeline"],
        codec_construction_params: CodecConstructionParams,
        progress_bar=False,
    ):
        chars = codec_construction_params.resolved_include_chars

        for pipeline in data_pipelines:
            pipeline = pipeline.to_mode(PipelineMode.TARGETS)
            pipeline = pipeline.as_preloaded(progress_bar=progress_bar)
            data_gen = pipeline.create_data_generator()
            for sample in tqdm_wrapper(
                data_gen.generate(),
                total=len(data_gen),
                desc="Computing codec",
                progress_bar=progress_bar,
            ):
                # assert(sample.inputs is None)  # Inputs shall not be generated here
                for c in sample.targets:
                    chars.add(c)

        return Codec(sorted(list(chars)))

    @staticmethod
    def from_texts(texts: List[str], codec_construction_params: CodecConstructionParams):
        """Compute a codec from given text

        First computes a set of all available characters.
        Then, a Codec is created

        Parameters
        ----------
        texts : obj:`list` of :obj:`str`
            a list of strings
        whitelist
            a list of characters that are forced to be in the codec
        Returns
        -------
            Codec based on the set of characters + whitelist
        """
        chars = set() if codec_construction_params.include is None else set(codec_construction_params.include)

        for text in texts:
            for c in text:
                chars.add(c)

        return Codec(sorted(list(chars)))

    def __post_init__(self):
        """Construct a codec based on a given charset (symbols)

        A symbol is typically a character (e.g. a, b, c, d, ...) in OCR, in OMR this might be
        the position of a note in the staff.
        The labels are required for training, since the DNN will only predict integer numbers (classes).
        Given a set of symbols the codec will automatically assign a label to each character and store it for
        processing of a line.

        The codec then is used to `decode` and `encode` a line.


        As first index a __blank__ (empty string) will be added as required for the CTC algorithm.
        """
        charset = list(self.charset)
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
        """Get the number of characters in the charset

        this is equal to the maximum possible label.

        Returns
        -------
            number of characters in the charset
        """
        return len(self.charset)

    def size(self):
        """Get the number of characters in the charset

        this is equal to the maximum possible label.

        Returns
        -------
            number of characters in the charset
        """
        return len(self.charset)

    def encode(self, s):
        """Encode the string into labels

        Parameters
        ----------
        s : str
            sequence of characters

        Returns
        -------
            sequence of labels

        See Also
        --------
            decode
        """
        return [self.char2code[c] for c in s]

    def decode(self, l):
        """Decode the sequence of labels into a sequence of characters

        Parameters
        ----------
        l : list of int
            sequence of labels as predicted by the neural net

        Returns
        -------
            sequence of characters

        See Also
        --------
            encode
        """
        return [self.code2char[c] for c in l]

    def extend(self, codec):
        """extend the codec by the given characters

        If a character is already present it will be skipped.
        The new characters will be added at the end of the codec (highest label numbers)

        Parameters
        ----------
        codec : list of str
            the characters to add
        Returns
        -------
        list of int
            the positions/labels of added characters
        See Also
        --------
            shrink
        """
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
        """remove the given `codec` from this Codec

        This algorithm will compute the positions of the codes in the old charset and ignore non present chars.
        This output can then be used to delete specific nodes in the neural net.

        Parameters
        ----------
        codec : list of str
            chars to remove if present
        Returns
        -------
        list of int
            positions/labels of the chars that shall be removed on the old charset
        """
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

    def align(self, codec, shrink=True, extend=True):
        """Change the codec to the new `codec` but keep the positions of chars that are in both codecs.

        This function is used to compute a codec change: deleted labels, added characters.

        Parameters
        ----------
        codec : list of str
            Characters of the new codec
        shrink : bool
            Shrink the codec by unavailable chars
        extend : bool
            Extend new chars to the codec
        Returns
        -------
        list of int
            list of the deleted positions
        list of int
            list of the labels of the newly added characters
        See Also
        --------
            shrink
            extend
        """
        deleted_positions = self.shrink(codec) if shrink else []
        added_positions = self.extend(codec) if extend else []
        return deleted_positions, added_positions


def ascii_codec():
    """default ascii codec

    Returns
    -------
    Codec
        codec based on the default ascii characters
    """
    ascii_labels = ["", " ", "~"] + [chr(x) for x in range(33, 126)]
    return Codec(ascii_labels)
