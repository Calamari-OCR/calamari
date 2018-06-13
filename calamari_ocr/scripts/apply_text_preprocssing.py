import argparse
import codecs
from calamari_ocr.utils import glob_all
from tqdm import tqdm

from calamari_ocr.proto import TextProcessorParams
from calamari_ocr.ocr.text_processing import default_text_normalizer_params, default_text_regularizer_params, text_processor_from_proto

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True,
                        help="Text files to apply text processing")
    parser.add_argument("--text_regularization", type=str, nargs="+", default=["extended"],
                        help="Text regularization to apply.")
    parser.add_argument("--text_normalization", type=str, default="NFC",
                        help="Unicode text normalization to apply. Defaults to NFC")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="No not overwrite files, just run")

    args = parser.parse_args()

    # Text pre processing (reading)
    preproc = TextProcessorParams()
    preproc.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(preproc.children.add(), default=args.text_normalization)
    default_text_regularizer_params(preproc.children.add(), groups=args.text_regularization)
    strip_processor_params = preproc.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    txt_proc = text_processor_from_proto(preproc, "pre")

    print("Resolving files")
    text_files = glob_all(args.files)

    for path in tqdm(text_files, desc="Processing", total=len(text_files)):
        with codecs.open(path, "r", "utf-8") as f:
            content = f.read()

        content = txt_proc.apply(content)

        if args.verbose:
            print(content)

        if not args.dry_run:
            with codecs.open(path, "w", "utf-8") as f:
                f.write(content)


if __name__ == "__main__":
    main()
