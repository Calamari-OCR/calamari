import argparse
from tqdm import tqdm

from google.protobuf.json_format import MessageToJson

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.proto import Predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, default=[], nargs="+", required=True,
                        help="Protobuf files to convert")
    parser.add_argument("--logits", action="store_true",
                        help="Do write logits")
    args = parser.parse_args()

    files = glob_all(args.files)
    for file in tqdm(files, desc="Converting"):
        predictions = Predictions()
        with open(file, 'rb') as f:
            predictions.ParseFromString(f.read())

        if not args.logits:
            for prediction in predictions.predictions:
                prediction.logits.rows = 0
                prediction.logits.cols = 0
                prediction.logits.data[:] = []

        out_json_path = split_all_ext(file)[0] + ".json"
        with open(out_json_path, 'w') as f:
            f.write(MessageToJson(predictions, including_default_value_fields=True))


if __name__ == "__main__":
    main()
