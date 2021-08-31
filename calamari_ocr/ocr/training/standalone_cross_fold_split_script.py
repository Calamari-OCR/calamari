import argparse
import json

from calamari_ocr.ocr import CrossFold

if __name__ == "__main__":
    # Standalone script to run the cross fold splitting in a separate thread
    # this script is called from cross_fold_trianer.py
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--progress_bar", action="store_true")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cross_fold = CrossFold.from_dict(cfg)
    cross_fold.create_folds(progress_bar=args.progress_bar)
    cross_fold.write_folds_to_json(args.dir)
    with open(args.config, "w") as f:
        json.dump(cross_fold.to_dict(), f)
