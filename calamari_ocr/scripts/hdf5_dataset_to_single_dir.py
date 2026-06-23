from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from PIL import Image
from tqdm import tqdm
import logging

import os
import argparse
from calamari_ocr.utils import glob_all
from tfaip.data.pipeline.definitions import PipelineMode

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Write contents of h5 dataset to flat file directory")
    parser.add_argument(
        "--files",
        nargs="+",
        help="List all hdf5 files that shall be processed.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the folds")

    args = parser.parse_args()
    
    files = glob_all(args.files)
    dr = Hdf5.from_dict({"files": files})
    dr.prepare_for_mode(PipelineMode.EVALUATION)
    gen = dr.create(PipelineMode.EVALUATION)
    
    if len(gen) == 0:
        raise Exception("Empty dataset")

    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for s in tqdm(gen.generate(), total=len(gen), desc="Sample"):
        sname = s.meta["id"].replace("/", "-")
        spath = os.path.join(output_dir, sname)
        with open(spath + ".gt.txt", "w") as f:
            f.write(s.targets)
        Image.fromarray(s.inputs).save(spath + ".png")

if __name__ == "__main__":
    main()

