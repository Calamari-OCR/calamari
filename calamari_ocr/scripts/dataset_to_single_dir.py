import argparse
import os
import shutil
import skimage.io as skimage_io
from tqdm import tqdm

from calamari_ocr.utils import glob_all, split_all_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=str, required=True,
                        help="The image files to copy")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="")
    parser.add_argument("--index_files", action="store_true")
    parser.add_argument("--convert_images", type=str,
                        help="Convert the image to a given type (by default use original format). E. g. jpg, png, tif, ...")
    parser.add_argument("--gt_ext", type=str, default=".gt.txt")
    parser.add_argument("--index_ext", type=str, default=".index")

    args = parser.parse_args()

    if args.convert_images and not args.convert_images.startswith("."):
        args.convert_images = "." + args.convert_images

    args.target_dir = os.path.expanduser(args.target_dir)

    print("Resolving files")
    image_files = glob_all(args.files)
    gt_files = [split_all_ext(p)[0] + ".gt.txt" for p in image_files]

    if len(image_files) == 0:
        raise Exception("No files found")

    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    for i, (img, gt) in tqdm(enumerate(zip(image_files, gt_files)), total=len(gt_files), desc="Copying"):
        if not os.path.exists(img) or not os.path.exists(gt):
            # skip non existing examples
            continue

        # img with optional convert
        try:
            ext = split_all_ext(img)[1]
            target_ext = args.convert_images if args.convert_images else ext
            target_name = os.path.join(args.target_dir, "{:08}{}".format(i, target_ext))
            if ext == target_ext:
                shutil.copyfile(img, target_name)
            else:
                data = skimage_io.imread(img)
                skimage_io.imsave(target_name, data)

        except:
            continue

        # gt txt
        target_name = os.path.join(args.target_dir, "{:08}{}".format(i, args.gt_ext))
        shutil.copyfile(gt, target_name)

        if args.index_files:
            target_name = os.path.join(args.target_dir, "{:08}{}".format(i, args.index_ext))
            with open(target_name, "w") as f:
                f.write(str(i))


if __name__ == "__main__":
    main()


