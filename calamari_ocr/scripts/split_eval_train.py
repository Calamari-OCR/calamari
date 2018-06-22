import argparse
import shutil
from tqdm import tqdm
import os

from calamari_ocr.utils import glob_all, split_all_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                        help="All img files, an appropriate .gt.txt must exist")
    parser.add_argument("--n_eval", type=float, required=True,
                        help="The (relative or absolute) count of training files (or -1 to use the remaining)")
    parser.add_argument("--n_train", type=float, required=True,
                        help="The (relative or absolute) count of training files (or -1 to use the remaining)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write the splits")
    parser.add_argument("--eval_sub_dir", type=str, default="eval")
    parser.add_argument("--train_sub_dir", type=str, default="train")

    args = parser.parse_args()

    img_files = sorted(glob_all(args.files))
    if len(img_files) == 0:
        raise Exception("No files were found")

    gt_txt_files = [split_all_ext(p)[0] + ".gt.txt" for p in img_files]

    if args.n_eval < 0:
        pass
    elif args.n_eval < 1:
        args.n_eval = int(args.n_eval) * len(img_files)
    else:
        args.n_eval = int(args.n_eval)

    if args.n_train < 0:
        pass
    elif args.n_train < 1:
        args.n_train = int(args.n_train) * len(img_files)
    else:
        args.n_train = int(args.n_train)

    if args.n_eval < 0 and args.n_train < 0:
        raise Exception("Either n_eval or n_train may be < 0")

    if args.n_eval < 0:
        args.n_eval = len(img_files) - args.n_train
    elif args.n_train < 0:
        args.n_train = len(img_files) - args.n_eval

    if args.n_eval + args.n_train > len(img_files):
        raise Exception("Got {} eval and {} train files = {} in total, but only {} files are in the dataset".format(
            args.n_eval, args.n_train, args.n_eval + args.n_train, len(img_files)
        ))

    def copy_files(imgs, txts, out_dir):
        assert(len(imgs) == len(txts))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for img, txt in tqdm(zip(imgs, txts), total=len(imgs), desc="Writing to {}".format(out_dir)):
            if not os.path.exists(img):
                print("Image file at {} not found".format(img))
                continue

            if not os.path.exists(txt):
                print("Ground truth file at {} not found".format(txt))
                continue

            shutil.copyfile(img, os.path.join(out_dir, os.path.basename(img)))
            shutil.copyfile(txt, os.path.join(out_dir, os.path.basename(txt)))

    copy_files(img_files[:args.n_eval], gt_txt_files[:args.n_eval], os.path.join(args.output_dir, args.eval_sub_dir))
    copy_files(img_files[args.n_eval:], gt_txt_files[args.n_eval:], os.path.join(args.output_dir, args.train_sub_dir))



if __name__ == "__main__":
    main()
