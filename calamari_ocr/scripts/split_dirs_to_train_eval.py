import argparse
import os
import random
import sys


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_sub_out", type=str, default="train")
    parser.add_argument("--eval_sub_out", type=str, default="eval")
    parser.add_argument("--train_amount", type=float, required=True,
                        help="If >= 1 this value is interpreted as absolute value, else as relative value")
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()

    if args.seed > 0:
        random.seed(args.seed)

    all_dirs_ = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]

    if len(all_dirs_) == 0:
        raise Exception("No directories found at '{}'".format(args.base_dir))

    train_dir = os.path.join(args.output_dir, args.train_sub_out)
    eval_dir = os.path.join(args.output_dir, args.eval_sub_out)

    mkdir(train_dir)
    mkdir(eval_dir)

    n_train = int(args.train_amount) if args.train_amount >= 1 else int(len(all_dirs_) * args.train_amount)
    n_eval = len(all_dirs_) - n_train

    indices = list(range(len(all_dirs_)))
    random.shuffle(indices)

    train_dirs_ = [all_dirs_[i] for i in indices[:n_train]]
    eval_dirs_ = [all_dirs_[i] for i in indices[n_train:]]

    def make_lns(dirs_, out_dir):
        for d_ in dirs_:
            os.symlink(os.path.join(args.base_dir, d_), os.path.join(out_dir, d_), target_is_directory=True)

    for td, od in zip([train_dirs_, eval_dirs_], [train_dir, eval_dir]):
        print("Processing '{}' to '{}'".format(td, od))
        make_lns(td, od)


if __name__ == "__main__":
    main()

