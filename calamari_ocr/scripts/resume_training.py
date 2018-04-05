from argparse import ArgumentParser

from calamari_ocr.utils import glob_all, split_all_ext
from calamari_ocr.ocr.dataset import FileDataSet
from calamari_ocr.ocr.trainer import Trainer

from calamari_ocr.proto import CheckpointParams

from google.protobuf import json_format

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The checkpoint used to resume")
    parser.add_argument("--validation", type=str, nargs="+",
                        help="Validation line files used for early stopping")
    parser.add_argument("files", type=str, nargs="+",
                        help="The files to use for training")

    args = parser.parse_args()

    # Train dataset
    input_image_files = glob_all(args.files)
    gt_txt_files = [split_all_ext(f)[0] + ".gt.txt" for f in input_image_files]

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = FileDataSet(input_image_files, gt_txt_files)

    print("Found {} files in the dataset".format(len(dataset)))

    # Validation dataset
    if args.validation:
        validation_image_files = glob_all(args.validation)
        val_txt_files = [split_all_ext(f)[0] + ".gt.txt" for f in validation_image_files]

        if len(set(val_txt_files)) != len(val_txt_files):
            raise Exception("Some validation images are occurring more than once in the data set.")

        validation_dataset = FileDataSet(validation_image_files, val_txt_files)
        print("Found {} files in the validation dataset".format(len(validation_dataset)))
    else:
        validation_dataset = None

    with open(args.checkpoint + '.json', 'r') as f:
        checkpoint_params = json_format.Parse(f.read(), CheckpointParams())

        trainer = Trainer(checkpoint_params, dataset,
                          validation_dataset=validation_dataset,
                          restore=args.checkpoint)
        trainer.train(progress_bar=True)


if __name__ == "__main__":
    main()