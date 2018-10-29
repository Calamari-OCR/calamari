from argparse import ArgumentParser
import os

from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
from calamari_ocr.ocr import DataSetType, create_dataset, DataSetMode
from calamari_ocr.ocr.trainer import Trainer

from calamari_ocr.proto import CheckpointParams

from google.protobuf import json_format


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The checkpoint used to resume")

    # validation files
    parser.add_argument("--validation", type=str, nargs="+",
                        help="Validation line files used for early stopping")
    parser.add_argument("--validation_text_files", nargs="+", default=None,
                        help="Optional list of validation GT files if they are in other directory")
    parser.add_argument("--validation_extension", default=None,
                        help="Default extension of the gt files (expected to exist in same dir)")
    parser.add_argument("--validation_dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)

    # input files
    parser.add_argument("--files", nargs="+",
                        help="List all image files that shall be processed. Ground truth fils with the same "
                             "base name but with '.gt.txt' as extension are required at the same location")
    parser.add_argument("--text_files", nargs="+", default=None,
                        help="Optional list of GT files if they are in other directory")
    parser.add_argument("--gt_extension", default=None,
                        help="Default extension of the gt files (expected to exist in same dir)")
    parser.add_argument("--dataset", type=DataSetType.from_string, choices=list(DataSetType), default=DataSetType.FILE)
    parser.add_argument("--no_skip_invalid_gt", action="store_true",
                        help="Do no skip invalid gt, instead raise an exception.")

    args = parser.parse_args()

    if args.gt_extension is None:
        args.gt_extension = DataSetType.gt_extension(args.dataset)

    if args.validation_extension is None:
        args.validation_extension = DataSetType.gt_extension(args.validation_dataset)

    # Training dataset
    print("Resolving input files")
    input_image_files = sorted(glob_all(args.files))
    if not args.text_files:
        gt_txt_files = [split_all_ext(f)[0] + args.gt_extension for f in input_image_files]
    else:
        gt_txt_files = sorted(glob_all(args.text_files))
        input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
        for img, gt in zip(input_image_files, gt_txt_files):
            if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = create_dataset(
        args.dataset,
        DataSetMode.TRAIN,
        images=input_image_files,
        texts=gt_txt_files,
        skip_invalid=not args.no_skip_invalid_gt
    )
    print("Found {} files in the dataset".format(len(dataset)))

    # Validation dataset
    if args.validation:
        print("Resolving validation files")
        validation_image_files = glob_all(args.validation)
        if not args.validation_text_files:
            val_txt_files = [split_all_ext(f)[0] + args.validation_extension for f in validation_image_files]
        else:
            val_txt_files = sorted(glob_all(args.validation_text_files))
            validation_image_files, val_txt_files = keep_files_with_same_file_name(validation_image_files, val_txt_files)
            for img, gt in zip(validation_image_files, val_txt_files):
                if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                    raise Exception("Expected identical basenames of validation file: {} and {}".format(img, gt))

        if len(set(val_txt_files)) != len(val_txt_files):
            raise Exception("Some validation images are occurring more than once in the data set.")

        validation_dataset = create_dataset(
            args.validation_dataset,
            DataSetMode.TRAIN,
            images=validation_image_files,
            texts=val_txt_files,
            skip_invalid=not args.no_skip_invalid_gt)
        print("Found {} files in the validation dataset".format(len(validation_dataset)))
    else:
        validation_dataset = None

    print("Resuming training")
    with open(args.checkpoint + '.json', 'r') as f:
        checkpoint_params = json_format.Parse(f.read(), CheckpointParams())

        trainer = Trainer(checkpoint_params, dataset,
                          validation_dataset=validation_dataset,
                          weights=args.checkpoint)
        trainer.train(progress_bar=True)


if __name__ == "__main__":
    main()