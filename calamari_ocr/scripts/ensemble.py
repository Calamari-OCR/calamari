import json
import os
from argparse import ArgumentParser
from typing import List

from tensorflow import keras

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.model.ensemblegraph import EnsembleGraph
from calamari_ocr.ocr.model.graph import Graph


def split(args):
    ckpt = SavedCalamariModel(args.model)
    keras_model = keras.models.load_model(
        ckpt.ckpt_path,
        custom_objects={
            "Graph": Graph,
            "EnsembleGraph": EnsembleGraph,
            "VoterGraph": EnsembleGraph,
        },
    )

    def extract_keras_model(i):
        inputs = keras_model.input
        outputs = keras_model.output
        assert isinstance(outputs, dict)
        assert isinstance(inputs, dict)
        names_to_extract = [
            "blank_last_logits",
            "blank_last_softmax",
            "softmax",
            "decoded",
            "out_len",
        ]
        split_outputs = {}
        for name in names_to_extract:
            src_name = f"{name}_{i}"
            if src_name not in outputs:
                return None
            split_outputs[name] = outputs[src_name]

        return keras.Model(inputs=inputs, outputs=split_outputs)

    split_models: List[keras.Model] = []
    print("Starting to split models")
    while True:
        model = extract_keras_model(len(split_models))
        if model is None:
            break
        split_models.append(model)

    print(f"Split model into {len(split_models)}.")
    print(f"Saving models to {ckpt.dirname}/{ckpt.basename}_split_(i).ckpt")

    with open(ckpt.json_path) as f:
        ckpt_dict = json.load(f)
        ckpt_dict["scenario_params"]["model_params"]["ensemble"] = -1
        ckpt_dict["scenario_params"]["data_params"]["ensemble_"] = -1

    for i, split_model in enumerate(split_models):
        path = os.path.join(ckpt.dirname, f"{ckpt.basename}_split_{i}.ckpt")
        with open(path + ".json", "w") as f:
            json.dump(ckpt_dict, f, indent=2)
        split_model.save(path)
        print(f"Saved {i + 1}/{len(split_models)}")


def main():
    parser = ArgumentParser()
    sub_parser = parser.add_subparsers(title="Program mode", required=True, dest="mode")
    split_parser = sub_parser.add_parser("split")
    split_parser.add_argument("model")

    args = parser.parse_args()

    if args.mode == "split":
        split(args)


if __name__ == "__main__":
    main()
