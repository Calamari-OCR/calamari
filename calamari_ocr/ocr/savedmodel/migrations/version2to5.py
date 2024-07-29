import inspect
import os
from typing import Optional
import logging

import h5py
from tensorflow import keras
from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr.savedmodel.migrations.version3_4to5 import migrate3to5
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import TrainerParams

logger = logging.getLogger(__name__)


def update_model(params: dict, path: str):
    logger.info(f"Updating model at {path}")
    trainer_params = TrainerParams.from_dict(params)
    post_init(trainer_params)
    scenario_params = trainer_params.scenario
    scenario = CalamariScenario(scenario_params)
    inputs = scenario.data.create_input_layers()
    outputs = scenario.graph.predict(inputs)
    pred_model = keras.models.Model(inputs, outputs)
    with h5py.File(path + ".h5", "r") as f:
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]
        graph = pred_model.layers[3]._layers[0]
        load_weights_from_hdf5_group(f, [l for l in graph.layer_instances if len(l.weights) > 0] + [graph.logits])

    logger.info(f"Writing converted model at {path}.tmp.h5")
    pred_model.save(path + ".tmp.h5", include_optimizer=False)
    logger.info(f"Attempting to load converted model at {path}.tmp.h5")
    keras.models.load_model(
        path + ".tmp.h5",
        custom_objects=CalamariScenario.model_cls().all_custom_objects(),
    )
    logger.info(f"Replacing old model at {path}.h5")
    os.remove(path + ".h5")
    os.rename(path + ".tmp.h5", path + ".h5")
    logger.info(f"New model successfully written")
    keras.backend.clear_session()


def convert_codec(codec: dict):
    return codec


def image_processor(name, args: Optional[dict] = None):
    return {
        "name": name,
        "modes": ["prediction", "training", "evaluation"],
        "args": args,
    }


def convert_image_processor(proc: dict):
    flat = []

    def recurse_convert(p: dict):
        t = p.get("type", "DEFAULT_NORMALIZER")
        if t == "DEFAULT_NORMALIZER":
            flat.extend(default_data_normalizer())
        elif t == "MULTI_NORMALIZER":
            for c in p["children"]:
                recurse_convert(c)
        elif t == "NOOP_NORMALIZER":
            pass
        elif t == "RANGE_NORMALIZER":
            flat.append(image_processor("DataRangeNormalizer"))
        elif t == "CENTER_NORMALIZER":
            flat.append(image_processor("CenterNormalizer"))
        elif t == "FINAL_PREPARATION":
            flat.append(
                image_processor(
                    "FinalPreparation",
                    args={
                        "normalize": not p.get("noNormalize", False),
                        "invert": not p.get("noInvert", False),
                        "transpose": not p.get("noTranspose", False),
                        "pad": p.get("pad", 0),
                        "pad_value": p.get("padValue", 0),
                        "as_uint8": True,
                    },
                )
            )
        elif t == "SCALE_TO_HEIGHT":
            flat.append(image_processor("ScaleToHeightProcessor"))
        else:
            raise ValueError(f"Unknown type {t}")

    recurse_convert(proc)
    return flat


def text_processor(name, args: Optional[dict] = None):
    return {
        "name": name,
        "modes": ["targets", "training", "evaluation"],
        "args": args,
    }


def convert_text_processor(proc: dict):
    flat = []

    def recurse_convert(p: dict):
        t = p.get("type", "DEFAULT_NORMALIZER")
        if t == "DEFAULT_NORMALIZER":
            flat.extend(default_text_preprocessor())
        elif t == "DEFAULT_PRE_NORMALIZER":
            flat.extend(default_text_preprocessor())
        elif t == "DEFAULT_POST_NORMALIZER":
            flat.extend(default_text_preprocessor())
        elif t == "MULTI_NORMALIZER":
            for c in p["children"]:
                recurse_convert(c)
        elif t == "NOOP_NORMALIZER":
            pass
        elif t == "STRIP_NORMALIZER":
            flat.append(text_processor("StripTextProcessor"))
        elif t == "BIDI_NORMALIZER":
            conv = {"BIDI_AUTO": None, "BIDI_LTR": "L", "BIDI_RTL": "R"}
            flat.append(
                text_processor(
                    "BidiTextProcessor",
                    args={"bidi_direction": conv[p.get("bidiDirection", "BIDI_AUTO")]},
                )
            )
        elif t == "TEXT_NORMALIZER":
            conv = ["NFC", "NFKC", "NFD", "NFKD"]
            value = p.get("unicodeNormalization", 0)
            if isinstance(value, int):
                value = conv[value]
            else:
                assert value in conv, f"{value} not in {conv}"
            flat.append(
                text_processor(
                    "TextNormalizer",
                    args={"unicode_normalization": value},
                )
            )
        elif t == "TEXT_REGULARIZER":
            flat.append(text_processor("TextRegularizer", args={"replacements": p.get("replacements", [])}))
        elif t == "STR_TO_CHAR_LIST":
            flat.append(text_processor("StrToCharList"), args={"chars": p.get("characters", [])})
        else:
            raise ValueError(f"Unknown type {t}")

    recurse_convert(proc)
    return flat


def convert_layer(layer: dict):
    mapping = {
        "CONVOLUTIONAL": "convolutional",
        "MAX_POOLING": "max_pooling",
        "LSTM": "lstm",
        "TRANSPOSED_CONVOLUTIONAL": "transposed_conv",
        "DILATED_BLOCK": "dilated_block",
        "CONCAT": "concat",
    }
    return {
        "type": mapping[layer.get("type", "CONVOLUTIONAL")],
        "filters": layer.get("filters", 0),
        "kernel_size": layer.get("kernelSize", {"x": 0, "y": 0}),
        "stride": layer.get("stride", {"x": 0, "y": 0}),
        "dilated_depth": layer.get("dilatedDepth", 0),
        "concat_indices": layer.get("concatIndices", []),
        "hidden_nodes": layer.get("hiddenNodes", 0),
        "peepholes": layer.get("peepholes", False),
        "lstm_direction": "bidirectional",
    }


def migrate2to5(d: dict):
    d = migrate2to3(d)
    d = migrate3to5(d)
    return d


def migrate2to3(d: dict):
    solver_mapper = {
        "ADAM_SOLVER": "Adam",
        "MOMENTUM_SOLVER": "SGD",
    }
    model = d.get("model", {})
    network = model.get("network", {})
    backend = network.get("backend", {})
    codec = model.get("codec", {})
    data_preprocessor = model.get("dataPreprocessor", {})
    text_preprocessor = model.get("textPreprocessor", {})
    text_postprocessor = model.get("textPostprocessor", {})

    converted_pre_processors = convert_image_processor(data_preprocessor) + convert_text_processor(text_preprocessor)

    converted_pre_processors.append(
        {
            "name": "PrepareSampleProcessor",
            "modes": ["prediction", "training", "evaluation"],
        }
    )

    converted_post_processors = convert_text_processor(text_postprocessor)
    converted_post_processors.insert(0, {"name": "CTCDecoderProcessor"})
    converted_post_processors.insert(0, {"name": "ReshapeOutputsProcessor"})

    # migrate to dict based on tfaip
    return {
        "random_seed": backend.get("randomSeed", 0),
        "optimizer_params": {
            "optimizer": solver_mapper[network.get("solver", "MOMENTUM_SOLVER")],
            "clip_grad": network.get("clippingNorm", 0),
            "momentum": network.get("momentum", 0),
        },
        "learning_rate_params": {
            "lr": network.get("learningRate", 0),
        },
        "scenario_params": {
            "scenario_base_path_": inspect.getfile(CalamariScenario),
            "scenario_module_": CalamariScenario.__module__,
            "model_params": {
                "layers": [convert_layer(l) for l in network.get("layers", [])],
                "dropout": network.get("dropout", 0),
                "classes": network.get("classes", 0),
                "ctc_merge_repeated": network.get("ctcMergeRepeated", False),
            },
            "data_params": {
                "preproc_max_tasks_per_child": 250,
                "resource_base_path_": ".",
                "skip_invalid_gt_": d.get("skipInvalidGt", False),
                "input_channels": model.get("channels", 1),
                "line_height_": model.get("lineHeight"),
                "codec": convert_codec(codec),
                "pre_processors_": {
                    "run_parallel": True,
                    "sample_processors": converted_pre_processors,
                },
                "post_processors_": {
                    "run_parallel": True,
                    "sample_processors": converted_post_processors,
                },
                "scenario_module_": "calamari_ocr.ocr.backends.scenario",
                "tfaip_commit_hash_": "b234c8ce1428b33d6830a7a4a3d7bc13fedd69ba",
                "tfaip_version_": "1.0.0",
            },
        },
    }


def default_data_normalizer():
    return [
        {
            "name": "DataRangeNormalizer",
            "modes": ["prediction", "training", "evaluation"],
            "args": {},
        },
        {
            "name": "CenterNormalizer",
            "modes": ["prediction", "training", "evaluation"],
            "args": {"extra_params": [4, 1.0, 0.3]},
        },
        {
            "name": "FinalPreparation",
            "modes": ["prediction", "training", "evaluation"],
            "args": {
                "normalize": True,
                "invert": True,
                "transpose": True,
                "pad": 16,
                "pad_value": False,
                "as_uint8": True,
            },
        },
        {
            "name": "AugmentationProcessor",
            "modes": ["training"],
            "args": {"augmenter_type": "simple"},
        },
    ]


def default_text_preprocessor():
    return [
        {
            "name": "TextNormalizer",
            "modes": ["targets", "training", "evaluation"],
            "args": {"unicode_normalization": "NFC"},
        },
        {
            "name": "TextRegularizer",
            "modes": ["targets", "training", "evaluation"],
            "args": {
                "replacements": [
                    {"regex": False, "old": "\u00b5", "new": "\u03bc"},
                    {"regex": False, "old": "\u2013\u2014\u2014", "new": "-"},
                    {"regex": False, "old": "\u2013\u2014", "new": "-"},
                    {"regex": False, "old": '"', "new": "''"},
                    {"regex": False, "old": "`", "new": "'"},
                    {"regex": False, "old": "\u201c", "new": "''"},
                    {"regex": False, "old": "\u201d", "new": "''"},
                    {"regex": False, "old": "\u00b4", "new": "'"},
                    {"regex": False, "old": "\u2018", "new": "'"},
                    {"regex": False, "old": "\u2019", "new": "'"},
                    {"regex": False, "old": "\u201c", "new": "''"},
                    {"regex": False, "old": "\u201d", "new": "''"},
                    {"regex": False, "old": "\u201c", "new": "''"},
                    {"regex": False, "old": "\u201e", "new": ",,"},
                    {"regex": False, "old": "\u2026", "new": "..."},
                    {"regex": False, "old": "\u2032", "new": "'"},
                    {"regex": False, "old": "\u2033", "new": "''"},
                    {"regex": False, "old": "\u2034", "new": "'''"},
                    {"regex": False, "old": "\u3003", "new": "''"},
                    {"regex": False, "old": "\u2160", "new": "I"},
                    {"regex": False, "old": "\u2161", "new": "II"},
                    {"regex": False, "old": "\u2162", "new": "III"},
                    {"regex": False, "old": "\u2163", "new": "IV"},
                    {"regex": False, "old": "\u2164", "new": "V"},
                    {"regex": False, "old": "\u2165", "new": "VI"},
                    {"regex": False, "old": "\u2166", "new": "VII"},
                    {"regex": False, "old": "\u2167", "new": "VIII"},
                    {"regex": False, "old": "\u2168", "new": "IX"},
                    {"regex": False, "old": "\u2169", "new": "X"},
                    {"regex": False, "old": "\u216a", "new": "XI"},
                    {"regex": False, "old": "\u216b", "new": "XII"},
                    {"regex": False, "old": "\u216c", "new": "L"},
                    {"regex": False, "old": "\u216d", "new": "C"},
                    {"regex": False, "old": "\u216e", "new": "D"},
                    {"regex": False, "old": "\u216f", "new": "M"},
                    {"regex": False, "old": "\u2170", "new": "i"},
                    {"regex": False, "old": "\u2171", "new": "ii"},
                    {"regex": False, "old": "\u2172", "new": "iii"},
                    {"regex": False, "old": "\u2173", "new": "iv"},
                    {"regex": False, "old": "\u2174", "new": "v"},
                    {"regex": False, "old": "\u2175", "new": "vi"},
                    {"regex": False, "old": "\u2176", "new": "vii"},
                    {"regex": False, "old": "\u2177", "new": "viii"},
                    {"regex": False, "old": "\u2178", "new": "ix"},
                    {"regex": False, "old": "\u2179", "new": "x"},
                    {"regex": False, "old": "\u217a", "new": "xi"},
                    {"regex": False, "old": "\u217b", "new": "xii"},
                    {"regex": False, "old": "\u217c", "new": "l"},
                    {"regex": False, "old": "\u217d", "new": "c"},
                    {"regex": False, "old": "\u217e", "new": "d"},
                    {"regex": False, "old": "\u217f", "new": "m"},
                    {"regex": True, "old": "(?u)\\s+", "new": " "},
                    {"regex": True, "old": "(?u)\\n", "new": ""},
                    {"regex": True, "old": "(?u)^\\s+", "new": ""},
                    {"regex": True, "old": "(?u)\\s+$", "new": ""},
                ]
            },
        },
        {
            "name": "StripTextProcessor",
            "modes": ["targets", "training", "evaluation"],
            "args": None,
        },
    ]
