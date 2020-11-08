def convert_codec(codec: dict):
    raise NotImplementedError

def convert_text_processor(proc: dict):
    t = proc.get('type', 'DEFAULT_NORMALIZER')
    raise NotImplementedError


def convert_layer(layer: dict):
    mapping = {
        "CONVOLUTIONAL": 'convolutional',
        'MAX_POOLING': 'max_pooling',
        'LSTM': 'lstm',
        'TRANSPOSED_CONVOLUTIONAL': 'transposed_conv',
        'DILATED_BLOCK': 'dilated_block',
        'CONCAT': 'concat'
    }
    return {
        'type': mapping[layer.get('type', 'CONVOLUTIONAL')],
        'filters': layer.get('filters', 0),
        'kernel_size': layer.get('kernelSize', {'x': 0, 'y': 0}),
        'stride': layer.get('stride', {'x': 0, 'y': 0}),
        'dilated_depth': layer.get('dilatedDepth', 0),
        'concat_indices': layer.get('concatIndices', []),
        'hidden_nodes': layer.get('hiddenNodes', 0),
        'peepholes': layer.get('peepholes', False),
        'lstm_direction': 'bidirectional',
    }


def migrate(d: dict):
    solver_mapper = {
        "ADAM_SOLVER": "Adam",
        "MOMENTUM_SOLVER": "SGD",
    }
    model = d.get('model', {})
    network = model.get('network', {})
    backend = network.get('backend', {})
    codec = model.get('codec', {})
    data_preprocessor = model.get('dataPreprocessor', {})
    text_preprocessor = model.get('textPreprocessor', {})
    text_postprocessor = model.get('textPostprocessor', {})

    # migrate to dict based on tfaip
    out = {
        "epochs": 0,
        "current_epoch": 0,
        "samples_per_epoch": 1,
        "train_accum_steps": 1,
        "tf_cpp_min_log_level": 2,
        "force_eager": False,
        "skip_model_load_test": True,
        "test_every_n": 1,
        "lav_every_n": 0,
        "checkpoint_dir": "/home/cwick/Documents/Projects/calamari/asdf",
        "write_checkpoints": True,
        "export_best": True,
        "export_final": False,
        "no_train_scope": None,
        "calc_ema": False,
        "random_seed": backend.get('randomSeed', 0),
        "profile": False,
        "device_params": {
            "gpus": [],
            "gpu_auto_tune": False,
            "gpu_memory": None,
            "soft_device_placement": True,
            "dist_strategy": "default"
        },
        "optimizer_params": {
            "optimizer": solver_mapper[network.get('solver', "MOMENTUM_SOLVER")],
            "clip_grad": network.get('clippingNorm', 0),
            "momentum": network.get('momentum', 0),
            "rho": 0.0,
            "centered": False,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07
        },
        "learning_rate_params": {
            "type": "ExpDecay",
            "lr": network.get('learningRate', 0),
            "learning_circle": 3,
            "lr_decay_rate": 0.99,
            "decay_fraction": 0.1,
            "final_epochs": 50,
            "step_function": True,
            "warmup_epochs": 10,
            "warmup_factor": 10,
            "constant_epochs": 10,
            "steps_per_epoch_": 1,
            "epochs_": 5
        },
        "scenario_params": {
            "debug_graph_construction": False,
            "debug_graph_n_examples": 1,
            "print_eval_limit": 10,
            "export_frozen": False,
            "export_serve": True,
            "model_params": {
                "layers": [convert_layer(l) for l in network.get('layers', [])],
                "dropout": network.get('dropout', 0),
                "classes": network.get('classes', 0),
                "ctc_merge_repeated": network.get('ctcMergeRepeated', False),
            },
            "data_params": {
                "val_list": [
                    "DUMMY"
                ],
                "val_batch_size": d.get('batchSize', 0),
                "val_limit": -1,
                "val_prefetch": 32,
                "val_num_processes": d.get('processes', 0),
                "val_batch_drop_remainder": False,
                "train_lists": [
                    "DUMMY"
                ],
                "train_list_ratios": [
                    1.0
                ],
                "train_batch_size": d.get('batchSize', 0),
                "train_limit": -1,
                "train_prefetch": 32,
                "train_num_processes": d.get('processes', 0),
                "train_batch_drop_remainder": False,
                "lav_lists": None,
                "preproc_max_tasks_per_child": 250,
                "resource_base_path_": ".",
                "skip_invalid_gt_": d.get('skip_invalid_gt', False),
                "input_channels": 1,
                "downscale_factor_": -1,
                "line_height_": 48,
                "raw_dataset": False,
                "codec": convert_codec(codec),
                "text_processor": {
                    "type": "MultiTextProcessor",
                    "processors": [
                        {
                            "type": "TextNormalizer",
                            "unicode_normalization": "NFC"
                        },
                        {
                            "type": "TextRegularizer",
                            "replacements": [
                                {
                                    "regex": false,
                                    "old": "\u00b5",
                                    "new": "\u03bc"
                                },
                                {
                                    "regex": false,
                                    "old": "\u2013\u2014\u2014",
                                    "new": "-"
                                },
                                {
                                    "regex": false,
                                    "old": "\u2013\u2014",
                                    "new": "-"
                                },
                                {
                                    "regex": false,
                                        "old": "\"",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "`",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201d",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u00b4",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2018",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2019",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201d",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201e",
                                        "new": ",,"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2026",
                                        "new": "..."
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2032",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2033",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2034",
                                        "new": "'''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u3003",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2160",
                                        "new": "I"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2161",
                                        "new": "II"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2162",
                                        "new": "III"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2163",
                                        "new": "IV"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2164",
                                        "new": "V"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2165",
                                        "new": "VI"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2166",
                                        "new": "VII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2167",
                                        "new": "VIII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2168",
                                        "new": "IX"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2169",
                                        "new": "X"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216a",
                                        "new": "XI"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216b",
                                        "new": "XII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216c",
                                        "new": "L"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216d",
                                        "new": "C"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216e",
                                        "new": "D"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216f",
                                        "new": "M"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2170",
                                        "new": "i"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2171",
                                        "new": "ii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2172",
                                        "new": "iii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2173",
                                        "new": "iv"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2174",
                                        "new": "v"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2175",
                                        "new": "vi"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2176",
                                        "new": "vii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2177",
                                        "new": "viii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2178",
                                        "new": "ix"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2179",
                                        "new": "x"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217a",
                                        "new": "xi"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217b",
                                        "new": "xii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217c",
                                        "new": "l"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217d",
                                        "new": "c"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217e",
                                        "new": "d"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217f",
                                        "new": "m"
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\s+",
                                        "new": " "
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\n",
                                        "new": ""
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)^\\s+",
                                        "new": ""
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\s+$",
                                        "new": ""
                                    }
                                ]
                            },
                            {
                                "type": "StripTextProcessor"
                            }
                        ]
                    },
                    "text_post_processor": {
                        "type": "MultiTextProcessor",
                        "processors": [
                            {
                                "type": "TextNormalizer",
                                "unicode_normalization": "NFC"
                            },
                            {
                                "type": "TextRegularizer",
                                "replacements": [
                                    {
                                        "regex": false,
                                        "old": "\u00b5",
                                        "new": "\u03bc"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2013\u2014\u2014",
                                        "new": "-"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2013\u2014",
                                        "new": "-"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\"",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "`",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201d",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u00b4",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2018",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2019",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201d",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201c",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u201e",
                                        "new": ",,"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2026",
                                        "new": "..."
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2032",
                                        "new": "'"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2033",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2034",
                                        "new": "'''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u3003",
                                        "new": "''"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2160",
                                        "new": "I"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2161",
                                        "new": "II"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2162",
                                        "new": "III"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2163",
                                        "new": "IV"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2164",
                                        "new": "V"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2165",
                                        "new": "VI"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2166",
                                        "new": "VII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2167",
                                        "new": "VIII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2168",
                                        "new": "IX"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2169",
                                        "new": "X"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216a",
                                        "new": "XI"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216b",
                                        "new": "XII"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216c",
                                        "new": "L"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216d",
                                        "new": "C"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216e",
                                        "new": "D"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u216f",
                                        "new": "M"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2170",
                                        "new": "i"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2171",
                                        "new": "ii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2172",
                                        "new": "iii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2173",
                                        "new": "iv"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2174",
                                        "new": "v"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2175",
                                        "new": "vi"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2176",
                                        "new": "vii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2177",
                                        "new": "viii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2178",
                                        "new": "ix"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u2179",
                                        "new": "x"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217a",
                                        "new": "xi"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217b",
                                        "new": "xii"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217c",
                                        "new": "l"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217d",
                                        "new": "c"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217e",
                                        "new": "d"
                                    },
                                    {
                                        "regex": false,
                                        "old": "\u217f",
                                        "new": "m"
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\s+",
                                        "new": " "
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\n",
                                        "new": ""
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)^\\s+",
                                        "new": ""
                                    },
                                    {
                                        "regex": true,
                                        "old": "(?u)\\s+$",
                                        "new": ""
                                    }
                                ]
                            },
                            {
                                "type": "StripTextProcessor"
                            }
                        ]
                    },
                    "data_processor": {
                        "type": "MultiDataProcessor",
                        "processors": [
                            {
                                "type": "DataRangeNormalizer"
                            },
                            {
                                "type": "CenterNormalizer",
                                "line_height": 48,
                                "extra_params": [
                                    4,
                                    1.0,
                                    0.3
                                ],
                                "debug": false
                            },
                            {
                                "type": "FinalPreparation",
                                "normalize": true,
                                "invert": true,
                                "transpose": true,
                                "pad": 16,
                                "pad_value": 0,
                                "as_uint8": true
                            }
                        ]
                    },
                    "data_augmenter": {
                        "type": "SimpleDataAugmenter",
                        "params": {}
                    },
                    "data_aug_params": {
                        "reference": "relative",
                        "amount": 0,
                        "percentage": 0
                    },
                    "train_reader": {
                        "type": "FileDataReaderFactory",
                        "params": {
                            "files": [
                                "calamari_ocr/test/data/uw3_50lines/train/*.png"
                            ],
                            "text_files": null,
                            "data_set_type": 1,
                            "data_set_mode": 0,
                            "gt_extension": ".gt.txt",
                            "skip_invalid": true,
                            "data_reader_args": {
                                "line_generator_params": {
                                    "fonts": [],
                                    "font_size": 0,
                                    "min_script_offset": 0,
                                    "max_script_offset": 0
                                },
                                "text_generator_params": {
                                    "word_length_mean": 0,
                                    "word_length_sigma": 0,
                                    "charset": [],
                                    "super_charset": [],
                                    "sub_charset": [],
                                    "number_of_words_mean": 0,
                                    "number_of_words_sigma": 0,
                                    "word_separator": " ",
                                    "sub_script_p": 0,
                                    "super_script_p": 0,
                                    "bold_p": 0,
                                    "italic_p": 0,
                                    "letter_spacing_p": 0,
                                    "letter_spacing_mean": 0,
                                    "letter_spacing_sigma": 0
                                },
                                "pad": null,
                                "text_index": 0
                            }
                        }
                    },
                    "val_reader": null,
                    "predict_reader": null
                },
                "export_net_config_": false,
                "net_config_filename_": "net_config.json",
                "frozen_dir_": "frozen",
                "frozen_filename_": "frozen_model.pb",
                "default_serve_dir_": "best.ckpt.h5",
                "additional_serve_dir_": "additional",
                "trainer_params_filename_": "best.ckpt.json",
                "scenario_params_filename_": null,
                "scenario_base_path_": "/home/cwick/Documents/Projects/calamari/calamari_ocr/ocr/backends/scenario.py",
                "scenario_module_": "calamari_ocr.ocr.backends.scenario",
                "id_": "asdf_calamari_ocr.ocr.backends.scenario_2020-11-08",
                "tfaip_commit_hash_": "b234c8ce1428b33d6830a7a4a3d7bc13fedd69ba",
                "tfaip_version_": "1.0.0"
            },
            "warmstart_params": {
                "model": null,
                "allow_partial": false,
                "trim_graph_name": true,
                "rename": [],
                "exclude": null,
                "include": null
            },
            "early_stopping_params": {
                "mode_": "min",
                "current_": 1.0,
                "monitor_": "val_cer_metric",
                "n_": -1,
                "best_model_output_dir": "/home/cwick/Documents/Projects/calamari/asdf",
                "best_model_name": "",
                "frequency": 1,
                "n_to_go": 5,
                "lower_threshold": 0.0,
                "upper_threshold": 0.9
            },
            "saved_checkpoint_sub_dir_": "",
            "checkpoint_sub_dir_": "checkpoint/checkpoint_{epoch:04d}",
            "checkpoint_save_freq_": 2,
            "version": 2,
            "skip_invalid_gt": true,
            "stats_size": 100,
            "data_aug_retrain_on_original": true,
            "current_stage_": -1,
            "codec_whitelist": [],
            "keep_loaded_codec": false,
            "preload_training": true,
            "preload_validation": true,
            "auto_compute_codec": true,
            "progress_bar": true,
            "auto_upgrade_checkpoints": true
        }
