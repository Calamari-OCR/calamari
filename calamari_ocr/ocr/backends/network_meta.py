import re


def default_network_meta():
    return {
        "inter_threads": 0,
        "intra_threads": 0,
        "ctc_merge_repeated": True,
        "use_peepholes": False,
        "dropout": False,
        "solver": "Adam",
        "ctc": "Default",
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "layers": [],
        "cudnn": True,
        "features": 40,                     # i. e. the line heigth
    }

