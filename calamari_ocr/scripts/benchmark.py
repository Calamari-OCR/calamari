import argparse
import tempfile
import time
import numpy as np
from prettytable import PrettyTable

from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams

import multiprocessing


def benchmark_prediction(model, batch_size, processes, n_examples, runs=10):
    params = PredictorParams(silent=True)
    predictor = Predictor.from_checkpoint(params, model)

    data = (np.random.random((400, 48)) * 255).astype(np.uint8)
    print("Running with bs={}, proc={}, n={}".format(batch_size, processes, n_examples))
    start = time.time()
    for i in range(runs):
        list(predictor.predict_raw([data] * n_examples))
    end = time.time()

    return (end - start) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", default=tempfile.gettempdir(), type=str)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processes", default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    batch_sizes = [1, 5, 10, 20, 50]
    tab = PrettyTable(["n"] + list(map(str, batch_sizes)))
    for n_examples in [1, 10, 100, 1000]:
        results = [benchmark_prediction(args.model, bs, args.processes, n_examples) for bs in batch_sizes]
        tab.add_row([n_examples] + results)

    print(tab)


if __name__ == "__main__":
    main()
