import argparse
import tempfile
import time
import numpy as np
import os
from prettytable import PrettyTable

from calamari_ocr.ocr.predictor import Predictor

import multiprocessing


def benchmark_prediction(model, batch_size, processes, n_examples, runs=10):
    predictor = Predictor(checkpoint=model,
                          batch_size=batch_size,
                          processes=processes,
                          )

    data = (np.random.random((400, 48)) * 255).astype(np.uint8)
    print("Running with bs={}, proc={}, n={}".format(batch_size, processes, n_examples))
    # to warmup
    list(predictor.predict_raw([data] * n_examples))
    start = time.time()
    for i in range(runs):
        list(predictor.predict_raw([data] * n_examples, batch_size=batch_size))
    end = time.time()

    return (end - start) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", default=tempfile.gettempdir(), type=str)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processes", default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    batch_sizes = [1, 5, 10, 20, 50]
    tab = PrettyTable(['n'] + list(map(str, batch_sizes)))
    for n_examples in [1, 10, 100, 1000]:
        results = [benchmark_prediction(args.model, bs, args.processes, n_examples) for bs in batch_sizes]
        tab.add_row([n_examples] + results)

    print(tab)


if __name__ == "__main__":
    main()
