import multiprocessing
import os
import time
import subprocess
from tqdm import tqdm


def parallel_map(f, d, desc="", processes=1, progress_bar=False):
    if processes <= 0:
        processes = os.cpu_count()

    if processes == 1:
        if progress_bar:
            out = list(tqdm(map(f, d), desc=desc, total=len(d)))
        else:
            out = list(map(f, d))

    else:
        with multiprocessing.Pool(processes=processes) as pool:
            if progress_bar:
                out = list(tqdm(pool.imap(f, d), desc=desc, total=len(d)))
            else:
                out = pool.map(f, d)

    return out


def prefix_run_command(command, prefix, args):
    if type(command) is not list and type(command) is not tuple:
        raise Exception("The command must be a list or tuple of commands and arguments")

    if prefix:
        prefix = prefix.format(args).split()
    else:
        prefix = []

    return prefix + command


def run(command, verbose=False):
    if type(command) is not list and type(command) is not tuple:
        raise Exception("The command must be a list or tuple of commands and arguments")

    if verbose:
        print("Excecuting: {}".format(" ".join(command)))

    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    while True:
        line = process.stdout.readline().rstrip().decode("utf-8")

        # check if process has finished
        if process.poll() is not None:
            break

        # check if output is present
        if line is None or len(line) == 0:
            time.sleep(0.1)
        else:
            yield line