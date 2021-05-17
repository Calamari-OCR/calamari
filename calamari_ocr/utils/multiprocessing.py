import os
import time
import subprocess
import logging
import sys
from threading import Thread
from queue import Queue, Empty


ON_POSIX = "posix" in sys.builtin_module_names


logger = logging.getLogger(__name__)


def prefix_run_command(command, prefix, args):
    if type(command) is not list and type(command) is not tuple:
        raise Exception("The command must be a list or tuple of commands and arguments")

    if prefix:
        prefix = prefix.format(args).split()
    else:
        prefix = []

    return prefix + command


def enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


def run(command, verbose=False):
    if type(command) is not list and type(command) is not tuple:
        raise Exception("The command must be a list or tuple of commands and arguments")

    if verbose:
        logger.info("Executing: {}".format(" ".join(command)))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=env,
        close_fds=ON_POSIX,
        text=True,
        bufsize=1,
        encoding="utf-8",
    )
    # Make nonblocking output
    stdout_queue = Queue()
    stdout_reader = Thread(target=enqueue_output, args=(process.stdout, stdout_queue), daemon=True)
    stdout_reader.start()
    stderr_queue = Queue()
    stderr_reader = Thread(target=enqueue_output, args=(process.stderr, stderr_queue), daemon=True)
    stderr_reader.start()
    while True:
        try:
            out = stdout_queue.get_nowait()
        except Empty:
            out = None

        try:
            err = stderr_queue.get_nowait()
        except Empty:
            err = None

        # check if process has finished
        if process.poll() is not None:
            break

        # check if output is present
        if (out is None or len(out) == 0) and (err is None or len(err) == 0):
            time.sleep(0.1)
        else:
            yield out, err

    if process.returncode != 0:
        raise Exception("Error: Process finished with code {}".format(process.returncode))
