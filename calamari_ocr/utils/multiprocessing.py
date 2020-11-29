import os
import time
import subprocess
import logging


logger = logging.getLogger(__name__)


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
        logger.info("Executing: {}".format(" ".join(command)))

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False, env=env)
    while True:
        line = process.stdout.readline().decode('utf-8')

        # check if process has finished
        if process.poll() is not None:
            break

        # check if output is present
        if line is None or len(line) == 0:
            time.sleep(0.1)
        else:
            yield line

    if process.returncode != 0:
        raise Exception("Error: Process finished with code {}".format(process.returncode))
