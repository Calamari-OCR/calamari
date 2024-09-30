import os

# cannot use importlib.resources until we move to 3.9+ forimportlib.resources.files
import sys

if sys.version_info < (3, 10):
    import importlib_resources
else:
    import importlib.resources as importlib_resources
from pathlib import Path
import atexit
from contextlib import ExitStack


def keep_files_with_same_file_name(files1, files2):
    valid_files = set(map(filename, files1)).intersection(map(filename, files2))
    return [f for f in files1 if filename(f) in valid_files], [f for f in files2 if filename(f) in valid_files]


def filename(file) -> str:
    return split_all_ext(os.path.basename(file))[0]


def split_all_ext(path):
    path, basename = os.path.split(path)
    pos = basename.find(".")
    if pos == -1:
        return os.path.join(path, basename), ""
    return os.path.join(path, basename[:pos]), basename[pos:]


def checkpoint_path(path):
    i = path.rfind(".ckpt")
    if i < 0:
        raise FileNotFoundError("File path '{}' does not contain ckpt.".format(path))

    return path[: i + 5]


file_manager = ExitStack()
atexit.register(file_manager.close)


def resource_filename(pkg: str, fname: str) -> Path:
    ref = importlib_resources.files(pkg) / fname
    return file_manager.enter_context(importlib_resources.as_file(ref))


if __name__ == "__main__":
    assert checkpoint_path("model.ckpt") == "model.ckpt"
    assert checkpoint_path("model.ckpt.json") == "model.ckpt"
