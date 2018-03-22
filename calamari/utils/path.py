import os


def split_all_ext(path):
    path, basename = os.path.split(path)
    pos = basename.find(".")
    return os.path.join(path, basename[:pos]), basename[pos:]