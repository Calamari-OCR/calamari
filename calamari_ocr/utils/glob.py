import glob
import os


def glob_all(paths):
    if isinstance(paths, list):
        out = []
        for p in paths:
            out += glob.glob(os.path.expanduser(p))

        return out
    else:
        return glob.glob(paths)