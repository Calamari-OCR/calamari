import glob
import os


def glob_all(paths):
    if isinstance(paths, list):
        out = []
        for p in paths:
            if p.endswith(".files"):
                with open(p, 'r') as f:
                    for line in f:
                        out += glob.glob(line)
            else:
                out += glob.glob(os.path.expanduser(p))

        return out
    else:
        return glob_all([paths])
