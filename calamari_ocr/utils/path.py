import os


def split_all_ext(path):
    path, basename = os.path.split(path)
    pos = basename.find(".")
    return os.path.join(path, basename[:pos]), basename[pos:]


def checkpoint_path(path):
    i = path.rfind(".ckpt")
    if i < 0:
        raise FileNotFoundError("File path '{}' does not contain ckpt.".format(path))

    return path[:i + 5]


if __name__=="__main__":
    assert(checkpoint_path("model.ckpt") == "model.ckpt")
    assert(checkpoint_path("model.ckpt.json") == "model.ckpt")
