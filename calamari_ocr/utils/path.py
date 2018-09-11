import os


def keep_files_with_same_file_name(files1, files2):
    valid_files = set(map(filename, files1)).intersection(map(filename, files2))
    return [f for f in files1 if filename(f) in valid_files], [f for f in files2 if filename(f) in valid_files]


def filename(file) -> str:
    return split_all_ext(os.path.basename(file))[0]


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
