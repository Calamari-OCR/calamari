import glob
import os


def glob_all(paths, resolve_files_suffix=".files", include_base_dir=True, apply_glob_on_resolved=True):
    if isinstance(paths, str):
        return glob_all(
            [paths],
            resolve_files_suffix=resolve_files_suffix,
            include_base_dir=include_base_dir,
            apply_glob_on_resolved=apply_glob_on_resolved,
        )
    else:
        out = []
        for p in paths:
            if p.endswith(resolve_files_suffix):
                basedir = os.path.dirname(p)
                with open(p, "r") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if include_base_dir:
                            line = os.path.join(basedir, line)
                        if apply_glob_on_resolved:
                            out += glob_all(
                                [line],
                                resolve_files_suffix=resolve_files_suffix,
                                include_base_dir=include_base_dir,
                                apply_glob_on_resolved=apply_glob_on_resolved,
                            )
                        else:
                            out.append(line)
            else:
                out += glob.glob(os.path.expanduser(p))

        return out
