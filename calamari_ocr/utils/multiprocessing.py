import multiprocessing
import os
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
