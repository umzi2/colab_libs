import time
from functools import wraps
import os
from typing import Iterator, Any


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__}: {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def scandir(dir_path: str, suffix: str | None = None, recursive: bool = False) -> Iterator[Any]:
    # https://github.com/neosr-project/neosr/blob/3638c169ca57d18828e8487aecee117b76645900/neosr/utils/misc.py

    if (suffix is not None) and not isinstance(suffix, str | tuple):
        msg = '"suffix" must be a string or tuple of strings'
        raise TypeError(msg)

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if suffix is None or entry.path.endswith(suffix):
                    yield entry.path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
