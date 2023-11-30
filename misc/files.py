import os


def list_files(dir, ext=None):
    if ext is None:
        return [os.path.join(dir, f) for f in os.listdir(dir)]
    return [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext)]
