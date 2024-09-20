import os

def make_path(*args):
    path = os.path.join(*args)
    if not(os.path.isdir(path)):
        os.makedirs(path)