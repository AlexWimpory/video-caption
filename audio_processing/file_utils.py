import os
from os.path import dirname, basename, splitext, join
import pickle

"""
Methods for:
* Searching directories
* Changing file names
* Saving and loading objects with pickle
"""


def apply_to_path(f, path_name, extension):
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                f(os.path.join(root, file))


def return_from_path(f, path_name, extension):
    results = []
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                try:
                    result = f(os.path.join(root, file))
                    results.append(result)
                except BaseException as err:
                    print(f'Unable to process {file} in {root}')
                    print(err)
    return results


def append_to_file_name(path, new_str, extension):
    dir_name = dirname(path)
    file_name = basename(path)
    split = splitext(file_name)
    if not extension:
        extension = split[1]
    return join(dir_name, split[0] + new_str + extension)


def save_object(obj, path):
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)


def load_object(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


if __name__ == '__main__':
    apply_to_path(print, '../audio_speech', '.wav')
