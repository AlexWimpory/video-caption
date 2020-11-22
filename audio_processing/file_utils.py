import os
from os.path import dirname, basename, splitext, join


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


if __name__ == '__main__':
    apply_to_path(print, '../audio_speech', '.wav')
