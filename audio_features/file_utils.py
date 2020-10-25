import glob


def apply_to_path(f, path_name):
    for filepath in glob.iglob(path_name):
        f(filepath)


if __name__ == '__main__':
    apply_to_path(print, './data/*.wav')
