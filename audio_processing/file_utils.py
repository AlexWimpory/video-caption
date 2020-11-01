import os


def apply_to_path(f, path_name, extension):
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                f(os.path.join(root, file))


if __name__ == '__main__':
    apply_to_path(print, '../audio_speech', '.wav')
