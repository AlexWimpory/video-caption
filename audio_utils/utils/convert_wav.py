import subprocess
import os

"""Convert all .flac files in a path to .wav"""


def apply_to_path(f, path_name, extension):
    for root, dirs, files in os.walk(path_name):
        for file in files:
            if file.endswith(extension):
                f(os.path.join(root, file))


def convert_flac(filepath):
    subprocess.call(['ffmpeg', '-i', filepath, filepath.replace('.flac', '.wav')])


if __name__ == '__main__':
    apply_to_path(convert_flac, 'D:\\Audio Speech\\LibriSpeech\\train-clean-100', '.flac')
