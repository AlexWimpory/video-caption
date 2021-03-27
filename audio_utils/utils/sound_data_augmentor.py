import os
from audio_utils.utils.ffmpeg_processor import run_ffmpeg, run_ffprobe
from utils.file_utils import split_path_base_and_extension


def change_bit_rate(file_name, target_path):
    _, name, extension = split_path_base_and_extension(file_name)
    sample_rate = run_ffprobe(file_name).get_sample_rate()
    for percent in [0.5, 0.6, 0.7, 0.8, 0.9]:
        new_path_name = os.path.join(target_path, str(int(percent * 100)))
        if not os.path.isdir(new_path_name):
            os.mkdir(new_path_name)
        new_file_name = os.path.join(new_path_name, name + extension)
        run_ffmpeg(f'ffmpeg -y -i "{file_name}" -ar {int(sample_rate * percent)} "{new_file_name}"')
