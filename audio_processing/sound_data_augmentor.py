import os
from ffmpeg_processor import run_ffmpeg, run_ffprobe
from file_utils import apply_to_target_path, split_path_base_and_extension


def change_bit_rate(file_name, target_path):
    _, name, extension = split_path_base_and_extension(file_name)
    sample_rate = run_ffprobe(file_name).get_sample_rate()
    for percent in [0.5, 0.6, 0.7, 0.8, 0.9]:
        new_path_name = os.path.join(target_path, str(int(percent * 100)))
        if not os.path.isdir(new_path_name):
            os.mkdir(new_path_name)
        new_file_name = os.path.join(new_path_name, name + extension)
        run_ffmpeg(f'ffmpeg -y -i "{file_name}" -ar {int(sample_rate * percent)} "{new_file_name}"')


if __name__ == '__main__':
    apply_to_target_path(change_bit_rate, 'D:/Audio Features/UrbanSound8K/UrbanSound8K/audio/fold11', '.wav', 'D:/Audio Features/new/fold11')
