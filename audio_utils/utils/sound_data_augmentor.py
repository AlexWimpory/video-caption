import os
from functools import partial
from utils.ffmpeg_processor import run_ffmpeg, run_ffprobe
from ground_truth.ground_truth_processor import GroundtruthReader
from utils.file_utils import split_path_base_and_extension, apply_to_path


def change_bit_rate(file_name, target_path):
    _, name, extension = split_path_base_and_extension(file_name)
    sample_rate = run_ffprobe(file_name).get_sample_rate()
    for percent in [0.5, 0.6, 0.7, 0.8, 0.9]:
        new_path_name = os.path.join(target_path, str(int(percent * 100)))
        if not os.path.isdir(new_path_name):
            os.mkdir(new_path_name)
        new_file_name = os.path.join(new_path_name, name + extension)
        run_ffmpeg(f'ffmpeg -y -i "{file_name}" -ar {int(sample_rate * percent)} "{new_file_name}"')


def change_bit_rate_filter(target_path, gtp, filter_label, file_name):
    _, name, extension = split_path_base_and_extension(file_name)
    try:
        from_groundtruth = gtp.lookup_filename(name)
    except KeyError:
        print(name, file_name)
        return
    if filter_label is not None and filter_label not in from_groundtruth:
        print(f'skipping {file_name}')
        return
    sample_rate = run_ffprobe(file_name).get_sample_rate()
    for percent in [0.5, 0.6, 0.7, 0.8, 0.9]:
        new_path_name = os.path.join(target_path, str(int(percent * 100)))
        if not os.path.isdir(new_path_name):
            os.mkdir(new_path_name)
        new_file_name = os.path.join(new_path_name, name + extension)
        run_ffmpeg(f'ffmpeg -y -i "{file_name}" -ar {int(sample_rate * percent)} "{new_file_name}"')


if __name__ == '__main__':
    ground_truth_reader = GroundtruthReader('../ground_truth/data/fsd50k_dev_groundtruth.csv')
    prepare_audio_sound_groundtruth = partial(change_bit_rate_filter,
                                              'D:\\Audio Features\\new_4', ground_truth_reader, 'Bell')
    apply_to_path(prepare_audio_sound_groundtruth,
                  'D:\Audio Features\FSD50K\FSD50K.dev_audio',
                  '.wav')
