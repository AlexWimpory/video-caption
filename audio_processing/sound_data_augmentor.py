import csv
import os
import pandas as pd
from pandas import DataFrame
from ffmpeg_processor import run_ffmpeg, run_ffprobe
from file_utils import apply_to_target_path, split_path_base_and_extension, apply_to_path, split_base_and_extension, \
    load_object
from ground_truth_processor import GroundtruthReader
from model_predictor import ModelPredictor
from functools import partial
import matplotlib.pyplot as plt


def change_bit_rate(file_name, target_path):
    _, name, extension = split_path_base_and_extension(file_name)
    sample_rate = run_ffprobe(file_name).get_sample_rate()
    for percent in [0.5, 0.6, 0.7, 0.8, 0.9]:
        new_path_name = os.path.join(target_path, str(int(percent * 100)))
        if not os.path.isdir(new_path_name):
            os.mkdir(new_path_name)
        new_file_name = os.path.join(new_path_name, name + extension)
        run_ffmpeg(f'ffmpeg -y -i "{file_name}" -ar {int(sample_rate * percent)} "{new_file_name}"')


def find_incorrect_prediction_sample_rate(predictor, gtp, writer, path):
    file_name, extension = split_base_and_extension(path)
    sample_rate = run_ffprobe(path).get_sample_rate()
    percent = 100
    while percent > 0:
        run_ffmpeg(f'ffmpeg -y -i "{path}" -ar {int(sample_rate * percent/100)} temp_test.wav')
        prediction = [predictor.predict('temp_test.wav').predicted_class]
        actual = gtp.lookup_filename(file_name)
        if prediction != actual:
            writer.writerow([file_name, sample_rate, percent/100, sample_rate * (percent/100)])
            print(f'Processed {file_name} with a break percentage of {percent/100}')
            break
        else:
            percent -= 5


def sample_rate_test():
    with open('data/bit_rate_results10.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        predictor = ModelPredictor(model_name='model_1')
        gtp = GroundtruthReader('../audio_sounds/data/UrbanSound8K_groundtruth.csv')
        find_incorrect_prediction_sample_rate_with_objects = partial(find_incorrect_prediction_sample_rate, predictor, gtp, writer)
        apply_to_path(find_incorrect_prediction_sample_rate_with_objects, 'D:/Audio Features/UrbanSound8K/UrbanSound8K/audio/fold10', '.wav')


def plot_sample_rate_test_results():
    results = DataFrame()
    header_list = ['file_name', 'start_sample_rate', 'break_percentage', 'break_sample_rate']
    for i in range(1, 11):
        results = results.append(pd.read_csv(f'data/bit_rate_results{i}.csv', names=header_list))
    grouped = results.drop(['file_name', 'break_sample_rate'], axis=1).groupby(['break_percentage', 'start_sample_rate'])
    grouped.size().unstack().plot(kind='bar', stacked=True)
    plt.show()


if __name__ == '__main__':
    # apply_to_target_path(change_bit_rate, 'D:/Audio Features/UrbanSound8K/UrbanSound8K/audio/fold11',
    #                      '.wav', 'D:/Audio Features/new/fold11')

    # sample_rate_test()
    plot_sample_rate_test_results()
