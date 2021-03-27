from ground_truth.ground_truth_processor import GroundtruthReader
from functools import partial
import matplotlib.pyplot as plt
from audio_utils.utils.ffmpeg_processor import run_ffmpeg, run_ffprobe
from sounds.model_predictor import ModelPredictor
from utils.file_utils import split_base_and_extension, apply_to_path
import pandas as pd
from pandas import DataFrame
import csv


def find_incorrect_prediction_sample_rate(predictor, gtp, writer, path):
    file_name, extension = split_base_and_extension(path)
    sample_rate = run_ffprobe(path).get_sample_rate()
    percent = 100
    while percent > 0:
        run_ffmpeg(f'ffmpeg -y -i "{path}" -ar {int(sample_rate * percent / 100)} temp_test.wav')
        prediction = [predictor.predict('temp_test.wav').predicted_class]
        actual = gtp.lookup_filename(file_name)
        if prediction != actual:
            writer.writerow([file_name, sample_rate, percent / 100, sample_rate * (percent / 100)])
            print(f'Processed {file_name} with a break percentage of {percent / 100}')
            break
        else:
            percent -= 5


def sample_rate_test():
    with open('../../random_data/processing/data/bit_rate_results_new10.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        predictor = ModelPredictor(model_name='model_2')
        gtp = GroundtruthReader('../ground_truth/data/UrbanSound8K_groundtruth.csv')
        find_incorrect_prediction_sample_rate_with_objects = partial(find_incorrect_prediction_sample_rate, predictor,
                                                                     gtp, writer)
        apply_to_path(find_incorrect_prediction_sample_rate_with_objects,
                      'D:/Audio Features/UrbanSound8K/UrbanSound8K/audio/fold10', '.wav')


def plot_sample_rate_test_results():
    results = DataFrame()
    header_list = ['file_name', 'start_sample_rate', 'break_percentage', 'break_sample_rate']
    for i in range(1, 11):
        results = results.append(pd.read_csv(f'data/bit_rate_results_new{i}.csv', names=header_list))
    grouped = results.drop(['file_name', 'break_sample_rate'], axis=1).groupby(
        ['break_percentage', 'start_sample_rate'])
    grouped.size().unstack().plot(kind='bar', stacked=True)
    plt.xlabel('Break Sample Frequency Percentage')
    plt.ylabel('Number of Samples')
    plt.show()


if __name__ == '__main__':
    sample_rate_test()
    plot_sample_rate_test_results()
