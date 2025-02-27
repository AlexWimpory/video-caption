import os
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow.python.keras.models import load_model
from audio_pipeline import pipeline_config, logging_config
from audio_pipeline.audio_sounds.model_labeler import ModelLabelEncoder
from collections import Counter

logger = logging_config.get_logger(__name__)

"""
Performs the following operations:
1. Split an audio file into chunks
2. Calculate MFCCs for each chunk
3. Load the sound prediction model and give predictions for each chunk
4. If there is overlap create a single timeline from the results
"""


class SoundRecogniser:
    """
    Class which contains the sound classification model and the associated label encoder
    The model loaded can be changed in the config file
    Used to generate predictions from MFCCs
    """
    def __init__(self):
        cwd = os.path.join(os.path.dirname(__file__), pipeline_config.sound_model_file)
        # Load model and associated label encoder
        self._model = load_model(cwd)
        self._le = ModelLabelEncoder.load()
        logger.info(f'Loaded audio sound model from {cwd}')

    def process_file(self, file_name):
        """
        Generate list of dictionaries of classifications from MFCCs
        :param file_name: File to generate MFCCs from
        :return: List of dictionaries containing the predictied class, start/end time and the confidence
        """
        logger.info(f'Creating MFCCs for {file_name}')
        mfccs = create_mfcc(file_name)
        results = []
        for mfcc, start, end in mfccs:
            mfcc_array = np.array([mfcc_mean(mfcc)])
            predicted_vector = self._model.predict_classes(mfcc_array)
            predicted_class = self._le.inverse_transform(predicted_vector)

            predicted_probability_vector = self._model.predict(mfcc_array)
            predicted_probability = predicted_probability_vector[0]
            results.append({'class': predicted_class[0], 'conf': predicted_probability.max(),
                            'start': start, 'end': end})
        logger.info(f'Processed MFCCs, captured {len(results)} results')
        return results


def create_mfcc(path, sr=None):
    """
    Create an mfcc from the passed path
    """
    wave = AudioSegment.from_wav(path)
    # Get split times using overlap
    splits = calculate_splits(len(wave))
    mfccs = []
    mono = wave.channels != 1
    for split in splits:
        # Running Librosa on each chunk to calculate the MFCCs
        # This changes the format of the audio to a common format of mono and 22050Hz
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=split[0] / 1000,
                             duration=pipeline_config.duration / 1000, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs.append((mfcc, split[0] / 1000, (split[0] / 1000) + (pipeline_config.duration / 1000)))
    return mfccs


def mfcc_mean(mfcc):
    """
    MFCC matrix not a reasonable input for the MLP model so the matrix is converted to vector
    """
    return np.mean(mfcc.T, axis=0)


def calculate_splits(time_in_millis):
    """
    A function that given the length of an audio file create overlapping windows of capture time
    """
    split_times = []
    start_time = 0
    end_time = 0
    increment = int(pipeline_config.duration * pipeline_config.increment_percentage)
    while end_time < time_in_millis:
        end_time = start_time + pipeline_config.duration
        end_time = min(time_in_millis, end_time)
        split_times.append((start_time, end_time))
        start_time += increment
    return split_times


def process_overlap(sound_results):
    """
    Flatten overlap
    """
    new_results = []
    start = 0
    end = 1
    max_slice_size = pipeline_config.duration / (pipeline_config.duration * pipeline_config.increment_percentage)
    increment_time = (pipeline_config.duration * pipeline_config.increment_percentage) / 1000
    start_time = 0
    while True:
        result = process_result(sound_results[start:end])
        result['start'] = start_time
        result['end'] = start_time + increment_time
        new_results.append(result)
        start_time += increment_time
        end += 1
        if end > max_slice_size:
            start += 1
        if start >= len(sound_results):
            break
    return new_results


def process_result(sound_results_slice):
    """
    Find most common option or if no clear winner use the highest confidence
    """
    counter = Counter(result['class'] for result in sound_results_slice)
    classes = []
    last_count = 0
    for count in counter.most_common():
        if last_count != 0 and last_count != count[1]:
            break
        classes.append(count[0])
        last_count = count[1]
    sound_results = filter(lambda x: x['class'] in classes, sound_results_slice)
    sound_results = sorted(sound_results, key=lambda x: x['conf'], reverse=True)
    return sound_results[0].copy()
