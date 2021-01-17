import os
import numpy as np
import librosa
from pydub import AudioSegment
from tensorflow.python.keras.models import load_model
from audio_pipeline import config, logging_config
from audio_pipeline.audio_sounds.model_labeler import ModelLabelEncoder

logger = logging_config.get_logger(__name__)


class FeatureRecogniser:
    def __init__(self):
        cwd = os.path.join(os.path.dirname(__file__), config.feature_model_file)
        self._model = load_model(cwd)
        self._le = ModelLabelEncoder.load()
        logger.info(f'Loaded audio feature model from {cwd}')

    def process_file(self, file_name):
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
    """Create an mfcc from the passed path"""
    wave = AudioSegment.from_wav(path)
    splits = calculate_splits(len(wave))
    mfccs = []
    mono = wave.channels != 1
    for split in splits:
        y, sr = librosa.load(path, sr=sr, mono=mono, offset=split[0] / 1000,
                             duration=config.duration / 1000, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs.append((mfcc, split[0]/1000, (split[0]/1000) + (config.duration/1000)))
    return mfccs


def mfcc_mean(mfcc):
    """MFCC not a reasonable input for model so convert matrix to vector"""
    return np.mean(mfcc.T, axis=0)


def calculate_splits(time_in_millis):
    """A function that given the length of an audio file create overlapping windows of capture time"""
    split_times = []
    start_time = 0
    end_time = 0
    increment = int(config.duration * config.increment_percentage)
    while end_time < time_in_millis:
        end_time = start_time + config.duration
        end_time = min(time_in_millis, end_time)
        split_times.append((start_time, end_time))
        start_time += increment
    return split_times
