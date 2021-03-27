import json
import os
from functools import partial
from logging_config import get_logger
from sounds import sounds_config
from sounds.audio_sound_pre_processing import prepare_audio_sound
from sounds.model_labeler import ModelLabelEncoder
from sounds.model_predictor import ModelPredictor
import pandas as pd
import numpy as np
from sounds.model_structures import *
from sounds.model_trainer import AudioFeaturesModel, train_and_test_model
from pandas import DataFrame
from utils.file_utils import return_from_path, save_object, load_object

logger = get_logger(__name__)


def save_features(groundtruth, path, dataset_name):
    prepare_audio_sound_groundtruth = partial(prepare_audio_sound,
                                              f'{sounds_config.sounds_data_dir}/{groundtruth}')
    if not os.path.isdir(f'{sounds_config.sounds_data_dir}/{dataset_name}'):
        os.mkdir(f'{sounds_config.sounds_data_dir}/{dataset_name}')
    ftrs = return_from_path(prepare_audio_sound_groundtruth,
                            path,
                            sounds_config.extension)
    audio_sound_df = DataFrame(ftrs)
    save_object(audio_sound_df, f'{sounds_config.sounds_data_dir}/{dataset_name}/{os.path.basename(path)}.data')


def train_sounds(model, path):
    """Load the data and process it before training and testing"""
    dataframes = return_from_path(load_object, path, '.data')
    features_and_labels = pd.concat(dataframes)
    labels = features_and_labels['labels'].tolist()
    ftrs = np.array(features_and_labels['mfcc'].to_list())
    label_encoder = ModelLabelEncoder(labels)
    mdl_structure = model_1(label_encoder.encoded_labels.shape[1])
    mdl = AudioFeaturesModel(model, label_encoder, mdl_structure)
    mdl.compile()
    train_and_test_model(ftrs, label_encoder, mdl)


def test_sounds_file(model, path):
    predictor = ModelPredictor(model_name=model)
    res = predictor.predict(path)
    print(json.dumps(res.__dict__))


def test_sounds_dataframe(model, path):
    predictor = ModelPredictor(model_name=model)
    df = load_object(path)
    result = predictor.evaluate_dataframe(df)
    print(result)


if __name__ == '__main__':
    # save_features('UrbanSound8K_groundtruth.csv', 'D:\Audio Features\Small', 'small')
    train_sounds('model_4', '../sounds_data/urbansounds')
    test_sounds_file('model_4', '../sounds_data/178686-0-0-63.wav')
    test_sounds_dataframe('model_4', '../sounds_data/test_df.data')
