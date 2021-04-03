from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
from audio_pipeline import pipeline_config
import numpy as np
import pickle
import os

"""
Build and save an encoder that maps labels onto numerical values
* Numerical labels = [0,1,2]
* One hot encoding labels = [1,0,0],[0,1,0],[0,0,1]
"""


class ModelLabelEncoder:
    def __init__(self, labels):
        self._le = LabelEncoder()
        label_array = np.array(labels).ravel()
        self.encoded_labels = to_categorical(self._le.fit_transform(label_array))

    def inverse_transform(self, data):
        return self._le.inverse_transform(data)

    @staticmethod
    def load():
        cwd = os.path.join(os.path.dirname(__file__), pipeline_config.sound_label_file)
        with open(cwd, 'rb') as fin:
            return pickle.load(fin)
