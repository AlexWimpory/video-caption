from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pickle


class ModelLabelEncoder:
    def __init__(self):
        self._le = LabelEncoder()

    def encode_labels(self, labels):
        label_array = np.array(labels).ravel()
        t = to_categorical(self._le.fit_transform(label_array))
        return t

    def inverse_transform(self, data):
        return self._le.inverse_transform(data)

    def save(self, model_name):
        with open(f'data/{model_name}_labels.data', 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(model_name):
        with open(f'data/{model_name}_labels.data', 'rb') as fin:
            return pickle.load(fin)
