from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pickle

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

    @property
    def encoded_labels(self):
        return self.__encoded_labels

    @encoded_labels.setter
    def encoded_labels(self, encoded_labels):
        self.__encoded_labels = encoded_labels

    def inverse_transform(self, data):
        return self._le.inverse_transform(data)

    def transform(self, data):
        return self._le.transform(data)

    def transform_to_categorical(self, data):
        return to_categorical(self.transform(data))

    def decode_label(self, category):
        return self._le.inverse_transform([np.argmax(category)])

    def save(self, model_name):
        with open(f'data/{model_name}_labels.data', 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(model_name):
        with open(f'data/{model_name}_labels.data', 'rb') as fin:
            return pickle.load(fin)
