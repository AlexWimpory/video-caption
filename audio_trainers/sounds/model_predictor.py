from sounds.mfcc_creator import mfcc_mean, create_mfcc
from tensorflow.python.keras.models import load_model
import numpy as np
from sounds import sounds_config
from sounds.model_labeler import ModelLabelEncoder

"""Load the model and the label encoder which are used to predict the class of the input .wav file"""


class ModelPredictor:
    def __init__(self, model_name):
        self._model = load_model(f'{sounds_config.sounds_model_dir}/{model_name}.hdf5')
        self._le = ModelLabelEncoder.load(model_name)

    def predict(self, file_name):
        results = ModelPredictorResults()
        mfcc = np.array([mfcc_mean(create_mfcc(file_name))])
        predicted_vector = self._model.predict_classes(mfcc)
        predicted_class = self._le.inverse_transform(predicted_vector)
        results.predicted_class = predicted_class[0]

        predicted_probability_vector = self._model.predict(mfcc)
        predicted_probability = predicted_probability_vector[0]

        for i in range(len(predicted_probability)):
            category = self._le.inverse_transform(np.array([i]))
            results.predicted_probabilities[category[0]] = str(format(predicted_probability[i], '.8f'))

        return results

    def evaluate_dataframe(self, dataframe):
        labels = dataframe['labels'].tolist()
        labels = self._le.transform_to_categorical(labels)
        ftrs = np.array(dataframe['mfcc'].to_list())
        score = self._model.evaluate(ftrs, labels, verbose=0)
        accuracy = 100 * score[1]
        return accuracy


class ModelPredictorResults:
    def __init__(self):
        self.predicted_class = None
        self.predicted_probabilities = {}
