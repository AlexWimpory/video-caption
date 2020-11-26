from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from audio_feature_pre_processing import load_features
from model_labeler import ModelLabelEncoder
from model_structures import *
import numpy as np
import features_config


class AudioFeaturesModel:
    def __init__(self, model_name, le, layers):
        self.le = le
        self.model = Sequential(name=model_name)

        for layer in layers:
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model.summary()

    def test_model(self, x_data, y_data):
        score = self.model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        return accuracy

    def train_model(self, x_train, y_train, x_val, y_val):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5', verbose=1, save_best_only=True)
        self.model.fit(x_train, y_train, batch_size=features_config.num_batch_size,
                       epochs=features_config.num_epochs, validation_data=(x_val, y_val),
                       callbacks=[checkpointer], verbose=1)
        self.le.save(self.model.name)


def train_and_test_model(features, le, model):
    x_train, x_test, y_train, y_test = train_test_split(features, le.encoded_labels,
                                                        test_size=1 - features_config.train_ratio,
                                                        random_state=features_config.random_state)
    x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test,
                                                  test_size=features_config.test_ratio / (
                                                          features_config.test_ratio + features_config.validation_ratio),
                                                  random_state=features_config.random_state)

    pre_acc = model.test_model(x_test, y_test)
    print(f'Pre-trained accuracy = {pre_acc:.4f}')

    model.train_model(x_train, y_train, x_cv, y_cv)

    post_acc_train = model.test_model(x_train, y_train)
    print(f'Training accuracy = {post_acc_train:.4f}')

    post_acc_cv = model.test_model(x_cv, y_cv)
    print(f'Cross-validation accuracy = {post_acc_cv:.4f}')

    post_acc_test = model.test_model(x_test, y_test)
    print(f'Testing accuracy = {post_acc_test:.4f}')


if __name__ == '__main__':
    features_and_labels = load_features('data/UrbanSound8K_all.data')
    labels = [feature['labels'] for feature in features_and_labels]
    ftrs = np.array([feature['mfcc'] for feature in features_and_labels])
    label_encoder = ModelLabelEncoder(labels)
    mdl_structure = model_1(label_encoder.encoded_labels.shape[1])
    mdl = AudioFeaturesModel('model_1', label_encoder, mdl_structure)
    mdl.compile()
    train_and_test_model(ftrs, label_encoder, mdl)
