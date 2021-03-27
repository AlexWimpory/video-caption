from numpy.ma import argmax
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from audio_utils.utils.audio_utils import save_object
import numpy as np
import pandas as pd
from sounds import sounds_config
from sounds.model_plotter import plot_history, plot_confusion_matrix


class AudioFeaturesModel:
    def __init__(self, model_name, le, layers):
        self.le = le
        self.model = Sequential(name=model_name)
        # Builds layers based on the structure in model_structures
        for layer in layers:
            self.model.add(layer)

    def compile(self):
        """Compile the model and print the structure"""
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model.summary()

    def test_model(self, x_data, y_data):
        """Calculate the model's accuracy on the input dataset"""
        score = self.model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        return accuracy

    def train_model(self, x_train, y_train, x_val, y_val):
        """Train and save the model"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=sounds_config.patience, mode='min')
        checkpointer = ModelCheckpoint(filepath=f'{sounds_config.sounds_model_dir}/{self.model.name}.hdf5', verbose=1,
                                       save_best_only=True)
        history = self.model.fit(x_train, y_train, batch_size=sounds_config.num_batch_size,
                                 epochs=sounds_config.num_epochs, validation_data=(x_val, y_val),
                                 callbacks=[checkpointer, early_stopping], verbose=1)
        self.le.save(self.model.name)
        return history

    def calculate_confusion_matrix(self, x_test, y_test):
        """Calculate the probabilities required for the confusion matrix and create a dataframe"""
        y_pred = self.model.predict_classes(x_test)
        y_test = argmax(y_test, axis=1)
        con_mat = confusion_matrix(labels=y_test, predictions=y_pred).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        classes = self.le.inverse_transform(list(range(0, self.le.encoded_labels.shape[1])))
        return pd.DataFrame(con_mat_norm, index=classes, columns=classes)


def train_and_test_model(features, le, model):
    """Use the AudioFeaturesModel methods to train and test the model"""
    # Split the data into training,validation and testing
    x_train, x_test, y_train, y_test = train_test_split(features, le.encoded_labels,
                                                        test_size=1 - sounds_config.train_ratio,
                                                        random_state=44)
    x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test,
                                                  test_size=sounds_config.test_ratio / (
                                                          sounds_config.test_ratio + sounds_config.validation_ratio),
                                                  random_state=44)
    save_test_data(x_test, y_test, le, model.model.name)
    # Calculate pre trained accuracy
    pre_acc = model.test_model(x_test, y_test)
    print(f'Pre-trained accuracy = {pre_acc:.4f}')
    # Train the model and plot the learning curves
    plot_history(model.train_model(x_train, y_train, x_cv, y_cv))
    # Test trained model on training, validation and test sets
    post_acc_train = model.test_model(x_train, y_train)
    print(f'Training accuracy = {post_acc_train:.4f}')
    post_acc_cv = model.test_model(x_cv, y_cv)
    print(f'Cross-validation accuracy = {post_acc_cv:.4f}')
    post_acc_test = model.test_model(x_test, y_test)
    print(f'Testing accuracy = {post_acc_test:.4f}')
    # Calculate and plot the confusion matrix
    plot_confusion_matrix(model.calculate_confusion_matrix(x_test, y_test))


def save_test_data(x_test, y_test, le, model_name):
    test_df = DataFrame()
    test_df['mfcc'] = x_test.tolist()
    test_df['encoded_labels'] = y_test.tolist()
    test_df['labels'] = test_df['encoded_labels'].apply(le.decode_label)
    test_df.drop(columns=['encoded_labels'])
    save_object(test_df, f'{sounds_config.sounds_data_dir}/{model_name}_test_df.data')
