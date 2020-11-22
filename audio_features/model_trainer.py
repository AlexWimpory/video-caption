from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from audio_feature_pre_processing import load_features
import numpy as np
from model_labeler import ModelLabelEncoder


def create_tensorflow(num_labels, model_name):
    model = Sequential(name=model_name)

    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    return model


def test_model(x_data, y_data, model):
    score = model.evaluate(x_data, y_data, verbose=0)
    accuracy = 100 * score[1]
    return accuracy


def train_model(x_train, y_train, x_val, y_val, model, le):
    num_epochs = 100
    num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath=f'data/{model.name}.hdf5', verbose=1, save_best_only=True)

    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_val, y_val),
              callbacks=[checkpointer], verbose=1)

    le.save(model.name)


if __name__ == '__main__':
    features = load_features('data/UrbanSound8K_all.data')
    le = ModelLabelEncoder()
    labels = [feature['labels'] for feature in features]
    X = np.array([feature['mfcc'] for feature in features])
    encoded_labels = le.encode_labels(labels)

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    x_train, x_test, y_train, y_test = train_test_split(X, encoded_labels,
                                                        test_size=1 - train_ratio,
                                                        random_state=44)
    x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test,
                                                  test_size=test_ratio / (test_ratio + validation_ratio),
                                                  random_state=44)

    mdl = create_tensorflow(encoded_labels.shape[1], 'model_1')
    mdl.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    mdl.summary()

    pre_acc = test_model(x_test, y_test, mdl)
    print(f'Pre-trained accuracy = {pre_acc:.4f}')

    train_model(x_train, y_train, x_cv, y_cv, mdl, le)

    post_acc_train = test_model(x_train, y_train, mdl)
    print(f'Training accuracy = {post_acc_train:.4f}')

    post_acc_cv = test_model(x_cv, y_cv, mdl)
    print(f'Cross-validation accuracy = {post_acc_cv:.4f}')

    post_acc_test = test_model(x_test, y_test, mdl)
    print(f'Testing accuracy = {post_acc_test:.4f}')
