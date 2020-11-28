from tensorflow.python.keras.layers import Dense, Dropout, Activation


def model_1(num_labels):
    return [
        Dense(256, input_shape=(40,)),
        Activation('relu'),
        Dropout(0.2),

        Dense(256),
        Activation('relu'),
        Dropout(0.2),

        Dense(num_labels),
        Activation('softmax')
    ]
