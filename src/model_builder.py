from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Embedding, LSTM, GRU, Dense, BatchNormalization,
                                     Dropout, Flatten, Conv2D, InputLayer)

def build_text_classification_model(params):
    model = Sequential()
    model.add(Embedding(input_dim=params["vocab_size"], output_dim=params["embed_dim"], input_length=params["seq_len"]))
    model.add(LSTM(64) if params["rnn_type"] == "LSTM" else GRU(64))

    used_layers = 2
    remaining_layers = params["total_layers"] - used_layers
    units = 128

    for _ in range(remaining_layers // 3):
        model.add(Dense(units, activation=params["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropout_rate"]))
        units = max(16, units // 2)

    leftover = remaining_layers % 3
    if leftover >= 1:
        model.add(Dense(units, activation=params["activation"]))
    if leftover == 2:
        model.add(BatchNormalization())

    output_units = 1 if params["output_type"] == "binary" else params["num_classes"]
    activation = 'sigmoid' if params["output_type"] == "binary" else ('softmax' if params["output_type"] == "multiclass" else None)
    model.add(Dense(output_units, activation=activation))
    model.build(input_shape=(None, params["seq_len"]))

    return model


def build_image_classification_model(params):
    model = Sequential()
    model.add(InputLayer(input_shape=(params["img_height"], params["img_width"], params["channels"])))

    filters = 64
    remaining_layers = params["total_layers"]

    while remaining_layers >= 3:
        model.add(Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropout_rate"]))
        filters = min(512, filters * 2)
        remaining_layers -= 3

    if remaining_layers >= 1:
        model.add(Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same'))

    if remaining_layers == 2:
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1 if params["output_type"] == "binary" else params["num_classes"],
                    activation='sigmoid' if params["output_type"] == "binary" else 'softmax'))
    model.build(input_shape=(None, params["img_height"], params["img_width"], params["channels"]))

    return model


def build_regression_model(params):
    model = Sequential()
    model.add(InputLayer(input_shape=(params["num_features"],)))

    units = 128
    remaining = params["total_layers"] - 2

    for _ in range(remaining // 3):
        model.add(Dense(units, activation=params["activation"]))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropout_rate"]))
        units = max(16, units // 2)

    if remaining % 3 >= 1:
        model.add(Dense(units, activation=params["activation"]))
    if remaining % 3 == 2:
        model.add(BatchNormalization())

    model.add(Dense(1))
    model.build(input_shape=(None, params["num_features"]))

    return model


def build_time_series_model(params):
    model = Sequential()
    model.add(InputLayer(input_shape=(params["seq_len"], params["num_features"])))

    units = 128
    remaining = params["total_layers"] - 1

    for _ in range(remaining // 3):
        model.add(LSTM(units, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropout_rate"]))
        units = max(16, units // 2)

    if remaining % 3 >= 1:
        model.add(LSTM(units))
    if remaining % 3 == 2:
        model.add(BatchNormalization())

    if params["output_type"] == "regression":
        model.add(Dense(1))
    else:
        model.add(Dense(params["num_classes"], activation='softmax'))

    model.build(input_shape=(None, params["seq_len"], params["num_features"]))

    return model


def build_audio_classification_model(params):
    model = Sequential()
    model.add(InputLayer(input_shape=(params["signal_length"], params["channels"])))

    filters = 64
    remaining = params["total_layers"] - 1

    while remaining >= 3:
        model.add(Conv2D(filters, kernel_size=(3,1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropout_rate"]))
        filters = min(512, filters * 2)
        remaining -= 3

    if remaining >= 1:
        model.add(Conv2D(filters, kernel_size=(3,1), activation='relu', padding='same'))

    if remaining == 2:
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1 if params["output_type"] == "binary" else params["num_classes"],
                    activation='sigmoid' if params["output_type"] == "binary" else 'softmax'))
    model.build(input_shape=(None, params["signal_length"], params["channels"]))

    return model
