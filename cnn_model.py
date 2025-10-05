from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, 3, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
