from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, 3, activation="relu", padding="same", input_shape=input_shape), # calculate local features in sequence direction, resulting in 128 feature maps 
        BatchNormalization(), # normalize activation to help learn faster and more stable
        MaxPooling1D(2), # reduce sequence length (temporal downsampling)

        Conv1D(256, 3, activation="relu", padding="same"), # continue to extract deeper features, 256 filters
        BatchNormalization(), # normalize activation to help learn faster and more stable
        MaxPooling1D(2), # reduce further

        Flatten(), # transform 2D tensor (time × channels) into 1D vector
        Dropout(0.4), # regularization, drop 40% random neurons
        Dense(128, activation="relu"), # fully-connected to learn general features
        Dense(num_classes, activation="softmax") # multi-dimensional classification output (probability)
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
