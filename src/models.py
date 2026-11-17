from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(filters=(32, 64), kernel_size=3, pooling="max",
              activation="relu", num_classes=10, input_shape=(28, 28, 1)):

    model = keras.Sequential(name="cnn")
    model.add(layers.Input(shape=input_shape))
    for f in filters:
        model.add(layers.Conv2D(f, kernel_size, padding="same"))
        model.add(layers.Activation(activation))
        if pooling == "max":
            model.add(layers.MaxPooling2D(2))
        else:
            model.add(layers.Conv2D(f, kernel_size, strides=2, padding="same"))
            model.add(layers.Activation(activation))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model