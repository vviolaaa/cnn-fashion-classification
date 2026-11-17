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

def build_regularized_cnn(filters=(32, 64), kernel_size=3, activation="relu",
                          use_aug=False, use_bn=False, dropout=0.0,
                          num_classes=10, input_shape=(28, 28, 1)):
    model = keras.Sequential(name="cnn_reg")
    model.add(layers.Input(shape=input_shape))
    if use_aug:  # активна лише під час навчання
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(0.05))
        model.add(layers.RandomZoom(0.1))
    for f in filters:
        model.add(layers.Conv2D(f, kernel_size, padding="same", use_bias=not use_bn))
        if use_bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(64, activation=activation))
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

def residual_block(x, filters, use_skip=True):
    """Два Conv3x3 + BN; skip-з'єднання додає вхід до виходу: y = F(x) + x."""
    shortcut = x
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if use_skip:
        if shortcut.shape[-1] != filters:  # вирівнюємо кількість каналів 1x1-згорткою
            shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.Add()([y, shortcut])
    return layers.Activation("relu")(y)


def build_res_cnn(use_skip=True, dropout=0.3, num_classes=10, input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = residual_block(x, 32, use_skip)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 64, use_skip)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet_cnn" if use_skip else "plain_cnn")