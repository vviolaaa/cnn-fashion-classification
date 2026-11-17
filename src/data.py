import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def load_data():
    (x_full, y_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=5000, random_state=SEED, stratify=y_full
    )
    prep = lambda x: (x.astype("float32") / 255.0)[..., np.newaxis]
    return (prep(x_train), y_train), (prep(x_val), y_val), (prep(x_test), y_test)