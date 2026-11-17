import json
import os

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from train_utils import LEARNING_RATE

RESULTS = "../results"

SEARCH_SPACE = {
    "filters": [16, 32, 64],      
    "kernel_size": [3, 5],       
    "dropout": [0.0, 0.25, 0.5], 
}
N_CONFIGS = 18


def make_model(filters, kernel_size, dropout):
    """CNN, що збігається з build_cnn(): другий блок = 64 фільтри, MaxPooling2D(2),
    Dense(64), ReLU, Adam 1e-3. Конфігурація (32, 3, 0.0) еквівалентна базовій."""
    model = keras.Sequential(name="cnn_tuned")
    model.add(layers.Input(shape=(28, 28, 1)))
    for f in (filters, 64):
        model.add(layers.Conv2D(f, kernel_size, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_hp_model(hp):
    """Функція побудови для KerasTuner: значення беруться з HyperParameters."""
    return make_model(
        filters=hp.Choice("filters", SEARCH_SPACE["filters"]),
        kernel_size=hp.Choice("kernel_size", SEARCH_SPACE["kernel_size"]),
        dropout=hp.Choice("dropout", SEARCH_SPACE["dropout"]),
    )


def early_stop():
    """Єдине правило ранньої зупинки для всіх моделей."""
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True)


class BudgetTracker(keras.callbacks.Callback):
    """KerasTuner копіює callbacks для кожного випробування, тому лічильник
    зберігаємо на рівні класу."""
    total = 0

    @classmethod
    def reset(cls):
        cls.total = 0

    def on_epoch_end(self, epoch, logs=None):
        BudgetTracker.total += 1


def trials_to_df(tuner):
    """Усі випробування тюнера у таблиці, відсортованій за val_accuracy."""
    rows = []
    for t in tuner.oracle.trials.values():
        row = dict(t.hyperparameters.values)
        row["val_accuracy"] = t.score
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("val_accuracy", ascending=False).reset_index(drop=True)


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/{name}", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)
    
def planned_epochs(df):
    start = df["tuner/initial_epoch"].fillna(0)
    return int((df["tuner/epochs"] - start).sum())