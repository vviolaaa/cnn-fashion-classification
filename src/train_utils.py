import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def set_seed():
    keras.utils.set_random_seed(SEED)


def compile_and_train(model, train, val, epochs=EPOCHS, callbacks=None):
    """Єдині умови: Adam, lr=1e-3, batch=128, loss=sparse_categorical_crossentropy."""
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    start = time.time()
    history = model.fit(train[0], train[1], validation_data=val,
                        epochs=epochs, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=2)
    return history, time.time() - start


def plot_history(history, title, path):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title(f"{title}: accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title(f"{title}: loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def evaluate(model, test, class_names, prefix):
    """Точність, матриця помилок, звіт по класах."""
    x, y = test
    loss, acc = model.evaluate(x, y, verbose=0)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)

    print(f"Test accuracy: {acc:.4f}, test loss: {loss:.4f}")
    print(classification_report(y, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{prefix}: confusion matrix")
    plt.tight_layout()
    plt.savefig(f"../results/{prefix}_confusion_matrix.png", dpi=150)
    plt.show()
    return acc, loss