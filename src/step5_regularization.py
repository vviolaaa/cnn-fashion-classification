import matplotlib
matplotlib.use("Agg")

import csv
import matplotlib.pyplot as plt
from tensorflow import keras
from data import load_data, CLASS_NAMES
from models import build_regularized_cnn
from train_utils import set_seed, compile_and_train, plot_history, evaluate

train, val, test = load_data()

variants = {
    "es_only":      dict(),
    "augmentation": dict(use_aug=True),
    "batchnorm":    dict(use_bn=True),
    "dropout":      dict(dropout=0.3),
    "all_combined": dict(use_aug=True, use_bn=True, dropout=0.3),
}

rows, histories = [], {}
for name, params in variants.items():
    print(f"\n=== {name} ===")
    set_seed()
    model = build_regularized_cnn(**params)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                       restore_best_weights=True)
    history, t = compile_and_train(model, train, val, epochs=30, callbacks=[es])
    histories[name] = history
    plot_history(history, name, f"../results/reg_{name}_history.png")
    acc, loss = evaluate(model, test, CLASS_NAMES, f"reg_{name}")
    h = history.history
    rows.append({
        "experiment": name,
        "params": model.count_params(),
        "epochs_run": len(h["loss"]),
        "train_time_s": round(t, 1),
        "final_train_acc": round(h["accuracy"][-1], 4),
        "best_val_acc": round(max(h["val_accuracy"]), 4),
        "min_val_loss": round(min(h["val_loss"]), 4),
        "gap_train_val_acc": round(h["accuracy"][-1] - h["val_accuracy"][-1], 4),
        "test_acc": round(acc, 4),
        "test_loss": round(loss, 4),
    })

with open("../results/regularization.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# Порівняння val loss усіх варіантів
plt.figure(figsize=(8, 5))
for name, h in histories.items():
    plt.plot(h.history["val_loss"], label=name)
plt.xlabel("epoch"); plt.ylabel("val loss"); plt.legend()
plt.title("Validation loss: regularization comparison")
plt.savefig("../results/reg_val_loss_comparison.png", dpi=150)