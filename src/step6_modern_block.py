import matplotlib
matplotlib.use("Agg")

import csv
from tensorflow import keras
from data import load_data, CLASS_NAMES
from models import build_res_cnn
from train_utils import set_seed, compile_and_train, plot_history, evaluate

train, val, test = load_data()

variants = {"plain_no_skip": False, "resnet_skip": True}

rows = []
for name, use_skip in variants.items():
    print(f"\n=== {name} ===")
    set_seed()
    model = build_res_cnn(use_skip=use_skip)
    model.summary()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                       restore_best_weights=True)
    history, t = compile_and_train(model, train, val, epochs=30, callbacks=[es])
    plot_history(history, name, f"../results/res_{name}_history.png")
    acc, loss = evaluate(model, test, CLASS_NAMES, f"res_{name}")
    h = history.history
    rows.append({
        "experiment": name,
        "params": model.count_params(),
        "epochs_run": len(h["loss"]),
        "train_time_s": round(t, 1),
        "final_train_acc": round(h["accuracy"][-1], 4),
        "best_val_acc": round(max(h["val_accuracy"]), 4),
        "min_val_loss": round(min(h["val_loss"]), 4),
        "test_acc": round(acc, 4),
        "test_loss": round(loss, 4),
    })

with open("../results/modern_block.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)