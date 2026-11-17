import matplotlib
matplotlib.use("Agg")  # графіки лише зберігаються, вікна не відкриваються

import csv
from data import load_data
from models import build_cnn
from train_utils import set_seed, compile_and_train, plot_history

train, val, test = load_data()

# Кожен експеримент змінює ЛИШЕ ОДИН параметр відносно baseline
experiments = {
    "baseline":        dict(filters=(32, 64),  kernel_size=3, pooling="max",    activation="relu"),
    "filters_16_32":   dict(filters=(16, 32),  kernel_size=3, pooling="max",    activation="relu"),
    "filters_64_128":  dict(filters=(64, 128), kernel_size=3, pooling="max",    activation="relu"),
    "kernel_5":        dict(filters=(32, 64),  kernel_size=5, pooling="max",    activation="relu"),
    "pooling_stride":  dict(filters=(32, 64),  kernel_size=3, pooling="stride", activation="relu"),
    "activation_tanh": dict(filters=(32, 64),  kernel_size=3, pooling="max",    activation="tanh"),
}

rows = []
for name, params in experiments.items():
    print(f"\n=== {name} ===")
    set_seed()
    model = build_cnn(**params)
    history, train_time = compile_and_train(model, train, val)
    plot_history(history, name, f"../results/{name}_history.png")

    test_loss, test_acc = model.evaluate(test[0], test[1], verbose=0)
    rows.append({
        "experiment": name,
        "config": ", ".join(f"{k}={v}" for k, v in params.items()),
        "params": model.count_params(),
        "train_time_s": round(train_time, 1),
        "val_acc": round(history.history["val_accuracy"][-1], 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
    })

with open("../results/experiments.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved to ../results/experiments.csv")