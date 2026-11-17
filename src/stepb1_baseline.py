import matplotlib
matplotlib.use("Agg")

from data import load_data, CLASS_NAMES
from models import build_cnn
from train_utils import set_seed, compile_and_train, plot_history, evaluate
from tuning import early_stop, save_json

train, val, test = load_data()
print("train/val/test:", train[0].shape, val[0].shape, test[0].shape)

set_seed()
model = build_cnn()  # дефолт: filters=(32, 64), kernel 3, max pooling, без Dropout
model.summary()

hist, t = compile_and_train(model, train, val, callbacks=[early_stop()])
plot_history(hist, "baseline", "../results/b_baseline_history.png")
acc, loss = evaluate(model, test, CLASS_NAMES, prefix="b_baseline")

metrics = {
    "model": "baseline",
    "config": {"filters": 32, "kernel_size": 3, "dropout": 0.0},
    "val_acc": max(hist.history["val_accuracy"]),
    "test_acc": acc,
    "test_loss": loss,
    "params": model.count_params(),
    "train_time_s": round(t, 1),
    "epochs_run": len(hist.epoch),
}
save_json(metrics, "b_baseline_metrics.json")
print(metrics)