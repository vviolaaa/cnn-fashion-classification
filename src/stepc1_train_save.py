"""Крок C.1: відтворюємо найкращу CNN з частини B (Hyperband) і зберігаємо її."""
import matplotlib
matplotlib.use("Agg")
from tensorflow import keras

from data import load_data, CLASS_NAMES
from train_utils import set_seed, compile_and_train, evaluate
from tuning import make_model, early_stop, load_json
from gradcam import MODEL_PATH, last_conv_layer_name

train, val, test = load_data()
hp = load_json("b_best_hp.json")["hyperband"]      # {'filters': 32, 'kernel_size': 5, 'dropout': 0.5}
print("Конфігурація:", hp)

set_seed()
model = make_model(**hp)
model.summary()
hist, t = compile_and_train(model, train, val, callbacks=[early_stop()])
acc, loss = evaluate(model, test, CLASS_NAMES, prefix="c_model")
print(f"test acc={acc:.4f}, loss={loss:.4f}, epochs={len(hist.epoch)}, time={t:.0f}s")

model.save(MODEL_PATH)
print("Збережено:", MODEL_PATH)
print("Шар для Grad-CAM:", last_conv_layer_name(model))