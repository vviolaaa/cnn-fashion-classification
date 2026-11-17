"""Крок C.1: добір 12 зображень з тестової вибірки (по 3 у кожній з 4 груп)."""
import numpy as np
import pandas as pd
from tensorflow import keras

from data import load_data, CLASS_NAMES
from gradcam import MODEL_PATH, RESULTS, HIGH, LOW

SEED = 42
_, _, (x_test, y_test) = load_data()
model = keras.models.load_model(MODEL_PATH)

probs = model.predict(x_test, verbose=0)
pred = probs.argmax(1)
conf = probs.max(1)
correct = pred == y_test

groups = {
    "correct_high": correct & (conf >= HIGH),
    "correct_low":  correct & (conf < LOW),
    "wrong_high":  ~correct & (conf >= HIGH),
    "wrong_low":   ~correct & (conf < LOW),
}

rng = np.random.default_rng(SEED)
rows = []
for name, mask in groups.items():
    candidates = rng.permutation(np.where(mask)[0])
    print(f"{name}: доступно {len(candidates)}")
    chosen, used_classes = [], set()
    for i in candidates:                       # спершу намагаємось взяти різні справжні класи
        if y_test[i] not in used_classes:
            chosen.append(i)
            used_classes.add(y_test[i])
        if len(chosen) == 3:
            break
    for i in candidates:                       # якщо не вистачило — добираємо будь-які
        if len(chosen) == 3:
            break
        if i not in chosen:
            chosen.append(i)
    for i in chosen:
        rows.append({"group": name, "idx": int(i),
                     "true": CLASS_NAMES[y_test[i]], "true_id": int(y_test[i]),
                     "pred": CLASS_NAMES[pred[i]], "pred_id": int(pred[i]),
                     "prob": round(float(conf[i]), 4)})

df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/c_examples.csv", index=False)
print(df.to_string(index=False))