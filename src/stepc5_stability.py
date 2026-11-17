"""Крок C.5: стійкість Grad-CAM до малих змін, що зберігають зміст зображення."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import load_data
from gradcam import (MODEL_PATH, RESULTS, last_conv_layer_name, gradcam, top_mask,
                     show_overlay, safe_corr)

_, _, (x_test, y_test) = load_data()
model = keras.models.load_model(MODEL_PATH)
layer = last_conv_layer_name(model)
rng = np.random.default_rng(42)

variants = {
    "яскравість x1.2": lambda x: np.clip(x * 1.2, 0, 1),
    "яскравість x0.8": lambda x: x * 0.8,
    "шум σ=0.03":      lambda x: np.clip(x + rng.normal(0, 0.03, x.shape), 0, 1).astype("float32"),
}

ex = pd.read_csv(f"{RESULTS}/c_examples.csv")
sel = ex.groupby("group", sort=False).head(2)          # по 2 зображення з кожної групи = 8

def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))

fig, axes = plt.subplots(len(sel), 4, figsize=(10, 2.6 * len(sel)))
rows = []
for r, (_, row) in enumerate(sel.iterrows()):
    img = x_test[row["idx"]]
    c0 = int(row["pred_id"])
    cam0 = gradcam(model, img, c0, layer)
    show_overlay(axes[r, 0], img, cam0, f"#{row['idx']} оригінал: {row['pred']} ({row['prob']:.2f})")
    for j, (name, f) in enumerate(variants.items(), start=1):
        img2 = f(img)
        probs = model.predict(img2[None], verbose=0)[0]
        c2 = int(probs.argmax())
        cam2 = gradcam(model, img2, c0, layer)         # той самий клас, щоб карти були порівнянні
        corr = safe_corr(cam0, cam2)
        ov = iou(top_mask(cam0), top_mask(cam2))
        rows.append({"idx": row["idx"], "variant": name,
                     "pred_changed": c2 != c0, "p_new": round(float(probs[c0]), 4),
                     "corr": round(corr, 3), "iou_top20": round(ov, 3)})
        show_overlay(axes[r, j], img2, cam2, f"{name}: p={probs[c0]:.2f}, r={corr:.2f}, IoU={ov:.2f}")
plt.tight_layout()
plt.savefig(f"{RESULTS}/c_stability.png", dpi=150)
plt.close()

df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/c_stability.csv", index=False)
print(df.to_string(index=False))
print("\nСередня кореляція карт:", df["corr"].mean().round(3),
      "| середній IoU:", df["iou_top20"].mean().round(3),
      "| змін прогнозу:", int(df["pred_changed"].sum()), "з", len(df))