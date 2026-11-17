"""Крок C.4: перевірка Grad-CAM методом закриття областей (Occlusion Sensitivity)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import load_data
from gradcam import (MODEL_PATH, RESULTS, PATCH, STRIDE, TOP_FRAC, last_conv_layer_name,
                     gradcam, top_mask, occlusion_map, prob_drop, random_drop,
                     show_overlay, group_figure_path, safe_corr)

(x_train, _), _, (x_test, y_test) = load_data()
FILL = float(x_train.mean())            # однакове значення заповнення для ВСІХ прикладів
model = keras.models.load_model(MODEL_PATH)
layer = last_conv_layer_name(model)
print(f"Патч {PATCH}x{PATCH}, крок {STRIDE}, заповнення {FILL:.3f}")

ex = pd.read_csv(f"{RESULTS}/c_examples.csv")
rows = []
for group, g in ex.groupby("group", sort=False):
    fig, axes = plt.subplots(len(g), 4, figsize=(10, 2.6 * len(g)))
    for r, (_, row) in enumerate(g.iterrows()):
        img, c = x_test[row["idx"]], int(row["pred_id"])   # досліджуваний клас = прогнозований
        cam = gradcam(model, img, c, layer)
        occ, p0 = occlusion_map(model, img, c, FILL)
        mask = top_mask(cam)
        d_cam = prob_drop(model, img, c, mask, FILL)         # закрили область Grad-CAM
        d_rand = random_drop(model, img, c, mask, FILL)      # закрили таку саму за формою, але в іншому місці
        corr = safe_corr(cam, occ)
        rows.append({"idx": row["idx"], "p0": round(p0, 4), "drop_gradcam": round(d_cam, 4),
                     "drop_random": round(d_rand, 4), "corr_cam_occ": round(corr, 3)})

        show_overlay(axes[r, 0], img, None, f"#{row['idx']} {row['true']} -> {row['pred']} ({p0:.2f})")
        show_overlay(axes[r, 1], img, cam, "Grad-CAM")
        im = axes[r, 2].imshow(occ, cmap="magma"); axes[r, 2].axis("off")
        axes[r, 2].set_title(f"Occlusion (r={corr:.2f})", fontsize=8)
        shown = img.copy(); shown[mask] = FILL
        show_overlay(axes[r, 3], shown, None,
                     f"закрито Grad-CAM: -{d_cam:.2f}; випадково: -{d_rand:.2f}")
    fig.suptitle(f"{group}: Grad-CAM vs Occlusion (світліше = більше падіння ймовірності)", fontsize=10)
    plt.tight_layout()
    plt.savefig(group_figure_path("c_occlusion", group), dpi=150)
    plt.close()

occ_df = pd.DataFrame(rows)
occ_df.to_csv(f"{RESULTS}/c_occlusion.csv", index=False)
print(occ_df.to_string(index=False))
print(f"\nCAM-область (топ {TOP_FRAC:.0%}) важливіша за випадкову у "
      f"{(occ_df.drop_gradcam > occ_df.drop_random).sum()} з {len(occ_df)} прикладів")