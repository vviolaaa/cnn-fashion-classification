"""Кроки C.2-C.3: карти Grad-CAM для прогнозованого (і справжнього) класу."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from data import load_data
from gradcam import (MODEL_PATH, RESULTS, last_conv_layer_name, gradcam,
                     cam_on_object, show_overlay, group_figure_path)

_, _, (x_test, y_test) = load_data()
model = keras.models.load_model(MODEL_PATH)
layer = last_conv_layer_name(model)
print("Шар для Grad-CAM:", layer, "->", model.get_layer(layer).output.shape)

ex = pd.read_csv(f"{RESULTS}/c_examples.csv")
ex["cam_on_object"] = 0.0

for group, g in ex.groupby("group", sort=False):
    fig, axes = plt.subplots(len(g), 3, figsize=(7.5, 2.6 * len(g)))
    for r, (k, row) in enumerate(g.iterrows()):
        img = x_test[row["idx"]]
        cam_pred = gradcam(model, img, row["pred_id"], layer)
        ex.loc[k, "cam_on_object"] = cam_on_object(cam_pred, img)
        axes[r, 0].imshow(img[..., 0], cmap="gray"); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"#{row['idx']}  справжній: {row['true']}", fontsize=8)
        show_overlay(axes[r, 1], img, cam_pred, f"Grad-CAM: {row['pred']} (p={row['prob']:.2f})")
        if row["pred_id"] != row["true_id"]:      # для помилок — ще й карта справжнього класу
            cam_true = gradcam(model, img, row["true_id"], layer)
            show_overlay(axes[r, 2], img, cam_true, f"Grad-CAM: справжній ({row['true']})")
        else:
            axes[r, 2].axis("off")
    fig.suptitle(f"{group} | шар: {layer}", fontsize=10)
    plt.tight_layout()
    plt.savefig(group_figure_path("c_gradcam", group), dpi=150)
    plt.close()

ex.to_csv(f"{RESULTS}/c_examples.csv", index=False)
print(ex[["group", "idx", "true", "pred", "prob", "cam_on_object"]].to_string(index=False))