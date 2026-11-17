"""Крок C.7: візуальна таблиця — замість номерів зображень показуємо самі зображення + Grad-CAM.
Колонку 'Висновок' заповнюємо вручну в results/c_table.csv (НЕ перезапускайте stepc6 після цього)."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from tensorflow import keras

from data import load_data
from gradcam import MODEL_PATH, RESULTS, last_conv_layer_name, gradcam, show_overlay, group_figure_path

_, _, (x_test, _) = load_data()
model = keras.models.load_model(MODEL_PATH)
layer = last_conv_layer_name(model)

ex = pd.read_csv(f"{RESULTS}/c_examples.csv")
tb = pd.read_csv(f"{RESULTS}/c_table.csv", encoding="utf-8-sig").fillna("")
assert list(tb["Зображення"]) == [f"#{i}" for i in ex["idx"]], "порядок рядків c_table.csv не збігається з c_examples.csv"

HEADERS = ["Зображення", "Grad-CAM\n(прогноз. клас)", "Справжній\nклас", "Прогнозований\nклас",
           "Ймовірність\nпрогнозу", "Область\nGrad-CAM", "Результат\nзакриття області", "Висновок"]
WIDTHS = [1, 1, 1.1, 1.2, 1, 1.5, 2.3, 3.6]
WRAP = [0, 0, 12, 14, 8, 16, 28, 48]


def cell(ax, text, wrap, bg, color="black", left=False, bold=False):
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=bg, zorder=0))
    txt = "\n".join(textwrap.fill(p, wrap) for p in str(text).split("\n")) if wrap else str(text)
    ax.text(0.03 if left else 0.5, 0.5, txt, ha="left" if left else "center", va="center",
            fontsize=9, color=color, fontweight="bold" if bold else "normal", transform=ax.transAxes)


for group, g in ex.groupby("group", sort=False):
    n = len(g)
    fig = plt.figure(figsize=(17, 2.1 * n + 0.8))
    gs = fig.add_gridspec(n + 1, len(HEADERS), width_ratios=WIDTHS,
                          height_ratios=[0.35] + [1] * n, wspace=0.02, hspace=0.04)
    for j, h in enumerate(HEADERS):
        cell(fig.add_subplot(gs[0, j]), h, 0, "#d9d9d9", bold=True)
    for r, (k, row) in enumerate(g.iterrows(), start=1):
        img = x_test[row["idx"]]
        cam = gradcam(model, img, int(row["pred_id"]), layer)
        a0, a1 = fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])
        show_overlay(a0, img, None, f"#{row['idx']}")
        show_overlay(a1, img, cam, "")
        bg = "#ffffff" if r % 2 else "#f4f4f4"
        ok = row["pred_id"] == row["true_id"]
        t = tb.loc[k]
        vals = [t["Справжній клас"], t["Прогнозований клас"], f"{float(t['Ймовірність прогнозу']):.2f}",
                t["Область Grad-CAM"], t["Результат закриття області"], t["Висновок"] or "—"]
        for j, v in enumerate(vals, start=2):
            color = ("#1a7f37" if ok else "#c62828") if j == 3 else "black"
            cell(fig.add_subplot(gs[r, j]), v, WRAP[j], bg, color=color, left=(j == 7), bold=(j == 3))
    fig.suptitle(f"Група: {group}", fontsize=12, y=0.995)
    fig.text(0.01, 0.0, "Результат закриття: зміна ймовірності прогнозованого класу (p до − p після). "
             "«+» = ймовірність впала, «−» = зросла. «випадкова» = закрито область тієї ж форми в іншому місці; "
             "r = кореляція карт Grad-CAM і Occlusion.", fontsize=8, va="top")
    fig.savefig(group_figure_path("c_visual_table", group), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("збережено", group_figure_path("c_visual_table", group))