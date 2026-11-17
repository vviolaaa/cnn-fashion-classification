"""Крок b5. Аналіз результатів KerasTuner."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tuning import RESULTS, load_json

pd.set_option("display.width", 160)
hb = pd.read_csv(f"{RESULTS}/b_hyperband_trials.csv")
rs = pd.read_csv(f"{RESULTS}/b_random_trials.csv")
cfg = ["filters", "kernel_size", "dropout"]

print("Hyperband: топ-5 випробувань")
print(hb[cfg + ["tuner/epochs", "val_accuracy"]].head(5).to_string(index=False))
print("\nRandomSearch: топ-5 випробувань")
print(rs[cfg + ["val_accuracy"]].head(5).to_string(index=False))

# Як змінюється результат конфігурації зі збільшенням бюджету епох.
pivot = hb.pivot_table(index=cfg, columns="tuner/epochs", values="val_accuracy")
pivot = pivot.sort_values(by=list(pivot.columns)[-1], ascending=False, na_position="last")
print("\nval_accuracy конфігурацій за етапами Hyperband (колонки = бюджет епох):")
print(pivot.round(4).to_string())
pivot.round(4).to_csv(f"{RESULTS}/b_hyperband_stages.csv")

# Графік 1: бюджет епох vs val_accuracy
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(hb["tuner/epochs"], hb["val_accuracy"], alpha=0.7)
ax.set_xlabel("бюджет епох випробування (tuner/epochs)")
ax.set_ylabel("val_accuracy")
ax.set_title("Hyperband: ранні етапи vs фінальні")
plt.tight_layout()
plt.savefig(f"{RESULTS}/b_hyperband_trials.png", dpi=150)
plt.close()

# Графік 2: теплова карта val_accuracy 
final = hb.loc[hb.groupby(cfg)["tuner/epochs"].idxmax()]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
vmin, vmax = final["val_accuracy"].min(), final["val_accuracy"].max()
for ax, d in zip(axes, sorted(final["dropout"].unique())):
    sub = final[final["dropout"] == d].pivot(
        index="filters", columns="kernel_size", values="val_accuracy")
    sns.heatmap(sub, annot=True, fmt=".4f", cmap="viridis", vmin=vmin, vmax=vmax,
                ax=ax, cbar=ax is axes[-1])
    ax.set_title(f"dropout = {d}")
fig.suptitle("Hyperband: val_accuracy конфігурацій")
plt.tight_layout()
plt.savefig(f"{RESULTS}/b_hyperband_heatmap.png", dpi=150)
plt.close()

s = load_json("b_search_summary.json")
print("\nБюджет пошуку:")
for name, v in s.items():
    print(f"  {name}: випробувань {v['trials']}, епох {v['total_epochs']}, час {v['time_s']} с")