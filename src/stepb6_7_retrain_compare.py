import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import load_data, CLASS_NAMES
from train_utils import set_seed, compile_and_train, plot_history, evaluate
from tuning import make_model, early_stop, load_json, RESULTS

train, val, test = load_data()
best = load_json("b_best_hp.json")
results = [load_json("b_baseline_metrics.json")]

for name, hp in [("hyperband_best", best["hyperband"]), ("random_best", best["random"])]:
    print(f"\n=== {name}: {hp} ===")
    set_seed()
    model = make_model(**hp)
    hist, t = compile_and_train(model, train, val, callbacks=[early_stop()])
    plot_history(hist, name, f"{RESULTS}/b_{name}_history.png")
    # тестова вибірка використовується лише тут, після вибору конфігурації
    acc, loss = evaluate(model, test, CLASS_NAMES, prefix=f"b_{name}")
    results.append({
        "model": name, "config": hp,
        "val_acc": max(hist.history["val_accuracy"]),
        "test_acc": acc, "test_loss": loss,
        "params": model.count_params(),
        "train_time_s": round(t, 1),
        "epochs_run": len(hist.epoch),
    })

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS}/b_comparison.csv", index=False)
print("\nПорівняння:")
print(df.drop(columns="config").to_string(index=False))

# Графік порівняння моделей
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [("test_acc", "Test accuracy (вісь не з нуля)"),
          ("params", "Кількість параметрів"),
          ("train_time_s", "Час навчання, с")]
for ax, (col, title) in zip(axes, panels):
    ax.bar(df["model"], df[col], color=["gray", "tab:blue", "tab:orange"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
axes[0].set_ylim(df["test_acc"].min() - 0.01, df["test_acc"].max() + 0.005)
plt.tight_layout()
plt.savefig(f"{RESULTS}/b_comparison_bars.png", dpi=150)
plt.close()