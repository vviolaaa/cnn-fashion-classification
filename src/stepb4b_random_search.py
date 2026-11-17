import matplotlib
matplotlib.use("Agg")
import time
import pandas as pd
import keras_tuner as kt

from data import load_data, SEED
from train_utils import set_seed, EPOCHS, BATCH_SIZE
from tuning import (build_hp_model, early_stop, BudgetTracker, trials_to_df,
                    planned_epochs, save_json, load_json, N_CONFIGS, RESULTS)

train, val, test = load_data()

hb_df = pd.read_csv(f"{RESULTS}/b_hyperband_trials.csv")
hb_epochs = planned_epochs(hb_df)
hb_time = load_json("b_search_summary.json")["hyperband"]["time_s"]
rs_trials = int(min(N_CONFIGS, max(1, round(hb_epochs / EPOCHS))))
print(f"Бюджет Hyperband: {hb_epochs} епох -> RandomSearch: {rs_trials} випробувань по {EPOCHS} епох")

set_seed()
BudgetTracker.reset()
rs = kt.RandomSearch(
    build_hp_model, objective="val_accuracy",
    max_trials=rs_trials, seed=SEED,
    directory=f"{RESULTS}/kerastuner", project_name="random", overwrite=True,
)
t0 = time.time()
rs.search(train[0], train[1], validation_data=val, epochs=EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[early_stop(), BudgetTracker()], verbose=0)
rs_time = time.time() - t0

df = trials_to_df(rs)
df.to_csv(f"{RESULTS}/b_random_trials.csv", index=False)

keys = ["filters", "kernel_size", "dropout"]
best = rs.get_best_hyperparameters(1)[0].values
hp_all = load_json("b_best_hp.json")
hp_all["random"] = {k: best[k] for k in keys}
save_json(hp_all, "b_best_hp.json")
save_json({
    "hyperband": {"trials": len(hb_df), "total_epochs": hb_epochs, "time_s": hb_time},
    "random": {"trials": len(df), "total_epochs": BudgetTracker.total,
               "time_s": round(rs_time, 1)},
}, "b_search_summary.json")
print(f"RandomSearch: випробувань {len(df)}, епох {BudgetTracker.total}, час {rs_time:.0f} с")
print("Найкраща:", hp_all["random"])