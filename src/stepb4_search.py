import matplotlib
matplotlib.use("Agg")
import time
import keras_tuner as kt

from data import load_data, SEED
from train_utils import set_seed, EPOCHS, BATCH_SIZE
from tuning import (build_hp_model, early_stop, trials_to_df, planned_epochs,
                    save_json, RESULTS)

train, val, test = load_data()

set_seed()
hb = kt.Hyperband(
    build_hp_model, objective="val_accuracy",
    max_epochs=EPOCHS, factor=3, hyperband_iterations=1, seed=SEED,
    directory=f"{RESULTS}/kerastuner", project_name="hyperband", overwrite=True,
)
t0 = time.time()
hb.search(train[0], train[1], validation_data=val, batch_size=BATCH_SIZE,
          callbacks=[early_stop()], verbose=0)
hb_time = time.time() - t0

df = trials_to_df(hb)
df.to_csv(f"{RESULTS}/b_hyperband_trials.csv", index=False)

keys = ["filters", "kernel_size", "dropout"]
best = hb.get_best_hyperparameters(1)[0].values
save_json({"hyperband": {k: best[k] for k in keys}}, "b_best_hp.json")
save_json({"hyperband": {"trials": len(df), "total_epochs": planned_epochs(df),
                         "time_s": round(hb_time, 1)}}, "b_search_summary.json")
print(f"Hyperband: випробувань {len(df)}, епох (за розкладом) {planned_epochs(df)}, "
      f"час {hb_time:.0f} с")
print("Найкраща:", {k: best[k] for k in keys})