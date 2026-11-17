from data import load_data, CLASS_NAMES
from models import build_cnn
from train_utils import set_seed, compile_and_train, plot_history, evaluate

train, val, test = load_data()

# п. 2: побудова та аналіз структури
set_seed()
model = build_cnn(filters=(32, 64), kernel_size=3, pooling="max", activation="relu")
model.summary()

# п. 3: навчання
history, train_time = compile_and_train(model, train, val)
print(f"Training time: {train_time:.1f} s, params: {model.count_params()}")
plot_history(history, "Baseline", "../results/baseline_history.png")

# п. 3: оцінка на тесті
evaluate(model, test, CLASS_NAMES, "baseline")