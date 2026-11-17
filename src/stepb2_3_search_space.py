import keras_tuner as kt
from itertools import product

from tuning import SEARCH_SPACE, N_CONFIGS, build_hp_model

configs = list(product(*SEARCH_SPACE.values()))
print("Простір пошуку:")
for k, v in SEARCH_SPACE.items():
    print(f"  {k}: {v}")
print(f"Кількість конфігурацій: {len(configs)} (очікувано {N_CONFIGS})")

# Перевірка: модель будується з об'єкта HyperParameters
hp = kt.HyperParameters()
model = build_hp_model(hp)
print("Значення за замовчуванням:", hp.values)
model.summary()