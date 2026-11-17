# %%
from data import load_data, CLASS_NAMES
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
print(x_train.shape, x_val.shape, x_test.shape)

# %%
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(x_train[idx].squeeze(), cmap="gray")
    ax.set_title(CLASS_NAMES[i])
    ax.axis("off")
plt.tight_layout()
plt.savefig("../results/class_examples.png", dpi=150)
plt.show()