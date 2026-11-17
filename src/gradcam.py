"""Допоміжні функції для Grad-CAM та Occlusion Sensitivity (практична робота C)."""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

RESULTS = "../results"
MODEL_PATH = f"{RESULTS}/c_model.keras"
HIGH, LOW = 0.90, 0.60          # пороги впевненості: висока >= 0.90, низька < 0.60
TOP_FRAC = 0.20                 # "важлива область" = 20% пікселів з найвищим Grad-CAM
PATCH, STRIDE = 6, 2            # параметри Occlusion Sensitivity (однакові для всіх)


# ---------- Grad-CAM ----------
def last_conv_layer_name(model):
    """Назва шару ReLU одразу після останнього Conv2D (карта ознак 14x14 перед MaxPooling)."""
    idx = max(i for i, l in enumerate(model.layers) if isinstance(l, layers.Conv2D))
    return model.layers[idx + 1].name


def gradcam(model, img, class_idx, layer_name):
    """Grad-CAM для класу class_idx. Градієнт беремо від логіта (до softmax),
    бо softmax насичується і для впевнених прогнозів дає майже нульові градієнти.
    Повертає карту 28x28, нормовану до [0, 1]."""
    x = tf.convert_to_tensor(img[None], dtype=tf.float32)
    dense = model.layers[-1]
    with tf.GradientTape() as tape:
        h = x
        for layer in model.layers[:-1]:
            h = layer(h, training=False)
            if layer.name == layer_name:
                conv_out = h
        score = (tf.matmul(h, dense.kernel) + dense.bias)[:, class_idx]
    grads = tape.gradient(score, conv_out)                  # (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(1, 2))            # (1, C): важливість каналів
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)[..., None]
    cam = tf.image.resize(cam, img.shape[:2], method="bilinear").numpy()[..., 0]
    return cam / cam.max() if cam.max() > 0 else cam


def top_mask(cam, frac=TOP_FRAC):
    """Маска пікселів з найвищим значенням карти (верхні frac)."""
    thr = np.quantile(cam, 1 - frac)
    return (cam >= thr) & (cam > 0)


def cam_on_object(cam, img, thr=0.1):
    """Частка 'маси' карти, що припадає на об'єкт (пікселі яскравіші за thr), а не на фон."""
    obj = img[..., 0] > thr
    return float(cam[obj].sum() / cam.sum()) if cam.sum() > 0 else float("nan")


def safe_corr(a, b):
    """Кореляція Пірсона; NaN, якщо одна з карт константна (наприклад, Grad-CAM порожній)."""
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------- Occlusion Sensitivity ----------
def occlusion_map(model, img, class_idx, fill, patch=PATCH, stride=STRIDE):
    """Карта падіння ймовірності класу при закритті патча patch x patch.
    Усереднюється по всіх патчах, що покривають піксель."""
    H, W = img.shape[:2]
    p0 = float(model.predict(img[None], verbose=0)[0][class_idx])
    batch, coords = [], []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            m = img.copy()
            m[y:y + patch, x:x + patch, :] = fill
            batch.append(m)
            coords.append((y, x))
    probs = model.predict(np.array(batch), verbose=0)[:, class_idx]
    drop, cnt = np.zeros((H, W)), np.zeros((H, W))
    for (y, x), p in zip(coords, probs):
        drop[y:y + patch, x:x + patch] += p0 - p
        cnt[y:y + patch, x:x + patch] += 1
    return drop / np.maximum(cnt, 1), p0


def prob_drop(model, img, class_idx, mask, fill):
    """Падіння ймовірності класу, якщо закрити всі пікселі за маскою."""
    p0 = model.predict(img[None], verbose=0)[0][class_idx]
    m = img.copy()
    m[mask] = fill
    p1 = model.predict(m[None], verbose=0)[0][class_idx]
    return float(p0 - p1)


def random_drop(model, img, class_idx, mask, fill, n=30, seed=42):
    """Контроль: та сама форма маски, але зсунута у випадкове місце (середнє по n зсувах)."""
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n):
        dy, dx = rng.integers(-10, 11, size=2)
        shifted = np.roll(mask, (dy, dx), axis=(0, 1))
        drops.append(prob_drop(model, img, class_idx, shifted, fill))
    return float(np.mean(drops))


# ---------- Візуалізація ----------
def show_overlay(ax, img, cam, title=""):
    ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def group_figure_path(prefix, group):
    return f"{RESULTS}/{prefix}_{group}.png"