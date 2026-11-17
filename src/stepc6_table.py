"""Крок C.6: збираємо підсумкову таблицю (числові колонки; висновки пишемо вручну)."""
import pandas as pd
from gradcam import RESULTS

ex = pd.read_csv(f"{RESULTS}/c_examples.csv")
occ = pd.read_csv(f"{RESULTS}/c_occlusion.csv")
df = ex.merge(occ, on="idx")

table = pd.DataFrame({
    "Зображення": [f"#{i}" for i in df["idx"]],
    "Справжній клас": df["true"],
    "Прогнозований клас": df["pred"],
    "Ймовірність прогнозу": df["prob"].map("{:.2f}".format),
    "Область Grad-CAM": df["cam_on_object"].map(lambda v: "карта порожня" if pd.isna(v) else f"{v:.0%} на об'єкті"),
    "Результат закриття області": [f"Grad-CAM: -{a:.2f}; випадкова: -{b:.2f}; r=" + ("н/д" if pd.isna(r) else f"{r:.2f}")
                                   for a, b, r in zip(df.drop_gradcam, df.drop_random, df.corr_cam_occ)],
    "Висновок": "",            # <-- заповнити вручну після перегляду малюнків
})
table.insert(0, "Група", df["group"])
table.to_csv(f"{RESULTS}/c_table.csv", index=False, encoding="utf-8-sig")
with open(f"{RESULTS}/c_table.md", "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(table.columns) + " |\n")
    f.write("|" + "---|" * len(table.columns) + "\n")
    for _, r in table.iterrows():
        f.write("| " + " | ".join(str(v) for v in r) + " |\n")
print(table.to_string(index=False))