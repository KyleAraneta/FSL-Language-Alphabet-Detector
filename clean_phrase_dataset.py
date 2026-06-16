import os
import re
import pandas as pd

DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")
BACKUP_FILE = os.path.join("data", "phrases", "fsl_phrase_data_before_final_clean.csv")

DELETE_LABELS = {
    "AYOS_LANG_ITS_OKAY",
    "SORRY_PASENSYA",
}

RENAME_LABELS = {
    "AYOS_LANG__ITS_OKAY": "AYOS_LANG_ITS_OKAY",
    "SORRY__PASENSYA": "SORRY_PASENSYA",
}

REMOVE_LABELS_WITH_LESS_THAN_5_SAMPLES = True

def clean_label(label):
    label = str(label).upper().strip()
    label = label.replace("?", "")
    label = label.replace("/", "_")
    label = label.replace("-", "_")
    label = label.replace(" ", "_")
    label = re.sub(r"_+", "_", label)
    return label.strip("_")

if not os.path.exists(DATA_FILE):
    print("Phrase dataset not found.")
    exit()

df = pd.read_csv(DATA_FILE, encoding="utf-8", encoding_errors="replace")

if df.empty:
    print("Phrase dataset is empty.")
    exit()

df.to_csv(BACKUP_FILE, index=False)
print(f"Backup saved as: {BACKUP_FILE}")

df["label"] = df["label"].apply(clean_label)
df["label"] = df["label"].replace(RENAME_LABELS)

print("\nBefore cleaning:")
print(df["label"].value_counts().sort_index())

before_rows = len(df)

df = df[~df["label"].isin(DELETE_LABELS)]

if REMOVE_LABELS_WITH_LESS_THAN_5_SAMPLES:
    counts = df["label"].value_counts()
    weak_labels = counts[counts < 5].index.tolist()

    if weak_labels:
        print("\nRemoving labels with fewer than 5 samples:")
        for label in weak_labels:
            print(f"- {label}: {counts[label]}")
        df = df[~df["label"].isin(weak_labels)]

df.to_csv(DATA_FILE, index=False)

print("\nDONE.")
print(f"Before rows: {before_rows}")
print(f"After rows: {len(df)}")

print("\nAfter cleaning:")
print(df["label"].value_counts().sort_index())