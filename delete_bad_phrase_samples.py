import os
import pandas as pd

DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")
BACKUP_FILE = os.path.join("data", "phrases", "fsl_phrase_data_before_delete_bad_samples.csv")

# Change this if you want to delete another phrase later.
TARGET_LABEL = "ANO"

if not os.path.exists(DATA_FILE):
    print("Phrase dataset not found.")
    exit()

df = pd.read_csv(DATA_FILE, encoding="utf-8", encoding_errors="replace")

if df.empty:
    print("Phrase dataset is empty.")
    exit()

# Backup first
df.to_csv(BACKUP_FILE, index=False)
print(f"Backup saved as: {BACKUP_FILE}")

# Clean label format
df["label"] = df["label"].astype(str).str.upper().str.strip()
df["label"] = df["label"].str.replace("?", "", regex=False)
df["label"] = df["label"].str.replace("/", "_", regex=False)
df["label"] = df["label"].str.replace("-", "_", regex=False)
df["label"] = df["label"].str.replace(" ", "_", regex=False)

before_count = len(df)
target_count = (df["label"] == TARGET_LABEL).sum()

df = df[df["label"] != TARGET_LABEL]

after_count = len(df)

df.to_csv(DATA_FILE, index=False)

print("")
print("DONE.")
print(f"Deleted label: {TARGET_LABEL}")
print(f"Deleted samples: {target_count}")
print(f"Before rows: {before_count}")
print(f"After rows: {after_count}")
print("")
print("Now recollect the correct gesture for this phrase.")