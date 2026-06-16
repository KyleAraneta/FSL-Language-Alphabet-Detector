import os
import pandas as pd

DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")
BACKUP_FILE = os.path.join("data", "phrases", "fsl_phrase_data_before_rename_backup.csv")

RENAME_MAP = {
    "AYOS_LANG_ITS_OKAY": "AYOS_LANG",
    "SORRY_PASENSYA": "PASENSYA",
    "MALIGAYANG_PAGDATING": "MALIGAYANG_PAGDATING",
    "KAMI_AY_MGA": "KAMI_AY_MGA",
    "HINDI_KO_ALAM": "HINDI_KO_ALAM",
}

if not os.path.exists(DATA_FILE):
    print("Phrase dataset not found.")
    exit()

df = pd.read_csv(DATA_FILE, encoding="utf-8", encoding_errors="replace")

if df.empty:
    print("Phrase dataset is empty.")
    exit()

df.to_csv(BACKUP_FILE, index=False)
print(f"Backup saved as: {BACKUP_FILE}")

df["label"] = df["label"].astype(str).str.upper().str.strip()
df["label"] = df["label"].str.replace("?", "", regex=False)
df["label"] = df["label"].str.replace("/", "_", regex=False)
df["label"] = df["label"].str.replace("-", "_", regex=False)
df["label"] = df["label"].str.replace(" ", "_", regex=False)

df["label"] = df["label"].replace(RENAME_MAP)

df.to_csv(DATA_FILE, index=False)

print("Done renaming phrase labels.")
print("")
print("Updated phrase counts:")
print(df["label"].value_counts().sort_index())