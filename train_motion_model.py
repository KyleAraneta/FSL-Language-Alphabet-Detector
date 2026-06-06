import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

MOTION_DATA_FILE = "fsl_motion_data.csv"
MOTION_MODEL_FILE = "fsl_motion_model.joblib"

if not os.path.exists(MOTION_DATA_FILE):
    print("No fsl_motion_data.csv found. Run collect_motion_data.py first.")
    exit()

df = pd.read_csv(MOTION_DATA_FILE, encoding="utf-8", encoding_errors="replace")

if df.empty:
    print("Motion dataset is empty. Collect J, Z, and NONE first.")
    exit()

df["label"] = df["label"].astype(str).str.upper().str.strip()
df = df[df["label"].isin(["J", "Z", "NONE"])]

print("Samples per movement:")
print(df["label"].value_counts().sort_index())

X = df.drop(columns=["label"])
y = df["label"]

if y.nunique() < 2:
    print("Collect at least 2 movement classes first.")
    exit()

if y.value_counts().min() < 5:
    print("Collect at least 5 samples per movement class.")
    print("Recommended: 50 to 100 each for J, Z, and NONE.")
    exit()

test_size_count = max(int(len(y) * 0.2), y.nunique())
test_size = test_size_count / len(y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("")
print("Motion Accuracy:", accuracy_score(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))

joblib.dump(model, MOTION_MODEL_FILE)

print("")
print(f"Motion model saved as {MOTION_MODEL_FILE}")