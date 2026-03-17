import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Save plots without opening a window
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier


# =========================
# SETTINGS
# =========================
CSV_PATH = "Data/CA_multihot_seasons.csv"
RANDOM_STATE = 42

TARGET_PREFIX = "MostSeriousBiasType__"   # target columns
LEAKAGE_PREFIX = "MostSeriousBias__"      # columns to remove from X
ID_COL = "RecordID"

os.makedirs("figures", exist_ok=True)


# Prep data
df = pd.read_csv(CSV_PATH)

# These are the one-hot target columns we want to predict
target_cols = [col for col in df.columns if col.startswith(TARGET_PREFIX)]
if not target_cols:
    raise ValueError(f"No target columns found with prefix '{TARGET_PREFIX}'")

# These columns leak the answer, so remove them from the features
leakage_cols = [col for col in df.columns if col.startswith(LEAKAGE_PREFIX)]

# Convert one-hot target columns into one class label per row
y_str = (
    df[target_cols]
    .idxmax(axis=1)
    .str.replace(TARGET_PREFIX, "", regex=False)
)

# Drop target columns, leakage columns, and ID column from X
drop_cols = target_cols + leakage_cols
if ID_COL in df.columns:
    drop_cols.append(ID_COL)

X = df.drop(columns=drop_cols).copy()

# Encode string labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)
class_names = list(label_encoder.classes_)

print("Dataset shape:", df.shape)
print("Feature matrix shape:", X.shape)
print("Classes:", class_names)
print("\nClass distribution:")
print(pd.Series(y_str).value_counts())
print("\nRemoved leakage columns:", len(leakage_cols))


# Helpers
def print_metrics(y_true, y_pred, class_names, title):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Balanced Accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
    print("Macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )


def save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    # Row-normalized confusion matrix: each row sums to 1
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)

    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


# 90/10 Split
# Stratify keeps class proportions similar in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nSplit sizes:")
print("Train:", X_train.shape, len(y_train))
print("Test: ", X_test.shape, len(y_test))


# Train random forest
# balanced_subsample makes rare classes matter more during training
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)



# Evaluate on test set
y_test_pred = rf_model.predict(X_test)

print_metrics(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="Random Forest | 90/10 Split | Test Results"
)

save_confusion_matrix(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="Random Forest | 90/10 Split | Test Confusion Matrix",
    filename="figures/rf_90_10_test_confusion_matrix.png"
)

print("\nDone. Saved:")
print("figures/rf_90_10_test_confusion_matrix.png")