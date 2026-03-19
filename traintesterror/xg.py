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
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


CSV_PATH = "Data/CA_multihot_seasons.csv"
RANDOM_STATE = 42

TARGET_PREFIX = "MostSeriousBiasType__"
LEAKAGE_PREFIX = "MostSeriousBias__"
ID_COL = "RecordID"

os.makedirs("figures", exist_ok=True)


# Prep data
df = pd.read_csv(CSV_PATH)

target_cols = [col for col in df.columns if col.startswith(TARGET_PREFIX)]
if not target_cols:
    raise ValueError(f"No target columns found with prefix '{TARGET_PREFIX}'")

leakage_cols = [col for col in df.columns if col.startswith(LEAKAGE_PREFIX)]

# Turn one-hot target columns into one class label per row
y_str = (
    df[target_cols]
    .idxmax(axis=1)
    .str.replace(TARGET_PREFIX, "", regex=False)
)

# Remove target columns, leakage columns, and ID column from X
drop_cols = target_cols + leakage_cols
if ID_COL in df.columns:
    drop_cols.append(ID_COL)

X = df.drop(columns=drop_cols).copy()

# Convert string labels to integers
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
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)

    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


# 90/10 split
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


# Compute balanced sample weights
# Rare classes get larger weights, common classes get smaller weights
xgb_train_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# Train XGBoost
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train,
    sample_weight=xgb_train_weights
)


# ── Train and Test Error ─────────────────────────────────────────────────────
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_error = 1 - train_acc
test_error = 1 - test_acc

print(f"\n{'='*50}")
print(f" XGBoost — Train & Test Error")
print(f"{'='*50}")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test  Accuracy: {test_acc:.4f}")
print(f"  Train Error:    {train_error:.4f}")
print(f"  Test  Error:    {test_error:.4f}")


# Evaluate on test set (full report)
print_metrics(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="XGBoost | 90/10 Split | Test Results"
)

save_confusion_matrix(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="XGBoost | 90/10 Split | Test Confusion Matrix",
    filename="figures/xgb_90_10_test_confusion_matrix.png"
)

print("\nDone. Saved:")
print("figures/xgb_90_10_test_confusion_matrix.png")