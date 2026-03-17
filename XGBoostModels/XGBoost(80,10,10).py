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


# 80/10/10 split
# First split off the final test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y
)

# Split the remaining 90% into train and validation
# test_size=1/9 gives 80% train and 10% validation overall
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=1/9,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

print("\nSplit sizes:")
print("Train:", X_train.shape, len(y_train))
print("Val:  ", X_val.shape, len(y_val))
print("Test: ", X_test.shape, len(y_test))


# compute balanced sample weights
# These weights make rare classes count more during training
xgb_train_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# Different xgboost settings
# Use validation macro F1 to choose the best model
xgb_candidates = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.10},
]

best_xgb_score = -1
best_xgb_params = None
best_xgb_model = None

for params in xgb_candidates:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=xgb_train_weights
    )

    y_val_pred = model.predict(X_val)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"\nXGB candidate {params} -> Validation Macro F1 = {val_macro_f1:.4f}")

    if val_macro_f1 > best_xgb_score:
        best_xgb_score = val_macro_f1
        best_xgb_params = params
        best_xgb_model = model

print("\nBest XGB params:", best_xgb_params)
print("Best validation Macro F1:", round(best_xgb_score, 4))


# Evaluate on validation set
y_val_pred = best_xgb_model.predict(X_val)

print_metrics(
    y_true=y_val,
    y_pred=y_val_pred,
    class_names=class_names,
    title="XGBoost | 80/10/10 Split | Validation Results"
)

save_confusion_matrix(
    y_true=y_val,
    y_pred=y_val_pred,
    class_names=class_names,
    title="XGBoost | 80/10/10 Split | Validation Confusion Matrix",
    filename="figures/xgb_80_10_10_val_confusion_matrix.png"
)


# Evaluate on test set
y_test_pred = best_xgb_model.predict(X_test)

print_metrics(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="XGBoost | 80/10/10 Split | Test Results"
)

save_confusion_matrix(
    y_true=y_test,
    y_pred=y_test_pred,
    class_names=class_names,
    title="XGBoost | 80/10/10 Split | Test Confusion Matrix",
    filename="figures/xgb_80_10_10_test_confusion_matrix.png"
)

print("\nDone. Saved:")
print("figures/xgb_80_10_10_val_confusion_matrix.png")
print("figures/xgb_80_10_10_test_confusion_matrix.png")