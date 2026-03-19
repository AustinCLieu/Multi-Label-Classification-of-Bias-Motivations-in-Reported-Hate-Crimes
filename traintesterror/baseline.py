"""
Majority Class Baseline — outputs train and test error.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('Data/CA_multihot_seasons.csv')

# ── Define feature groups to INCLUDE ────────────────────────────────────────
feature_prefixes = [
    'SuspectsRaceAsAGroup__',
    'MostSeriousLocation__',
    'MostSeriousUCR__',
    'TimeOfYear__',
]

feature_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
target_prefix = 'MostSeriousBiasType__'
target_cols = [c for c in df.columns if c.startswith(target_prefix)]

# ── Build X and y ────────────────────────────────────────────────────────────
X = df[feature_cols]
y_multihot = df[target_cols]
y = y_multihot.idxmax(axis=1).str.replace(target_prefix, '', regex=False)

# ── Train / test split (90/10) ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

# ── Majority Class Baseline ──────────────────────────────────────────────────
majority_class = y_train.value_counts().idxmax()
print(f"\nMajority class: {majority_class}")

# Predict majority class for all samples
y_train_pred = [majority_class] * len(y_train)
y_test_pred = [majority_class] * len(y_test)

# ── Train and Test Error ─────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_error = 1 - train_acc
test_error = 1 - test_acc

print(f"\n{'='*50}")
print(f" Majority Class Baseline — Train & Test Error")
print(f"{'='*50}")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test  Accuracy: {test_acc:.4f}")
print(f"  Train Error:    {train_error:.4f}")
print(f"  Test  Error:    {test_error:.4f}")