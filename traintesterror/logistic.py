import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

print(f"Features : {X.shape[1]} columns")
print(f"Samples  : {X.shape[0]}")
print(f"\nTarget class distribution:\n{y.value_counts()}\n")

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

# ── Logistic Regression with L2 regularization ───────────────────────────────
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_error = 1 - train_acc
test_error = 1 - test_acc

print(f"\n{'='*50}")
print(f" Logistic Regression (L2) — Train & Test Error")
print(f"{'='*50}")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test  Accuracy: {test_acc:.4f}")
print(f"  Train Error:    {train_error:.4f}")
print(f"  Test  Error:    {test_error:.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_test_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — MostSeriousBiasType')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── Top feature importances per class ────────────────────────────────────────
coef_df = pd.DataFrame(model.coef_, index=model.classes_, columns=feature_cols)

print("\nTop 5 most influential features per class:")
for cls in model.classes_:
    top = coef_df.loc[cls].abs().nlargest(5).index.tolist()
    print(f"  {cls}: {top}")