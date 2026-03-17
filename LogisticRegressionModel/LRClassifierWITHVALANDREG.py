import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# ── Train / val / test split (80/10/10) ──────────────────────────────────────
# First split off 20% for val+test, then split that 50/50
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train size : {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Val size   : {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test size  : {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

# ── Logistic Regression with L2 regularization ───────────────────────────────
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# ── Validation performance ───────────────────────────────────────────────────
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"\n{'='*60}")
print(" Validation Results")
print(f"{'='*60}")
print(f"Validation Accuracy: {round(val_acc, 4)}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# ── Test performance ─────────────────────────────────────────────────────────
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n{'='*60}")
print(" Test Results")
print(f"{'='*60}")
print(f"Test Accuracy: {round(test_acc, 4)}")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# ── Accuracy summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(" Accuracy Summary")
print(f"{'='*60}")
print(f"  Val  Accuracy: {round(val_acc, 4)}")
print(f"  Test Accuracy: {round(test_acc, 4)}")

# ── Confusion matrix (test set) ───────────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_test_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Test) — L2 Regularization')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_l2.png', dpi=150)
plt.show()

# ── Top 5 feature importances per class ──────────────────────────────────────
print(f"\n{'='*60}")
print(" Top 5 Features per Class")
print(f"{'='*60}")
coef_df = pd.DataFrame(model.coef_, index=model.classes_, columns=feature_cols)
for cls in model.classes_:
    top = coef_df.loc[cls].abs().nlargest(5).index.tolist()
    print(f"  {cls}: {top}")