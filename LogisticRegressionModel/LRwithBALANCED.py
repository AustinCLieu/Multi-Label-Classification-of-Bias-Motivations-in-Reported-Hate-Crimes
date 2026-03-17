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
# X: multi-hot feature matrix
X = df[feature_cols]

# y: convert multi-hot target back to a single label per row
# (each row has exactly one MostSeriousBiasType set to 1)
y_multihot = df[target_cols]
y = y_multihot.idxmax(axis=1).str.replace(target_prefix, '', regex=False)

print(f"Features : {X.shape[1]} columns")
print(f"Samples  : {X.shape[0]}")
print(f"\nTarget class distribution:\n{y.value_counts()}\n")

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# ── Logistic Regression with L2 regularization ───────────────────────────────
# C = inverse regularization strength (smaller C = stronger regularization)
model = LogisticRegression(
    penalty='l2',         # L2 (Ridge) regularization
    C=1.0,                # regularization strength (tune as needed)
    solver='lbfgs',       # efficient for multiclass
    #multi_class='auto',
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

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