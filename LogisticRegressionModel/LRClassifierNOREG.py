import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('CA_multihot_seasons.csv')

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define both models ───────────────────────────────────────────────────────
models = {
    'With L2 Regularization (C=1.0)': LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        multi_class='auto',
        max_iter=1000,
        random_state=42
    ),
    'No Regularization': LogisticRegression(
        penalty=None,
        solver='lbfgs',
        multi_class='auto',
        max_iter=1000,
        random_state=42
    ),
}

# ── Train, evaluate, and plot both ───────────────────────────────────────────
labels = sorted(y.unique())
results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {round(acc, 4)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') + '.png'
    plt.savefig(filename, dpi=150)
    plt.show()

# ── Side-by-side accuracy summary ────────────────────────────────────────────
print(f"\n{'='*60}")
print(" Accuracy Summary")
print(f"{'='*60}")
for name, acc in results.items():
    print(f"  {name}: {round(acc, 4)}")

# ── Top feature importances per class (both models) ──────────────────────────
print(f"\n{'='*60}")
print(" Top 5 Features per Class")
print(f"{'='*60}")
for name, model in models.items():
    print(f"\n--- {name} ---")
    coef_df = pd.DataFrame(model.coef_, index=model.classes_, columns=feature_cols)
    for cls in model.classes_:
        top = coef_df.loc[cls].abs().nlargest(5).index.tolist()
        print(f"  {cls}: {top}")