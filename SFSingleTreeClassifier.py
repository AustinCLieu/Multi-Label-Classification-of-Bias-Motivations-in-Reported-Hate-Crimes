"""
Single tree output that predicts the combinations of the possible bias types because it's multihotencoded.
For example, you can have a bias type of both race and religion. This would combine it into one output category or one Y value.
This is called a label powerset.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Visualization
from sklearn import tree
import matplotlib.pyplot as plt

DATA_PATH = "Data/SF_merged_data_multihot.csv"
RANDOM_STATE = 42 # Hitchhiker's guide to the galaxy. Used for train test split

dataframe = pd.read_csv(DATA_PATH)

# Feature columns (X): Race + Location
race_cols = []
location_cols = []
for col_name in dataframe.columns:
    # Keep columns like "SuspectsRaceASAGroup_white", etc
    if col_name.startswith("SuspectsRaceAsAGroup__"):
        race_cols.append(col_name)

for col_name in dataframe.columns:
    # Keep columns like "MostSeriousLocation_street", etc
    if col_name.startswith("MostSeriousLocation__"):
        location_cols.append(col_name)

X_cols = []
for c in race_cols:
    X_cols.append(c)
for c in location_cols:
    X_cols.append(c)

X = dataframe[X_cols].copy()

# Target/Output columns (Y): bias type label powerset of multihot encoded columns
y_cols = []
for col_name in dataframe.columns:
    if col_name.startswith("MostSeriousBiasType__"):
        y_cols.append(col_name)

Y = dataframe[y_cols].copy()

# Convert multihot encoded Y into a single multiclass label per row
labels = [] # stores final single label targets for each row

# Loop through 
for i in range(len(Y)):
    active = []
    # Check every bias type column for this row
    for col_name in y_cols:
        val = Y.at[i, col_name]
        # If the column is on (1), then this label is present
        if val == 1 or val is True:
            suffix = col_name.replace("MostSeriousBiasType__", "")
            active.append(suffix)
    # Make the label stable by sorting the parts
    active.sort()
    # Join into one label string
    joined = "|".join(active)
    labels.append(joined)
y_label = pd.Series(labels)

# Drop rows where target is missing (no bias type)
keep_rows = []
for i in range(len(y_label)):
    # keep the row if label is not empty
    if y_label.iloc[i] != "":
        keep_rows.append(True)
    else:
        keep_rows.append(False)

mask = pd.Series(keep_rows)

X = X[mask].reset_index(drop = True)
y_label = y_label[mask].reset_index(drop = True)

print("X shape:", X.shape)
print("Number of classes:", y_label.nunique())
print("Top 10 classes:\n", y_label.value_counts().head(10))

# Splitting of train and test data
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_label,
        test_size = 0.2,
        random_state = RANDOM_STATE,
        stratify = y_label
    )
except ValueError:
    # fallback: no stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_label,
        test_size = 0.2,
        random_state = RANDOM_STATE
    )

# Train one decision tree
clf = DecisionTreeClassifier(
    max_depth = 12,
    min_samples_leaf = 5,
    random_state = RANDOM_STATE,
    class_weight = "balanced"
)

clf.fit(X_train, y_train)

"""
# Plot only the top few levels so it’s readable
plt.figure(figsize = (28, 14))
tree.plot_tree(
    clf,
    feature_names = X_cols,     # feature column names
    class_names = clf.classes_, # class labels
    filled = True,
    rounded = True,
    # max_depth = 3,              # show only top 3 levels
    max_depth = 5,
    fontsize = 8
)

plt.tight_layout()
plt.savefig("decision_tree_top5.png", dpi = 300)
plt.show()
print("Saved decision_tree_top3.png")
"""


# Evaluate
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))