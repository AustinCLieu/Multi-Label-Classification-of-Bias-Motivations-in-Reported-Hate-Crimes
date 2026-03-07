"""
Single tree output that predicts the combinations of the possible bias types because it's multihotencoded.
For example, you can have a bias type of both race and religion. This would combine it into one output category or one Y value.
This is called a label powerset.
"""
from sklearn.tree import DecisionTreeClassifier
