import numpy as np
from sklearn.metrics import accuracy_score
from knn_classifier import KNNMaterialClassifier

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

knn = KNNMaterialClassifier()
knn.load("knn_model.pkl")

y_pred = np.array([knn.predict(x) for x in X_val])

#  IGNORE UNKNOWN CLASS
mask = y_pred != 6

filtered_y_val = y_val[mask]
filtered_y_pred = y_pred[mask]

acc = accuracy_score(filtered_y_val, filtered_y_pred)

print("Validation Accuracy (Primary Classes 0–5 only):", acc)
print("Unknown samples detected:", np.sum(y_pred == 6))
