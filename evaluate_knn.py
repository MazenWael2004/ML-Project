import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from knn_classifier import KNNMaterialClassifier

# 1) Load validation data
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# 2) Load trained KNN model
knn = KNNMaterialClassifier()
knn.load("knn_model.pkl")

# 3) Predict
y_pred = []
for x in X_val:
    y_pred.append(knn.predict(x))

# 4) Compute accuracy
acc = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", acc)
