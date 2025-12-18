import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from knn_classifier import KNNMaterialClassifier

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

knn = KNNMaterialClassifier()
knn.load("knn_model.pkl")

y_pred = np.array([knn.predict(x) for x in X_val])



# Unknown analysis
unknown_count = np.sum(y_pred == 6)
unknown_ratio = unknown_count / len(y_pred)

print("Unknown samples detected:", unknown_count)

mask = y_pred != 6
if np.any(mask):
    filtered_acc = accuracy_score(y_val[mask], y_pred[mask])
    print(" Accuracy :", filtered_acc)

