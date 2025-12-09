import numpy as np
import joblib
from knn_classifier import KNNMaterialClassifier

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

knn = KNNMaterialClassifier(k=3, weighting="distance")

knn.train(X_train, y_train)

knn.save("knn_model.pkl")

print("Model saved!")
