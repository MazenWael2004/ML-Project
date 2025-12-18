import numpy as np
from knn_classifier import KNNMaterialClassifier

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

knn = KNNMaterialClassifier(
    k=1,
    weighting="distance",
    unknown_threshold=0.45
)

knn.train(X_train, y_train)
knn.save("knn_model.pkl")

print(" KNN model trained and saved successfully")
