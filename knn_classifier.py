import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNMaterialClassifier:
    def __init__(self, k=30, weighting="distance"):
        self.k = k
        self.weighting = weighting
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weighting,
            metric="euclidean"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, x):
        # Get distances to neighbors
        dist, idx = self.model.kneighbors([x], n_neighbors=self.k)
        dist = dist[0]  # distances list

        min_d = np.min(dist)

        # Unknown class threshold
        if min_d > 0.9:
            return 6  # Unknown class ID

        # Normal prediction
        return self.model.predict([x])[0]

    def save(self, path="knn_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="knn_model.pkl"):
        self.model = joblib.load(path)
