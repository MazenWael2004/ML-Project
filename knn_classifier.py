import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNMaterialClassifier:
    def __init__(self, k=7, weighting="distance"):
        self.k = k
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weighting,
            metric="euclidean"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, x):
        dist, _ = self.model.kneighbors([x], n_neighbors=self.k)
        min_d = np.min(dist)

        # Unknown threshold (NOT too aggressive)
        if min_d > 1.2:
            return 6  # Unknown

        return self.model.predict([x])[0]

    def save(self, path="knn_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="knn_model.pkl"):
        self.model = joblib.load(path)
