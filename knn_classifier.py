import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNMaterialClassifier:
    def __init__(self, k=11, weighting="distance", unknown_threshold=0.6):
        self.k = k
        self.unknown_threshold = unknown_threshold
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weighting,
            metric="euclidean"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, x):
        probs = self.model.predict_proba([x])[0]
        max_prob = np.max(probs)

        # Unknown class decision
        if max_prob < self.unknown_threshold:
            return 6  # Unknown

        return np.argmax(probs)

    def save(self, path="knn_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="knn_model.pkl"):
        self.model = joblib.load(path)
