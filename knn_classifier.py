import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

class KNNMaterialClassifier:
    def __init__(self, k=11, weighting="distance", n_components=200, unknown_threshold=0.3):
        self.k = k
        self.weighting = weighting
        self.n_components = n_components
        self.unknown_threshold = unknown_threshold
        self.pca = PCA(n_components=n_components)
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weighting,
            metric="euclidean"
        )

    def train(self, X_train, y_train):
        print("Training KNN Classifier...")
        with tqdm(desc="KNN Training") as pbar:
            pbar.set_description("Fitting PCA")
            X_train_reduced = self.pca.fit_transform(X_train)
            pbar.update(1)

            pbar.set_description("Fitting KNN Model")
            self.model.fit(X_train_reduced, y_train)
            pbar.update(1)
        print("KNN Training Complete!")

    def predict(self, x):
        x_reduced = self.pca.transform([x])
        probs = self.model.predict_proba(x_reduced)[0]
        max_prob = np.max(probs)

        # Unknown class decision
        if max_prob < self.unknown_threshold:
            return 6  # Unknown

        return np.argmax(probs)

    def save(self, path="knn_model.pkl"):
        model_data = {
            'model': self.model,
            'pca': self.pca,
            'k': self.k,
            'weighting': self.weighting,
            'n_components': self.n_components,
            'unknown_threshold': self.unknown_threshold
        }
        joblib.dump(model_data, path)

    def load(self, path="knn_model.pkl"):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.pca = model_data['pca']
        self.k = model_data['k']
        self.weighting = model_data['weighting']
        self.n_components = model_data['n_components']
        self.unknown_threshold = model_data['unknown_threshold']
