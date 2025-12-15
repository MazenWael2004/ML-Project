import joblib
import numpy as NP
from sklearn.svm import svc

class SVMClassifier:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = svc(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True
        )

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_data(self, x):
        # Get decision function value
        decision_value = self.model.decision_function([x])[0]
        probas = self.model.predict_proba([x])[0]
        max_proba = NP.max(probas)

        # Unknown class threshold
        if max_proba < 0.6:
            return 6  # Unknown class ID

        # Normal prediction
        return self.model.predict([x])[0]

    def save_model_on_disk(self, path="svm_model.pkl"):
        joblib.dump(self.model, path)

    def load_model_from_disk(self, path="svm_model.pkl"):
        self.model = joblib.load(path)