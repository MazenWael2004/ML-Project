import joblib
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

class SVMClassifier:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True
        )

    def train_model(self, X_train, y_train):
        print("Training SVM Classifier...")
        with tqdm(total=1, desc="Training SVM") as pbar:
            pbar.set_description("Fitting SVM Model")
            self.model.fit(X_train, y_train)
            pbar.update(1)
        print("SVM Training Complete!")

    def predict_data(self, x):
        # Get decision function value
        decision_value = self.model.decision_function([x])[0]
        probas = self.model.predict_proba([x])[0]
        max_proba = np.max(probas)

        # Unknown class threshold
        if max_proba < 0.6:
            return 6  # Unknown class ID

        # Normal prediction
        return self.model.predict([x])[0]


    def save_model_on_disk(self, path="svm_model.pkl"):
        model_data = {
            'model': self.model,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
        joblib.dump(model_data, path)

    def load_model_from_disk(self, path="svm_model.pkl"):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.C = model_data['C']
        self.kernel = model_data['kernel']
        self.gamma = model_data['gamma']
