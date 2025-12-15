import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from SVM_classifier import SVMClassifier

# 1) Load validation data
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")


# 2) Load trained SVM model
svm = SVMClassifier()
svm.load_model_from_disk("svm_model.pkl")


# Now we preidcct the loadedd model
y_pred = []
for x in X_val:
    y_pred.append(svm.predict_data(x))
# 4) Compute accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

