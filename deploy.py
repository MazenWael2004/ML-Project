import cv2
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# Paths

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "best_models"

DENSENET_PATH = MODELS_DIR / "best_densenet201.pth"
SVM_PATH = MODELS_DIR / "best_svm_model.pkl"   
KNN_PATH = MODELS_DIR / "best_knn_model.pkl"   
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load DenseNet-201 Feature Extractor

from torch import nn
NUM_CLASSES = 6   
cnn_model = models.densenet201(weights=None)
cnn_model.classifier = nn.Linear(1920, NUM_CLASSES)
cnn_model = cnn_model.to(device)
state_dict = torch.load(DENSENET_PATH, map_location=device)
cnn_model.load_state_dict(state_dict)
cnn_model.classifier = nn.Identity()
cnn_model.eval()

print("✅ DenseNet-201 loaded")


# Preprocessing 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load ML model 

USE_SVM = True   #false --> knn

if USE_SVM:
    svm_obj = joblib.load(SVM_PATH)
    svm_model = svm_obj["model"]   
    print("✅ SVM model loaded")
else:
    model = joblib.load(KNN_PATH)
    print("✅ KNN model loaded")

scaler = joblib.load(SCALER_PATH)
print("✅ Feature scaler loaded")
LE_PATH = MODELS_DIR / "label_encoder.pkl"
label_encoder = joblib.load(LE_PATH)
print("✅ Label Encoder loaded")





# Feature Extraction

def extract_cnn_features_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = cnn_model(img_tensor)

    features = features.view(features.size(0), -1)
    return features.cpu().numpy()  # (1, 1920)

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Deployment started — press Q to quit")

THRESHOLD = 0.2   # Unknown threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Feature extraction
    features = extract_cnn_features_from_frame(frame)
    features = scaler.transform(features)

    # Prediction with Unknown logic
    probs = svm_model.predict_proba(features)
    max_prob = np.max(probs)
    print("Max prob:", max_prob)


    if max_prob < THRESHOLD:
        label = "Unknown"
        color = (0, 0, 255)
    else:
        pred = np.argmax(probs)    
        label = label_encoder.inverse_transform([pred])[0]
        color = (0, 255, 0)

    # Display
    cv2.putText(
        frame,
        f"Prediction: {label} ({max_prob:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Material Classification Deployment", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
