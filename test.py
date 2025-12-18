import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib


CATEGORIES = ['cardboard', 'plastic', 'glass', 'metal', 'paper', 'trash']


def predict(dataPath, modelPath):
    device = torch.device("cpu")

    # -------------------------------
    # 1. CNN FEATURE EXTRACTOR
    # -------------------------------
    cnn = models.resnet50(pretrained=True)
    cnn.fc = nn.Identity()  # remove classifier
    cnn.to(device)
    cnn.eval()

    # -------------------------------
    # 2. LOAD SVM / KNN MODEL
    # -------------------------------
    clf = joblib.load(modelPath)  # SVM or KNN

    # -------------------------------
    # 3. IMAGE TRANSFORMS
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    results = []

    # -------------------------------
    # 4. PREDICTION LOOP
    # -------------------------------
    for img_name in sorted(os.listdir(dataPath)):
        img_path = os.path.join(dataPath, img_name)

        if not os.path.isfile(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            features = cnn(image)
            features = features.squeeze().cpu().numpy()

        # Predict using SVM / KNN
        pred_idx = clf.predict([features])[0]
        category = CATEGORIES[pred_idx]

        results.append((img_name, category))

    return results
if __name__ == "__main__":
    dataPath = "path/to/test/images"
    modelPath = "path/to/saved/model.pkl"

    predictions = predict(dataPath, modelPath)
    for img_name, category in predictions:
        print(f"{img_name}: {category}")