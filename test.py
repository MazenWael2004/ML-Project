import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib


CATEGORIES = ['cardboard', 'plastic', 'glass', 'metal', 'paper', 'trash']


def predict(dataFilePath, bestModelPath):
    device = torch.device("cpu")

   # CNN FEturee extractor
    cnn = models.resnet50(pretrained=True)
    cnn.fc = nn.Identity()  # remove classifier
    cnn.to(device)
    cnn.eval()

   # Load the best model
    clf = joblib.load(modelPath)  # SVM or KNN

    # imAgee trAnsformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    pred = []

    
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

        # now we predict
        pred_idx = clf.predict([features])[0]
        category = CATEGORIES[pred_idx]

        pred.append((img_name, category))

    return pred
if __name__ == "__main__":
    dataPath = "path/to/test/images"
    modelPath = "path/to/saved/model.pkl"

    predictions = predict(dataPath, modelPath)
    for img_name, category in predictions:
        print(f"{img_name}: {category}")