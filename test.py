import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


CATEGORIES = ['cardboard', 'plastic', 'glass', 'metal', 'paper', 'trash']


def predict(dataPath, modelPath):
    # Load the pre-trained model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CATEGORIES))
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare to store results
    results = []

    # Process each image in the dataPath
    for img_name in os.listdir(dataPath):
        img_path = os.path.join(dataPath, img_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            category = CATEGORIES[predicted.item()]

        results.append((img_name, category))

    return results