import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from pathlib import Path
import cv2
import random
import numpy as np
import shutil
from tqdm import tqdm

CATEGORIES = ['cardboard', 'plastic', 'glass', 'metal', 'paper', 'trash']
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(dataFilePath, bestModelPath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")

    IMG_SIZE = 224

    # CNN Feature Extractor
    cnn = models.resnet50(pretrained=True)
    cnn.fc = nn.Identity()
    cnn.to(device)
    cnn.eval()

    # Load classifier
    clf = joblib.load(bestModelPath)  # SVM or KNN

    # Transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    BASE_DIR = Path(dataFilePath)
    DATASET_PATH = Path(dataFilePath)

    # Preprocess images
    pred = []
    for img_path in DATASET_PATH.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".png"}:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = cnn(img_tensor).cpu().numpy()
                category_idx = clf.predict(features)[0]
                category_name = CATEGORIES[category_idx]
                pred.append((img_path.name, category_name))

    return pred


if __name__ == "__main__":
    dataPath = "path/to/test/images"
    modelPath = "path/to/saved/model.pkl"

    predictions = predict(dataPath, modelPath)
    for img_name, category in predictions:
        print(f"{img_name}: {category}")
    