import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from pathlib import Path
import cv2
import numpy as np

# Constants matching the training configuration
CATEGORIES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
UNKNOWN_CLASS_ID = 6  # Predicted when confidence is low

NUM_CLASSES = len(CATEGORIES) + 1  # 6 known + 1 unknown slot
IMG_SIZE = 224
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(dataFilePath, bestModelPath):
    """
    Predict waste material categories from images.

    Args:
        dataFilePath: Path to folder containing test images
        bestModelPath: Path to the trained classifier model (SVM or KNN)

    Returns:
        List of integer ID labels (0-5 for categories, 6 for unknown)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the DenseNet-201 model for feature extraction
    print("Loading DenseNet-201 feature extractor...")
    densenet_model = models.densenet201(weights=None)
    num_features = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(num_features, NUM_CLASSES)  # 7 classes total

    # Load the trained weights from the same directory as the classifier
    model_dir = Path(bestModelPath).parent
    densenet_path = model_dir / 'best_densenet201.pth'

    if densenet_path.exists():
        densenet_model.load_state_dict(torch.load(densenet_path, map_location=device))
        print(f" Loaded DenseNet-201 weights from {densenet_path}")
    else:
        print(f"Warning: DenseNet-201 weights not found at {densenet_path}")
        print("Using random initialization (predictions will be poor)")

    # Create CNN feature extractor (remove classifier)
    cnn_model = nn.Sequential(
        densenet_model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    # Load the classifier (SVM or KNN) - load the full saved object
    print(f"Loading classifier from {bestModelPath}...")

    # Import the classifier classes
    from knn_classifier import KNNMaterialClassifier
    from SVM_classifier import SVMClassifier

    # Determine classifier type from filename and load appropriately
    if 'knn' in str(bestModelPath).lower():
        classifier = KNNMaterialClassifier()
        classifier.load(bestModelPath)
        classifier_type = 'KNN'
        print(f" Loaded KNN classifier (k={classifier.k})")
    elif 'svm' in str(bestModelPath).lower():
        classifier = SVMClassifier()
        classifier.load_model_from_disk(bestModelPath)
        classifier_type = 'SVM'
        print(f" Loaded SVM classifier (kernel={classifier.kernel})")
    else:
        raise ValueError("Could not determine classifier type from filename")

    # Load feature scaler
    scaler_path = model_dir / 'feature_scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f" Loaded feature scaler from {scaler_path}")
    else:
        print(f"Warning: Feature scaler not found at {scaler_path}")
        scaler = None

    # Define preprocessing transforms (same as training validation transform)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get all image paths from the data folder
    data_path = Path(dataFilePath)
    image_paths = sorted([
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])

    print(f"\nFound {len(image_paths)} images in {dataFilePath}")

    # Extract features and make predictions
    predictions = []

    print("Extracting features and making predictions...")
    for img_path in image_paths:
        try:
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path.name}, skipping image")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            # Extract CNN features
            with torch.no_grad():
                features = cnn_model(img_tensor)
                features = features.view(-1).cpu().numpy()

            # Scale features if scaler is available
            if scaler is not None:
                features = scaler.transform([features])[0]

            # Make prediction using the classifier's predict method
            # This handles PCA transformation for KNN internally
            if classifier_type == 'KNN':
                pred_label = classifier.predict(features)
            else:  # SVM
                pred_label = classifier.predict_data(features)

            # Ensure prediction is an integer
            pred_label = int(pred_label)
            predictions.append(pred_label)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}, skipping image")
            continue

    print(f"\n Generated {len(predictions)} predictions")
    print(f"  Prediction distribution: {np.bincount(predictions, minlength=7)}")

    return predictions


if __name__ == "__main__":
    dataPath = "test"
    modelPath = "best_models/best_svm_model.pkl"

    predictions = predict(dataPath, modelPath)

    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        if pred < len(CATEGORIES):
            category = CATEGORIES[pred]
        elif pred == UNKNOWN_CLASS_ID:
            category = "unknown"
        print(f"Image {i}: {pred} ({category})")
