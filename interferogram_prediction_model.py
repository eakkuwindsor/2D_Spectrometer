import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH = os.path.join("interferogram_model", "interferogram_cnn.pth")
ID_TO_MATERIAL = {0: "BK7", 1: "FS", 2: "CaF2"}

class SmallCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_image_tensor(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python interferogram_prediction_model.py path/to/image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH} (train first)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SmallCNN(num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    x = load_image_tensor(image_path).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    print(f"Loaded model: {MODEL_PATH}")
    print("Prediction:", ID_TO_MATERIAL.get(pred_id, f"UnknownID({pred_id})"))
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {ID_TO_MATERIAL.get(i, f'ID{i}')}: {p:.4f}")

if __name__ == "__main__":
    main()
# python interferogram_prediction_model.py datasets/prediction_dataset/test_010.png