import os
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH = os.path.join("interferogram_model", "interferogram_cnn.pth")
PRED_DIR   = os.path.join("datasets", "prediction_dataset")
LABELS_CSV = os.path.join(PRED_DIR, "labels.csv")
OUT_CSV    = os.path.join(PRED_DIR, "results.csv")

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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH} (train first)")
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing labels: {LABELS_CSV} (generate prediction dataset first)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(LABELS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    total = 0
    correct = 0

    with open(OUT_CSV, "w", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow([
            "filename","true_material","true_id",
            "pred_material","pred_id","correct",
            "prob_BK7","prob_FS","prob_CaF2"
        ])

        for r in rows:
            fname = r["filename"]
            true_mat = r["material"]
            true_id = int(r["material_id"])
            img_path = os.path.join(PRED_DIR, fname)

            x = load_image_tensor(img_path).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = int(np.argmax(probs))
                pred_mat = ID_TO_MATERIAL[pred_id]

            is_correct = int(pred_id == true_id)
            total += 1
            correct += is_correct

            print(f"{fname} | true={true_mat} pred={pred_mat} {'✓' if is_correct else '✗'}")

            w.writerow([fname, true_mat, true_id, pred_mat, pred_id, is_correct,
                        float(probs[0]), float(probs[1]), float(probs[2])])

    acc = correct / total if total else 0.0
    print(f"\nTotal: {total}  Correct: {correct}  Accuracy: {acc:.3f}")
    print("Wrote ->", OUT_CSV)

if __name__ == "__main__":
    main()
