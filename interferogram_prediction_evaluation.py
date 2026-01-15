import os
import csv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("interferogram_model", "interferogram_cnn.pth")
PRED_DIR   = os.path.join("datasets", "prediction_dataset")
LABELS_CSV = os.path.join(PRED_DIR, "labels.csv")

# New output folder inside prediction_dataset
RESULTS_DIR = os.path.join(PRED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_CSV   = os.path.join(RESULTS_DIR, "results.csv")
ACC_PNG   = os.path.join(RESULTS_DIR, "accuracy_bar.png")
PROB_PNG  = os.path.join(RESULTS_DIR, "prob_histograms.png")
MIS_PNG   = os.path.join(RESULTS_DIR, "misclassified_grid.png")

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

def load_image_tensor(path, normalize=True):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Match training
    if normalize:
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def plot_accuracy_bars(overall_acc: float, per_class_acc: np.ndarray, class_names, out_path: str):
    labels = ["Overall"] + list(class_names)
    values = [overall_acc] + per_class_acc.tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy")
    ax.set_ylabel("Accuracy (0–1)")

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_prob_histograms(probs_max: np.ndarray, correct_mask: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs_max[correct_mask], bins=20, alpha=0.7, label="Correct")
    ax.hist(probs_max[~correct_mask], bins=20, alpha=0.7, label="Incorrect")
    ax.set_title("Model Confidence (max softmax probability)")
    ax.set_xlabel("Max Probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_misclassified_grid(mis_paths, mis_titles, out_path: str, max_images=12):
    if len(mis_paths) == 0:
        return
    n = min(max_images, len(mis_paths))
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = np.array(axes).reshape(1, -1)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis("off")

        if idx >= n:
            continue

        img = Image.open(mis_paths[idx]).convert("L")
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(mis_titles[idx], fontsize=9)

    fig.suptitle("Misclassified Examples", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

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

    num_classes = 3
    per_class_total = np.zeros(num_classes, dtype=int)
    per_class_correct = np.zeros(num_classes, dtype=int)

    per_sample_maxprob = []
    per_sample_correct = []

    mis_paths = []
    mis_titles = []

    with open(OUT_CSV, "w", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow([
            "filename","true_material","true_id",
            "pred_material","pred_id","correct",
            "prob_BK7","prob_FS","prob_CaF2"
        ])

        for i, r in enumerate(rows):
            fname = r["filename"]
            true_mat = r["material"]
            true_id = int(r["material_id"])
            img_path = os.path.join(PRED_DIR, fname)

            x = load_image_tensor(img_path, normalize=True).to(device)

            # Optional sanity check for first few images
            if i < 3:
                x_cpu = x.detach().cpu().numpy()
                print(f"[debug] {fname} tensor mean={x_cpu.mean():.4f} std={x_cpu.std():.4f}")

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = int(np.argmax(probs))
                pred_mat = ID_TO_MATERIAL[pred_id]

            is_correct = int(pred_id == true_id)
            total += 1
            correct += is_correct

            per_class_total[true_id] += 1
            per_class_correct[true_id] += is_correct

            per_sample_maxprob.append(float(np.max(probs)))
            per_sample_correct.append(bool(is_correct))

            if not is_correct:
                mis_paths.append(img_path)
                mis_titles.append(f"{fname}\ntrue={true_mat} pred={pred_mat}")

            print(f"{fname} | true={true_mat} pred={pred_mat} {'✓' if is_correct else '✗'}")

            w.writerow([
                fname, true_mat, true_id,
                pred_mat, pred_id, is_correct,
                float(probs[0]), float(probs[1]), float(probs[2])
            ])

    acc = correct / total if total else 0.0
    per_class_acc = np.array([
        (per_class_correct[i] / per_class_total[i]) if per_class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ], dtype=float)

    class_names = [ID_TO_MATERIAL[i] for i in range(num_classes)]

    print(f"\nTotal: {total}  Correct: {correct}  Accuracy: {acc:.3f}")
    print("Wrote ->", OUT_CSV)

    # Save plots into results/
    plot_accuracy_bars(acc, per_class_acc, class_names, ACC_PNG)

    probs_max = np.array(per_sample_maxprob, dtype=float)
    correct_mask = np.array(per_sample_correct, dtype=bool)
    plot_prob_histograms(probs_max, correct_mask, PROB_PNG)

    plot_misclassified_grid(mis_paths, mis_titles, MIS_PNG, max_images=12)

    print("\nSaved in ->", RESULTS_DIR)
    print(" ", ACC_PNG)
    print(" ", PROB_PNG)
    if len(mis_paths) > 0:
        print(" ", MIS_PNG)
    else:
        print(" (No misclassifications, so no misclassified_grid.png)")

if __name__ == "__main__":
    main()
