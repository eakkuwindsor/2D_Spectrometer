# Going to have to adjust this training model to run in GPU so I can stress test the 
# variations by a bigger fraction; GPU should run more images; thus should be more accurate

import os
import csv
import random
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Config
@dataclass
class CFG:
    data_dir: str = os.path.join("datasets", "train_dataset")
    labels_csv: str = "labels.csv"
    batch_size: int = 32
    epochs: int = 10 # Amount of training            
    lr: float = 1e-3
    val_frac: float = 0.2
    seed: int = 42
    num_workers: int = 0
    model_dir: str = "interferogram_model"
    model_name: str = "interferogram_cnn.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Will most likley do cuda later

cfg = CFG()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(cfg.seed)

# Dataset
class InterferogramDataset(Dataset):
    def __init__(self, data_dir, rows, normalize=True):
        self.data_dir = data_dir
        self.rows = rows
        self.normalize = normalize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = os.path.join(self.data_dir, r["filename"])
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0

        # Per-image normalization (big generalization win)
        if self.normalize:
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        x = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        y = torch.tensor(int(r["material_id"]), dtype=torch.long)
        return x, y

def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))

# Split so train/val have similar class proportions (no sklearn needed)
def stratified_split(rows, val_frac, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["material_id"])].append(r)

    train_rows, val_rows = [], []
    for k, bucket in buckets.items():
        rng.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_frac))
        val_rows.extend(bucket[:n_val])
        train_rows.extend(bucket[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows

# Model
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

# Train / Evaluation
def main():
    csv_path = os.path.join(cfg.data_dir, cfg.labels_csv)
    rows = load_rows(csv_path)

    # ✅ NEW: stratified split + print class counts
    train_rows, val_rows = stratified_split(rows, cfg.val_frac, seed=cfg.seed)
    print("Train counts:", Counter(int(r["material_id"]) for r in train_rows))
    print("Val counts:  ", Counter(int(r["material_id"]) for r in val_rows))

    train_ds = InterferogramDataset(cfg.data_dir, train_rows, normalize=True)
    val_ds   = InterferogramDataset(cfg.data_dir, val_rows, normalize=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = SmallCNN(num_classes=3).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Class-weighted loss (helps if any imbalance) - kinda optional, but makes it simpler
    train_counts = Counter(int(r["material_id"]) for r in train_rows)
    num_classes = 3
    weights = []
    for k in range(num_classes):
        weights.append(1.0 / max(1, train_counts.get(k, 1)))
    weights = torch.tensor(weights, dtype=torch.float32, device=cfg.device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for x, y in val_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[int(t), int(p)] += 1

        return loss_sum / total, correct / total, conf

    def train_one_epoch():
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        return loss_sum / total, correct / total

    os.makedirs(cfg.model_dir, exist_ok=True)
    model_out_path = os.path.join(cfg.model_dir, cfg.model_name)

    print("Device:", cfg.device)
    print("Train samples:", len(train_ds), " Val samples:", len(val_ds))
    print("Class weights:", weights.detach().cpu().numpy())

    best_val_acc = -1.0
    best_conf = None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch()
        va_loss, va_acc, conf = evaluate()

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_conf = conf.clone()
            torch.save(model.state_dict(), model_out_path)

    print(f"\nSaved best model -> {model_out_path} (best val acc = {best_val_acc:.3f})")
    if best_conf is not None:
        print("\nBest-val confusion matrix (rows=true, cols=pred):")
        print(best_conf)

if __name__ == "__main__":
    main()
