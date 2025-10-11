
"""
train.py (modified)
Saves per-epoch checkpoints, history (json + csv), and exports a TorchScript model for future inference.
"""
import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
import json
import csv
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------
# Small utilities & transforms
# -----------------------
def pil_load_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    def __call__(self, img):
        return img.resize(self.size, resample=Image.BILINEAR)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation:
    def __init__(self, degrees):
        self.deg = degrees
    def __call__(self, img):
        angle = random.uniform(-self.deg, self.deg)
        return img.rotate(angle, resample=Image.BILINEAR)

class ColorJitterSimple:
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0):
        self.b = brightness
        self.c = contrast
        self.s = saturation
    def __call__(self, img):
        if self.b > 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(1 - self.b, 1 + self.b))
        if self.c > 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(1 - self.c, 1 + self.c))
        if self.s > 0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(1 - self.s, 1 + self.s))
        return img

class ToTensorNormalize:
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.mean = np.array(mean).reshape(3,1,1)
        self.std = np.array(std).reshape(3,1,1)
    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3
        arr = arr.transpose(2,0,1)  # C,H,W
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).float()

# -----------------------
# Dataset (custom ImageFolder)
# -----------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        classes = [p.name for p in sorted(self.root.iterdir()) if p.is_dir()]
        if not classes:
            raise RuntimeError(f"No class subfolders found in {root_dir}. Expected subfolders per class.")
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for c in classes:
            folder = self.root / c
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    self.samples.append((str(img_path), self.class_to_idx[c]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}. Expected subfolders with images.")
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = pil_load_rgb(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------
# Lightweight CNN
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            #Detects edges, color gradients 
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #Detects small textures & leaf veins
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #Detects lesion textures & shapes
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #Condenses each feature map to one scalar value
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------
# Training/eval helpers
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Eval", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    acc = correct / total if total>0 else 0.0
    return running_loss / total if total>0 else 0.0, acc, all_preds, all_labels

def plot_confusion_matrix(cm, class_names, outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2. if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

def plot_training_curves(history, outpath):
    epochs = list(range(1, len(history['train_loss'])+1))
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(epochs, history['train_loss'], label='train_loss')
    ax[0].plot(epochs, history['val_loss'], label='val_loss')
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, [x*100 for x in history['train_acc']], label='train_acc')
    ax[1].plot(epochs, [x*100 for x in history['val_acc']], label='val_acc')
    ax[1].set_title("Accuracy (%)")
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

# -----------------------
# Persistence helpers
# -----------------------
def save_history_json(history, outpath):
    with open(outpath, 'w') as f:
        json.dump(history, f, indent=2)

def save_history_csv(history, outpath):
    # history keys: train_loss, val_loss, train_acc, val_acc
    epochs = len(history['train_loss'])
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        for e in range(epochs):
            w.writerow([e+1,
                        history['train_loss'][e],
                        history['val_loss'][e],
                        history['train_acc'][e],
                        history['val_acc'][e]])

def save_checkpoint(state, outpath):
    torch.save(state, outpath)

def export_torchscript(model, example_input, outpath):
    # attempts scripting first, falls back to tracing
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example_input)
    scripted.save(outpath)

# -----------------------
# Main
# -----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # transforms
    train_transform = lambda img: ToTensorNormalize()(ColorJitterSimple(0.2,0.2,0.2)(
                        RandomRotation(20)(RandomHorizontalFlip(0.5)(Resize((224,224))(img)))))
    val_transform = lambda img: ToTensorNormalize()(Resize((224,224))(img))

    # paths
    train_dir = Path(args.data_root) / "Train"
    val_dir = Path(args.data_root) / "Validation"
    test_dir = Path(args.data_root) / "Test"

    # checks
    for p in (train_dir, val_dir, test_dir):
        if not p.exists():
            raise FileNotFoundError(f"Expected directory not found: {p}\nMake sure --data_root points to the folder that contains Train/Validation/Test")
    print("Loading datasets from:", args.data_root)

    train_ds = ImageFolderDataset(str(train_dir), transform=train_transform)
    val_ds = ImageFolderDataset(str(val_dir), transform=val_transform)
    test_ds = ImageFolderDataset(str(test_dir), transform=val_transform)

    print("Classes:", train_ds.classes)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_acc = 0.0
    os.makedirs(args.experiments_dir, exist_ok=True)
    ckpt_best = os.path.join(args.experiments_dir, "best_model.pth")
    ckpt_last = os.path.join(args.experiments_dir, "last_model.pth")
    history_json = os.path.join(args.experiments_dir, "history.json")
    history_csv = os.path.join(args.experiments_dir, "history.csv")
    curves_path = os.path.join(args.experiments_dir, "training_curves.png")
    ts_path = os.path.join(args.experiments_dir, "model_scripted.pt")

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}%")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc*100:.2f}%")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save per-epoch "last" checkpoint and also best checkpoint when improved
        epoch_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'classes': train_ds.classes,
            'history': history
        }
        save_checkpoint(epoch_ckpt, ckpt_last)
        print(f"Saved last checkpoint to {ckpt_last}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(epoch_ckpt, ckpt_best)
            print(f"New best val acc: {best_val_acc*100:.2f}%, saved best checkpoint to {ckpt_best}")

        # save history json & csv each epoch (useful if job dies)
        save_history_json(history, history_json)
        save_history_csv(history, history_csv)

    # save training curves
    plot_training_curves(history, curves_path)
    print("Saved training curves to", curves_path)

    # --- test ---
    if os.path.exists(ckpt_best):
        print("Loading best checkpoint for test evaluation.")
        ckpt = torch.load(ckpt_best, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("Best checkpoint not found, using last checkpoint for test.")
        ckpt = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print("\n=== TEST REPORT ===")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(args.experiments_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, test_ds.classes, cm_path)
    print("Saved confusion matrix to", cm_path)

    # Export TorchScript for future inference (use the same input size as training)
    example = torch.randn(1, 3, 224, 224).to(device)
    try:
        export_torchscript(model.cpu(), example.cpu(), ts_path)
        print("Exported TorchScript model to", ts_path)
    except Exception as e:
        print("TorchScript export failed:", e)

    # also write a small metadata file for deployment
    metadata = {
        'classes': train_ds.classes,
        'input_size': [3, 224, 224],
        'best_val_acc': best_val_acc,
        'epochs': args.epochs
    }
    with open(os.path.join(args.experiments_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata.json")

# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/Users/dasaradha/Coding/4-1 mini project/Wheat Disease/Split Dataset",
                        help="Path to dataset root that contains Train/, Validation/, Test/")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--experiments_dir", type=str, default="../experiments")
    args = parser.parse_args()
    main(args)
