# ================== 20-class Low-Res Test Adapted Training Script ==================
# Colab + A100 友好：ResNet18 预训练，AMP，CosineLR，低清晰度模拟增强，导出 submission.csv
# 目录假设：
#   data_root/
#     train/  (按子文件夹=类别组织)
#     test/   (无标签，所有测试图片放这里)
# 如你的加载器已就绪，也可跳过Dataset部分，直接用你的 train_loader/test_loader。

import os, glob, random, math, time
from typing import List
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision.transforms.functional import InterpolationMode


# ------------------ Reproducibility ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # A100上开启加速


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------ Paths (修改为你的路径) ------------------
# data_root = "/content/data_root"  # TODO: 改成你的根目录
data_root = "/Users/2m/Documents/Monash/FIT5215/DL_ASS"  # TODO: 改成你的根目录
train_dir = os.path.join(data_root, "FIT5215_Dataset")
test_dir = os.path.join(data_root, "test_set")

assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"

# ------------------ Transforms ------------------
# 核心思路：训练在224上；随机「降到64/96再升到224」模拟低清晰度（与test域一致）。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def lowres_then_up_transform(p=0.35):
    # 以一定概率把图像降采样到64或96，再升采样回224
    # 用BICUBIC插值更接近自然缩放失真
    down_choices = [64, 96]
    ops = []
    size = random.choice(down_choices)
    ops.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
    ops.append(transforms.Resize(224, interpolation=InterpolationMode.BICUBIC))
    return transforms.RandomApply(ops, p=p)


train_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        lowres_then_up_transform(p=0.35),  # 低清晰度模拟（关键）
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# 验证/测试：把图片送到224；测试集原本是64×64，这里直接放大到224，以匹配预训练分布
test_tf = transforms.Compose(
    [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# ------------------ Datasets & Loaders ------------------
# 训练集（按子文件夹=类别）
full_train = ImageFolder(train_dir, transform=train_tf)
class_names = full_train.classes
num_classes = len(class_names)
print("Classes:", class_names, " | num_classes =", num_classes)

# 可选：从训练数据划分一小部分做验证（例如10%），没有验证也能直接训练
val_ratio = 0.1
val_size = int(len(full_train) * val_ratio)
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(
    full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# 验证集用“干净”评估变换
val_ds.dataset.transform = test_tf


# 测试集（无标签）：按文件名排序，保证可复现
class TestFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.paths = sorted(
            [
                p
                for p in glob.glob(os.path.join(root, "**/*"), recursive=True)
                if os.path.isfile(p)
                and p.lower().split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "webp"]
            ]
        )
        self.transform = transform
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 返回dummy标签（兼容你的save函数签名）
        return img, 0


test_ds = TestFolder(test_dir, transform=test_tf)

# DataLoader（A100可以适当大些）
train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train/Val/Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

# ------------------ Model ------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# 损失/优化/调度
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
num_epochs = 25
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# ------------------ Train & Eval ------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


best_val_acc = -1.0
best_state = None

for ep in range(1, num_epochs + 1):
    model.train()
    run_loss, run_correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item() * x.size(0)
        run_correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    train_loss = run_loss / max(total, 1)
    train_acc = run_correct / max(total, 1)
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step()

    print(
        f"[{ep:02d}/{num_epochs}] "
        f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
    )

    # 以验证准确率选最优（没有验证就用训练acc也行）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"  -> new best val_acc={best_val_acc:.4f}")

if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
print("Best val acc:", f"{best_val_acc:.4f}")


# ------------------ Kaggle CSV ------------------
# 你的函数稍作稳健化（不依赖固定 batch 偏移），按推理顺序生成 ID：0..N-1
def save_prediction_to_csv(
    model, loader, device, class_names, output_file="submission.csv"
):
    model.eval()
    ids, labels = [], []
    with torch.no_grad():
        idx_counter = 0
        for batchX, _ in loader:
            batchX = batchX.to(device, non_blocking=True).float()
            out = model(batchX)
            pred = out.argmax(dim=1).tolist()
            for p in pred:
                ids.append(idx_counter)
                labels.append(class_names[p])
                idx_counter += 1
    df = pd.DataFrame({"ID": ids, "Label": labels})
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file} (rows={len(ids)})")


save_prediction_to_csv(
    model, test_loader, device, class_names, output_file="submission.csv"
)
