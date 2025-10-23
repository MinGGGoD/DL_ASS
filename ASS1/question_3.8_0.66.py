############ 66% 的版本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 类别名称
class_names = ['birds', 'bottles', 'breads', 'butterfiles', 'cakes',
               'cats', 'chickens', 'cows', 'dogs', 'ducks',
               'elephants', 'fishes', 'handguns', 'horses', 'lions',
               'lipsticks', 'seals', 'snakes', 'spiders', 'vases']

# 改进的数据集类 - 支持保存文件名
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, return_paths=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            if self.labels is not None:
                return image, self.labels[idx], image_path
            else:
                return image, 0, image_path
        else:
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image, 0

# 改进的数据增强 - 添加多尺度训练
def get_train_transform(size=224):
    return transforms.Compose([
        # 多尺度训练，模拟不同分辨率
        transforms.RandomChoice([
            transforms.Resize((64, 64)),   # 模拟测试集尺寸
            transforms.Resize((128, 128)),
            transforms.Resize((224, 224)),
            transforms.Resize((256, 256)),
        ]),
        transforms.Resize((224, 224)),  # 最终统一到224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 验证集转换
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集转换 - 针对小尺寸优化
test_transform = transforms.Compose([
    # 先稍微放大，再裁剪，保持更多细节
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 改进的模型架构
class ImprovedImageClassifier(nn.Module):
    def __init__(self, num_classes=20, model_name='resnet50', pretrained=True):
        super(ImprovedImageClassifier, self).__init__()

        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features

            # 解冻更多层用于适应小图片
            for name, param in self.base_model.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # 更复杂的分类头
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)

            # EfficientNet更适合小图片
            for param in self.base_model.features[:-3].parameters():
                param.requires_grad = False

            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.base_model(x)

# 训练函数保持不变
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 改进的预测保存函数 - 确保正确的ID顺序
def save_prediction_to_csv_improved(model, test_image_paths, device, batch_size=32, output_file="submission.csv"):
    """
    改进版本：按照文件名顺序生成预测
    """
    model.eval()

    # 创建测试数据集和加载器
    test_dataset = ImageClassificationDataset(
        test_image_paths,
        transform=test_transform,
        return_paths=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_predictions = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images, _, paths = batch
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_paths.extend(paths)

    # 创建DataFrame
    df = pd.DataFrame({
        'ID': range(len(all_predictions)),
        'Label': [class_names[pred] for pred in all_predictions]
    })

    # 保存
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(df)}")

    # 显示前几个预测
    print("\n前10个预测:")
    print(df.head(10))

    return df

# 测试时增强（TTA）
def predict_with_tta(model, test_image_paths, device, n_augmentations=5):
    """
    测试时增强：对每张图片进行多次不同的预处理，然后平均预测结果
    """
    model.eval()

    # 多个测试变换
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    all_predictions = []

    for img_path in tqdm(test_image_paths, desc="TTA Prediction"):
        image = Image.open(img_path).convert('RGB')

        # 对每个变换进行预测
        predictions = []
        for transform in tta_transforms[:n_augmentations]:
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())

        # 平均所有预测
        avg_prediction = np.mean(predictions, axis=0)
        final_pred = np.argmax(avg_prediction)
        all_predictions.append(final_pred)

    # 创建DataFrame
    df = pd.DataFrame({
        'ID': range(len(all_predictions)),
        'Label': [class_names[pred] for pred in all_predictions]
    })

    return df

# 主训练流程
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 32
    num_epochs = 30  # 增加训练轮数
    learning_rate = 0.001
    weight_decay = 1e-4

    # 数据路径
    train_dir = '/content/FIT5215_Dataset'
    test_dir = '/content/test_set/official_test'

    # 加载训练数据
    train_image_paths = []
    train_labels = []

    if os.path.exists(train_dir):
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        train_image_paths.append(os.path.join(class_dir, img_name))
                        train_labels.append(class_idx)
                print(f"类别 {class_name}: {len([l for l in train_labels if l == class_idx])} 张图片")

    # 加载测试数据 - 保持文件名顺序
    test_image_paths = []
    if os.path.exists(test_dir):
        # 获取所有图片文件
        test_files = []
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                test_files.append(img_name)

        # 按文件名排序（确保ID顺序正确）
        test_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

        # 构建完整路径
        test_image_paths = [os.path.join(test_dir, f) for f in test_files]

    print(f"\n数据加载完成:")
    print(f"训练图片数量: {len(train_image_paths)}")
    print(f"测试图片数量: {len(test_image_paths)}")

    if len(train_image_paths) == 0:
        print("错误：没有找到训练数据！")
        return

    # 数据集划分
    train_paths, val_paths, train_labs, val_labs = train_test_split(
        train_image_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # 创建数据集 - 使用改进的多尺度训练
    train_dataset = ImageClassificationDataset(
        train_paths, train_labs,
        transform=get_train_transform()
    )
    val_dataset = ImageClassificationDataset(
        val_paths, val_labs,
        transform=val_transform
    )

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型 - 可以尝试EfficientNet
    # model = ImprovedImageClassifier(num_classes=20, model_name='efficientnet_b0', pretrained=True)
    model = ImprovedImageClassifier(num_classes=20, model_name='resnet50', pretrained=True)
    model = model.to(device)

    # 损失函数 - 添加标签平滑
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, num_classes, smoothing=0.1):
            super().__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.confidence = 1.0 - smoothing

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.mean(torch.sum(-true_dist * pred, dim=-1))

    criterion = LabelSmoothingLoss(num_classes=20, smoothing=0.1)

    # 优化器
    optimizer = optim.AdamW([
        {'params': model.base_model.layer3.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.base_model.layer4.parameters(), 'lr': learning_rate * 0.5},
        {'params': model.base_model.fc.parameters(), 'lr': learning_rate}
    ], weight_decay=weight_decay)

    # 学习率调度 - Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # 训练循环
    best_val_acc = 0
    patience = 0
    max_patience = 7

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # 学习率调整
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"✓ Model saved with validation accuracy: {val_acc:.2f}%")
            patience = 0
        else:
            patience += 1

        # 早停
        if patience >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n加载最佳模型 (Val Acc: {checkpoint['val_acc']:.2f}%)")

    # 生成预测 - 使用改进的函数
    save_prediction_to_csv_improved(model, test_image_paths, device, batch_size=32)

    # 可选：使用TTA生成更稳定的预测
    print("\n生成TTA预测...")
    tta_df = predict_with_tta(model, test_image_paths, device, n_augmentations=3)
    tta_df.to_csv('submission_tta.csv', index=False)
    print("TTA predictions saved to submission_tta.csv")

if __name__ == "__main__":
    main()