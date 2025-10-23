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
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 动态获取类别名称的函数
def get_class_names(train_dir):
    """从训练数据目录动态获取类别名称"""
    if not os.path.exists(train_dir):
        raise ValueError(f"训练数据目录不存在: {train_dir}")
    
    # 获取所有子目录名称作为类别名称
    class_names = []
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        # 只包含目录，排除文件
        if os.path.isdir(item_path):
            class_names.append(item)
    
    # 排序以确保一致性
    class_names.sort()
    print(f"发现 {len(class_names)} 个类别: {class_names}")
    return class_names

# ==================== 高级数据增强 ====================
class MixUp:
    """MixUp数据增强：混合两个样本及其标签"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, images, labels):
        batch_size = images.size(0)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam

class CutMix:
    """CutMix数据增强：剪切并粘贴图片区域"""
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, images, labels):
        batch_size = images.size(0)
        if np.random.rand() > self.prob:
            return images, labels, labels, 1.0

        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(batch_size).to(images.device)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        labels_a, labels_b = labels, labels[rand_index]
        return images, labels_a, labels_b, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

# 高级数据增强管道
def get_advanced_train_transform():
    return transforms.Compose([
        # 随机选择不同的缩放策略
        transforms.RandomChoice([
            transforms.Resize((64, 64)),   # 25% 概率使用测试集大小
            transforms.Resize((96, 96)),
            transforms.Resize((128, 128)),
            transforms.Resize((224, 224)),  # 25% 概率使用标准大小
        ]),
        transforms.Resize((224, 224)),  # 统一到224x224
        transforms.RandomHorizontalFlip(p=0.5),
        # 更强的数据增强
        transforms.RandomApply([
            transforms.RandomRotation(30),
        ], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        ], p=0.5),
        # 随机擦除
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

# ==================== 模型集成 ====================
class EnsembleModel(nn.Module):
    """集成多个模型的预测"""
    def __init__(self, num_classes=20):
        super(EnsembleModel, self).__init__()

        # 模型1: ResNet50
        self.model1 = models.resnet50(pretrained=True)
        num_features1 = self.model1.fc.in_features
        self.model1.fc = nn.Sequential(
            nn.Linear(num_features1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 模型2: EfficientNet-B1
        self.model2 = models.efficientnet_b1(pretrained=True)
        num_features2 = self.model2.classifier[1].in_features
        self.model2.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # 模型3: DenseNet121
        self.model3 = models.densenet121(pretrained=True)
        num_features3 = self.model3.classifier.in_features
        self.model3.classifier = nn.Sequential(
            nn.Linear(num_features3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 可学习的权重
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        # 加权平均
        w = F.softmax(self.weights, dim=0)
        output = w[0] * out1 + w[1] * out2 + w[2] * out3
        return output

# ==================== 单模型但更强大 ====================
class PowerfulSingleModel(nn.Module):
    def __init__(self, num_classes=20, model_name='efficientnet_b2'):
        super(PowerfulSingleModel, self).__init__()

        if model_name == 'efficientnet_b2':
            self.base = models.efficientnet_b2(pretrained=True)
            # 解冻更多层
            for name, param in self.base.named_parameters():
                if 'features.5' in name or 'features.6' in name or 'features.7' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            num_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'convnext_tiny':
            self.base = models.convnext_tiny(pretrained=True)
            # ConvNeXt在小图片上表现很好
            for param in self.base.features[:-2].parameters():
                param.requires_grad = False

            num_features = self.base.classifier[2].in_features
            self.base.classifier[2] = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.base(x)

# ==================== 改进的数据集类 ====================
class AdvancedDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None,
                 training=False, mixup_alpha=0.2, cutmix_prob=0.5):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.training = training
        self.mixup = MixUp(alpha=mixup_alpha) if training else None
        self.cutmix = CutMix(prob=cutmix_prob) if training else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image, 0

# ==================== 训练技巧 ====================
def train_epoch_advanced(model, train_loader, criterion, optimizer, device, mixup=None, cutmix=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # 应用MixUp或CutMix
        if mixup and random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        elif cutmix:
            images, labels_a, labels_b, lam = cutmix(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# ==================== 知识蒸馏 ====================
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        # 硬标签损失
        hard_loss = self.ce_loss(student_outputs, labels)

        # 软标签损失（知识蒸馏）
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# ==================== 高级预测策略 ====================
def predict_with_advanced_tta(model, test_image_paths, device, class_names, n_tta=10):
    """
    高级测试时增强
    """
    model.eval()

    tta_transforms = []
    for i in range(n_tta):
        if i == 0:
            # 原始
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif i < 3:
            # 不同的resize策略
            size = [256, 288][i-1]
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif i < 5:
            # 水平翻转
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 随机裁剪
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        tta_transforms.append(transform)

    all_predictions = []

    for img_path in tqdm(test_image_paths, desc="Advanced TTA"):
        image = Image.open(img_path).convert('RGB')

        predictions = []
        with torch.no_grad():
            for transform in tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(device)
                output = model(img_tensor)
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())

        # 加权平均（给原始图片更高权重）
        weights = [2.0] + [1.0] * (n_tta - 1)
        weights = np.array(weights) / sum(weights)
        avg_prediction = np.average(predictions, axis=0, weights=weights)

        final_pred = np.argmax(avg_prediction)
        all_predictions.append(final_pred)

    df = pd.DataFrame({
        'ID': range(len(all_predictions)),
        'Label': [class_names[pred] for pred in all_predictions]
    })

    return df

# ==================== 主训练流程 ====================
def train_advanced_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 32  # 减小batch size以适应更大的模型
    num_epochs = 40
    learning_rate = 0.0005
    weight_decay = 1e-4

    # 数据路径
    train_dir = '/Users/2m/Documents/Monash/FIT5215/DL_ASS/FIT5215_Dataset'
    test_dir = '/Users/2m/Documents/Monash/FIT5215/DL_ASS/test_set/official_test'

    # 动态获取类别名称
    class_names = get_class_names(train_dir)
        
    # 加载数据
    train_image_paths = []
    train_labels = []

    if os.path.exists(train_dir):
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        train_image_paths.append(os.path.join(class_dir, img_name))
                        train_labels.append(class_idx)

    # 加载测试数据
    test_image_paths = []
    if os.path.exists(test_dir):
        test_files = sorted(os.listdir(test_dir),
                          key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        test_image_paths = [os.path.join(test_dir, f) for f in test_files
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"训练数据: {len(train_image_paths)} 张")
    print(f"测试数据: {len(test_image_paths)} 张")

    # 数据划分
    train_paths, val_paths, train_labs, val_labs = train_test_split(
        train_image_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    # 创建数据集
    train_dataset = AdvancedDataset(
        train_paths, train_labs,
        transform=get_advanced_train_transform(),
        training=True
    )
    val_dataset = AdvancedDataset(
        val_paths, val_labs,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    model = PowerfulSingleModel(num_classes=20, model_name='convnext_tiny')
    # 或使用集成模型（更慢但可能更准）
    # model = EnsembleModel(num_classes=20)

    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 分层学习率
    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name or 'fc' in name:
                param_groups.append({'params': param, 'lr': learning_rate})
            else:
                param_groups.append({'params': param, 'lr': learning_rate * 0.1})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # 学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # MixUp和CutMix
    mixup = MixUp(alpha=0.2)
    cutmix = CutMix(beta=1.0, prob=0.5)

    # 训练
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, mixup, cutmix
        )

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_advanced_model.pth')
            print(f"✓ 保存模型 (Val Acc: {val_acc:.2f}%)")

    # 生成预测
    model.load_state_dict(torch.load('best_advanced_model.pth'))

    # 使用高级TTA
    predictions_df = predict_with_advanced_tta(model, test_image_paths, device, class_names, n_tta=10)
    predictions_df.to_csv('submission_advanced.csv', index=False)
    print("高级预测已保存到 submission_advanced.csv")

    return model

# ==================== K折交叉验证（可选） ====================
def train_with_kfold(n_folds=5):
    """
    使用K折交叉验证训练多个模型，然后集成预测
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    train_dir = '/Users/2m/Documents/Monash/FIT5215/DL_ASS/FIT5215_Dataset'
    test_dir = '/Users/2m/Documents/Monash/FIT5215/DL_ASS/test_set/official_test'

    # 动态获取类别名称
    class_names = get_class_names(train_dir)
    
    # 加载训练数据
    train_image_paths = []
    train_labels = []

    if os.path.exists(train_dir):
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        train_image_paths.append(os.path.join(class_dir, img_name))
                        train_labels.append(class_idx)

    # 加载测试数据
    test_image_paths = []
    if os.path.exists(test_dir):
        test_files = sorted(os.listdir(test_dir),
                          key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        test_image_paths = [os.path.join(test_dir, f) for f in test_files
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"训练数据: {len(train_image_paths)} 张")
    print(f"测试数据: {len(test_image_paths)} 张")

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_image_paths, train_labels)):
        print(f"\n训练 Fold {fold+1}/{n_folds}")

        # 划分数据
        train_paths = [train_image_paths[i] for i in train_idx]
        train_labs = [train_labels[i] for i in train_idx]
        val_paths = [train_image_paths[i] for i in val_idx]
        val_labs = [train_labels[i] for i in val_idx]

        # 创建模型并训练
        model = PowerfulSingleModel(num_classes=len(class_names))
        model = model.to(device)

        # 这里应该添加完整的训练代码，暂时简化
        print(f"Fold {fold+1} 训练完成（实现略）")

        fold_models.append(model)

    # 集成所有fold的预测
    all_predictions = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for img_path in test_image_paths:
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        fold_preds = []
        for model in fold_models:
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                pred = F.softmax(output, dim=1).cpu().numpy()
            fold_preds.append(pred)

        # 投票或平均
        final_pred = np.mean(fold_preds, axis=0)
        all_predictions.append(np.argmax(final_pred))

    # 生成预测DataFrame
    df = pd.DataFrame({
        'ID': range(len(all_predictions)),
        'Label': [class_names[pred] for pred in all_predictions]
    })

    return df

if __name__ == "__main__":
    # 运行高级训练
    model = train_advanced_model()

    # 可选：K折交叉验证
    # predictions = train_with_kfold(n_folds=5)