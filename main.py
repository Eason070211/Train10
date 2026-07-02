import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import tarfile
from datetime import datetime

# ======================== CONFIGURATION ========================
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=".", 
                    help="Directory containing cifar-10-python.tar.gz (default: current dir)")
parser.add_argument("--resume", type=str, default=None, 
                    help="Path to checkpoint to resume from")
parser.add_argument("--lr", type=float, default=0.001, 
                    help="Initial learning rate")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, 
                    help="Total epochs to train")
parser.add_argument("--patience", type=int, default=10, 
                    help="Early stopping patience")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ======================== DATASET VALIDATION ========================
def setup_cifar10():
    """智能处理CIFAR-10数据集路径"""
    data_root = "./data"
    extracted_dir = os.path.join(data_root, "cifar-10-batches-py")
    
    # 情况1: 已解压好的数据集存在
    if os.path.exists(extracted_dir):
        print(f"📂 Found pre-extracted dataset at {extracted_dir}")
        return data_root
    
    # 情况2: 查找用户提供的tar.gz文件
    tar_path = None
    possible_paths = [
        os.path.join(args.data_dir, "cifar-10-python.tar.gz"),
        "cifar-10-python.tar.gz",
        os.path.join(os.getcwd(), "cifar-10-python.tar.gz")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            tar_path = path
            break
    
    if not tar_path:
        raise FileNotFoundError(
            "\n❌ CIFAR-10 dataset not found!\n"
            "Please place 'cifar-10-python.tar.gz' in:\n"
            f"1. Current directory, OR\n"
            f"2. Specified with --data-dir (current: {args.data_dir})\n"
            "Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
    
    # 自动解压到标准位置
    print(f"📦 Found dataset archive at {tar_path}")
    os.makedirs(data_root, exist_ok=True)
    print(f"⏳ Extracting to {data_root}... (this may take 10-30 seconds)")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_root)
    
    if os.path.exists(extracted_dir):
        print(f"✅ Successfully extracted to {extracted_dir}")
        return data_root
    else:
        raise RuntimeError("Extraction failed! Invalid archive format.")

# ======================== DATA PREPROCESSING ========================
# Standard CIFAR-10 normalization parameters
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 验证并获取数据集路径
try:
    data_root = setup_cifar10()
except Exception as e:
    print(e)
    exit(1)

# 加载数据集 (关键: download=False)
train_set = datasets.CIFAR10(
    root=data_root, 
    train=True, 
    download=False,  # 完全禁用自动下载
    transform=train_transform
)
test_set = datasets.CIFAR10(
    root=data_root, 
    train=False, 
    download=False,  # 完全禁用自动下载
    transform=test_transform
)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

print(f"📊 Dataset loaded: {len(train_set)} train, {len(test_set)} test samples")

# ======================== MODEL SETUP ========================
def create_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# ======================== TRAINING SETUP ========================
def main():
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    start_epoch = 0
    best_acc = 0.0
    log_dir = f"logs_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        torch.set_rng_state(checkpoint["rng_state"].to(device))
        
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"➡️ Resuming from epoch {start_epoch} | Current best accuracy: {best_acc:.2%}")
        
        # Auto-adjust LR when resuming
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"] * 0.1, 1e-5)
            print(f"📉 Adjusted LR to {param_group['lr']:.1e} for fine-tuning")
    else:
        print("🚀 Starting training from scratch")

    # TensorBoard setup
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except ImportError:
        writer = None
        print("⚠️ TensorBoard not available. Skipping logging.")

    # --- 早停机制初始化 ---
    patience_counter = 0
    
    # ======================== TRAINING LOOP ========================
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        scheduler.step(val_acc)
        
        # Log metrics
        print(f"Epoch {epoch+1}/{start_epoch + args.epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(test_loader):.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        if writer:
            writer.add_scalars("Loss", {"train": train_loss/len(train_loader), "val": val_loss/len(test_loader)}, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

        # --- 早停逻辑 ---
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "rng_state": torch.get_rng_state()
            }, "CIFAR10_best.pth")
            print(f"⭐ New best model saved (Val Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    if writer:
        writer.close()
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()