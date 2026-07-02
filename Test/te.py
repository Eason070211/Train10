import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms as transforms
import os

# ================= 1. 定义 ResNet18 模型（10分类） =================
def create_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 输出 10 类
    return model

# ================= 2. CIFAR-10 类别名称 =================
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ================= 3. 数据预处理（关键！） =================
# 注意：ResNet18 原生接受 224x224，但很多人在 CIFAR-10 上直接训 32x32。
# 请根据你训练时的代码，把下面的 (32, 32) 改成 (224, 224) 或保持不变。
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # ← 如果训练时用了 224，这里必须改！
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ================= 4. 加载模型权重 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)

# 加载 .pth 文件（请务必改成你的实际文件名！）
checkpoint = torch.load("CIFAR10_best.pth", map_location=device)

# 兼容两种保存方式：直接 state_dict 或 含 'model_state_dict' 的字典
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ ResNet18 权重加载成功！")

# ================= 5. 遍历当前文件夹下的图片并预测 =================
image_ext = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir('.') if f.lower().endswith(image_ext)]

if not image_files:
    print("⚠️ 当前文件夹没有找到 .jpg/.jpeg/.png 图片。")
else:
    print(f"📸 找到 {len(image_files)} 张图片：\n")
    with torch.no_grad():
        for img_name in image_files:
            try:
                img = Image.open(img_name).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                pred = model(img_tensor).argmax(1).item()
                print(f"{img_name}  →  {classes[pred]}")
            except Exception as e:
                print(f"❌ {img_name} 出错: {e}")