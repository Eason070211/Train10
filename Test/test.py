import torch
from torch import nn
from torchvision.models import resnet18  # 引入 resnet
from PIL import Image
import torchvision.transforms as transforms
import os
import glob


# ============ 1. 定义模型结构（必须用 ResNet18，和你训练时一致） ============
def create_model():
    # 注意：这里要用 weights=None，不能用路径
    model = resnet18(weights=None)
    # 修改全连接层为 10 分类（你的任务一定是 CIFAR-10）
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


# ============ 2. CIFAR-10 类别名称 ============
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ============ 3. 数据预处理（⚠️ 关键！这里要看你的训练代码） ============
# 你在训练 ResNet 时，用的是 CIFAR-10 的归一化（0.4914...）还是 ImageNet 的归一化（0.485...）？
# 因为你的 1.py 里用的是 CIFAR-10 归一化，我猜测你训练 ResNet 时可能也沿用了这个。
# 如果预测不准，可以尝试把下面的 mean/std 换成 [0.485, 0.456, 0.406] 和 [0.229, 0.224, 0.225]。
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ⚠️ ResNet18 默认输入是 224x224，不是 32x32！
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ============ 4. 加载模型和权重 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model().to(device)

# 加载权重文件
checkpoint = torch.load("CIFAR10_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # 这里保持不变，因为保存时用的 key 是 'model_state_dict'
model.eval()
print("ResNet18 模型权重加载成功！")

# 可以顺便看看这个模型的最佳准确率是多少
if 'best_acc' in checkpoint:
    print(f"验证集最佳准确率: {checkpoint['best_acc']:.4f}")


# ============ 5. 预测函数 ============
def predict_folder(image_folder_path):
    image_paths = glob.glob(os.path.join(image_folder_path, "*.*"))
    valid_ext = ['.jpg', '.jpeg', '.png']
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_ext]

    if not image_paths:
        print("文件夹里没有找到图片！")
        return

    print(f"找到 {len(image_paths)} 张图片，开始预测...\n")

    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                label = predicted.item()

                print(f"图片: {os.path.basename(img_path)} -> 预测类别: {classes[label]} (编号: {label})")
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")


# ============ 6. 运行入口 ============
if __name__ == '__main__':
    folder_path = "./"  # 改成你的图片文件夹路径
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 '{folder_path}' 不存在，请修改 folder_path 变量！")
    else:
        predict_folder(folder_path)