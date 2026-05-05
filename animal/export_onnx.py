import torch
import torch.nn as nn
from torchvision import models

def replace_relu6(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU6):
            setattr(model, name, nn.ReLU(inplace=True))
        else:
            replace_relu6(module)

# 参数
NUM_CLASSES = 5
MODEL_PATH = "animal_model.pth"
ONNX_PATH = "animal_model.onnx"

# 加载 MobileNetV2 结构
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
# 替换激活函数
# replace_relu6(model)

model.eval()

# 模拟一个输入（1张3通道224x224图片）
dummy_input = torch.randn(1, 3, 224, 224)

# 导出为 ONNX 格式
torch.onnx.export(
    model, dummy_input, "animal_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)


print(f"✅ 模型已成功导出为 ONNX 格式: {ONNX_PATH}")
