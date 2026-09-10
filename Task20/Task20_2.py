import torch
import torchvision.models as models

# 1. Modelinizi yükleyin veya eğitin
model = models.resnet18(pretrained=True)
model.eval()

# 2. Veri setinizdeki verilere benzer sahte bir girdi oluşturun
dummy_input = torch.randn(1, 3, 224, 224)

# 3. ONNX olarak kaydedin
torch.onnx.export(model, dummy_input, "modelim.onnx", verbose=True)
