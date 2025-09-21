import torch
import torchvision.models as models
import torch.nn as nn
import time

# GPU check
print("CUDA r u there?", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("say ur name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

# Dummy model (small one for speed)
class DroneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.base.classifier[1] = nn.Linear(self.base.last_channel, 5)
    def forward(self, x):
        return self.base(x)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model + random input
model = DroneClassifier().to(device)
input_tensor = torch.randn(1, 3, 224, 224).to(device)

print("Model on:", next(model.parameters()).device)
print("Input on:", input_tensor.device)

# Warmup
for _ in range(10):
    _ = model(input_tensor)

# Benchmark
start = time.time()
for _ in range(50):
    _ = model(input_tensor)
end = time.time()

print("Avg inference time: {:.2f} ms".format((end-start)/50*1000))
