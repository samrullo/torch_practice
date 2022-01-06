import torch
import torch.nn.functional as F

images = torch.randn(10, 1, 28, 28)
filter = torch.randn(6, 1, 3, 3)
output_features = F.conv2d(images, filter, stride=1, padding=1)
print(f"output features after F.conv2d : {output_features.shape}")

output_features = F.max_pool2d(output_features, 2)
print(f"output features after F.max_pool2d : {output_features.shape}")

output_features = F.avg_pool2d(output_features, 2)
print(f"output features after F.avg_pool2d : {output_features.shape}")
