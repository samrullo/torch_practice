import torch

images = torch.rand(10, 1, 28, 28)
conv_filter = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
output_features = conv_filter(images)
print(output_features.shape)
