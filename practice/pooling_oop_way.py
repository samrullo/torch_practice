import torch

images = torch.rand(10, 1, 28, 28)
conv_filter = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
max_pooling=torch.nn.MaxPool2d(2)
output_features=conv_filter(images)
print(f"after conv2d : {output_features.shape}")
output_features=max_pooling(output_features)
print(f"after MaxPool2d : {output_features.shape}")

avg_pooling=torch.nn.AvgPool2d(2)
output_features=avg_pooling(output_features)
print(f"after AvgPool2d : {output_features.shape}")