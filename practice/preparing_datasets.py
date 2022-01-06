import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.201),(0.301))])

trainset=torchvision.datasets.MNIST()