import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])

trainset = torchvision.datasets.CIFAR10(root=r"/Users/amrul/Documents/learning/data_science/pytorch/datasets",
                                        download=True,
                                        train=True,
                                        transform=transform
                                        )
testset = torchvision.datasets.CIFAR10(root=r"/Users/amrul/Documents/learning/data_science/pytorch/datasets",
                                       download=True,
                                       train=False,
                                       transform=transform
                                       )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

print(f"train data shape : {trainloader.dataset.train_data.shape}")
print(f"trainloader batch size : {trainloader.batch_size}")
print(f"test data shape : {testloader.dataset.test_data.shape}")
print(f"testloader batch size : {testloader.batch_size}")
