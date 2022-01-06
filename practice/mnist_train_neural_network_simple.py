import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4), (0.3))])

trainset = torchvision.datasets.MNIST(root=r"/Users/samrullo/Documents/learning/data_science/pytorch/datasets",
                                      download=True,
                                      train=True,
                                      transform=transform)
testset = torchvision.datasets.MNIST(root=r"/Users/samrullo/Documents/learning/data_science/pytorch/datasets",
                                     download=True,
                                     train=False,
                                     transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


# define Fully Connected Neural Network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# train the model
max_iters = 100
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(10):
    logging.info(f"------------ epoch {epoch} start -------------------")
    for iter, data in enumerate(trainloader):
        inputs, labels = data
        # reshape inputs to have 2 dimensions instead of 3
        inputs = inputs.view(-1, 28 * 28 * 1)

        # clear gradients
        optimizer.zero_grad()

        # forward > backward > update parameters
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if iter > max_iters:
            break
# evaluate our model
correct, total = 0, 0
predictions = []
net.eval()

for inputs, labels in testloader:
    inputs = inputs.view(-1, 28 * 28 * 1)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"test data set accuracy : {100 * correct / total}")
