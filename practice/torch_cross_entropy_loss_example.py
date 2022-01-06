# let's say we have 3 classes and the output
# of our model is -1.2 for class 0, 0.12 for class 1
# and 4.2 for class 2
# the ground truth is class 2
import torch
import torch.nn as nn

logits=torch.tensor([[-1.2,0.12,4.1]])
ground_truth=torch.tensor([2])
criterion=nn.CrossEntropyLoss()
loss=criterion(logits,ground_truth)
print(f"loss is {loss}")