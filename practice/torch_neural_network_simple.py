# first we will create neural network using tensors only
# input layer is flattened 28x28 image
# first hidden layer has 200 units
# output layer has 10 units

import torch

input_layer=torch.rand(1000,784)
weight_1=torch.rand(784,200)
weight_2=torch.rand(200,10)

out=torch.matmul(input_layer,weight_1)
out=torch.matmul(out,weight_2)
print(f"out shape : {out.shape}")