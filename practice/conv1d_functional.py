import torch
import torch.nn.functional as F

x=torch.rand(32,1,8000)
conv1d=torch.nn.Conv1d(in_channels=1,out_channels=8,kernel_size=(13,),stride=(1,),padding=0)
maxpool1d=torch.nn.MaxPool1d(3)
out=conv1d(x)
print("out shape after conv1d",out.shape)
out=maxpool1d(out)
print("out shape after maxpool1d",out.shape)
