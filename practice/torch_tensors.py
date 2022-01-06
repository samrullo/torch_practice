import torch
import matplotlib.pyplot as plt

# uniformly distributed random numbers
X = torch.rand(1000)
plt.hist(X)
plt.suptitle("Uniform distribution")
plt.show()

# random numbers from gaussian distribution
plt.figure()
X = torch.randn(1000)
plt.hist(X)
plt.suptitle("Gaussian distribution")
plt.show()

# torch dot product
x1 = torch.randn(13, 19)
x2 = torch.randn(19, 23)
x3 = torch.matmul(x1, x2)
print(f"x3 shape : {x3.shape}")

# elementwise multiplication
z=x1*x1
z2=z.sum(axis=-1)
print(f"z2 shape : {z2.shape}")
plt.hist(z2)
plt.suptitle("Elementwise multiplication of two gaussians")
plt.show()