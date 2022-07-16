import torch
import os
from torchvision import models
from torchvision import transforms
from PIL import Image
import logging
from logging_utils.init_logging import init_logging

init_logging()

resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                 )])

image_labels_path = r"C:\Users\amrul\PycharmProjects\dlwpt-code\data\p1ch2\imagenet_classes.txt"
with open(image_labels_path) as fh:
    labels = [line.strip() for line in fh.readlines()]

img_path=os.path.join(r"C:\Users\amrul\programming\pyqt_related\images", "apple.jpg")
image = Image.open(img_path)
# image.show()

image_t=preprocess(image)
batch_t=torch.unsqueeze(image_t,0)

resnet.eval()
out=resnet(batch_t)
_,index=torch.max(out,1)

percentage=torch.nn.functional.softmax(out,dim=1)[0]*100
print(labels[index[0]],percentage[index[0]].item())
