# Wrtten By: Raghwendra Dey

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import argparse
import numpy as np

class convNet(nn.Module):
    def __init__(self, num_classes=3):
        super(convNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(4*4*64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


parser = argparse.ArgumentParser(description="Covid-19 detection from X-ray Images")
parser.add_argument('--model', type=str, default='resnet18', help="Path to the weights of the model")
parser.add_argument('--image_path', type=str, default='./images/EUNJ6ZsXkAYcyW8.jfif', help="Path to the test image for prediction image_name must be of the form image.ext")
args = parser.parse_args()

if args.model == 'resnet18':
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 3)
    weights_path = './saved_models/resnet18.pt'
else:
    model = convNet()
    weights_path = './saved_models/LeNet_model.pt'

model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()
classes = ['Non-Pneumonia', 'Other Pneumonia', 'COVID-19']

if args.image_path.split('.')[-1] == 'dcm':
    import pydicom as dicom
    im = dicom.read_file(args.image_path).pixel_array
    if len(im.shape) != 3 or im.shape[2] != 3:
        im = np.stack((im,) * 3, -1)
else:
    import cv2
    im = cv2.imread(args.image_path)
    if len(im.shape) != 3 or im.shape[2] != 3:
        im = np.stack((im,) * 3, -1)

im = transforms.Compose([
    transforms.Resize(size=(35, 35)),
    transforms.ToTensor()
])(Image.fromarray(im))

im = im.unsqueeze(0)
out = model(im)
_, pred = torch.max(out, axis=1)
out = F.softmax(out, dim=1)
print()
print("Prediction: "+str(classes[pred]))
print()
print("Confidence value: ")
print(f"Non-Pneumonia: {out[0][0]*100:.4f} %, Other Pneumonia: {out[0][1]*100:.4f} %, COVID-19: {out[0][2]*100:.4f} %")
print()
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')

