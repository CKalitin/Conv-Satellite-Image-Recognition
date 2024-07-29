# Dataset source: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification?resource=download
# Human: First attempt: 72%, Second attempt: 88%, Third attempt: 96% <- score to beat, or at least approach

import os
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import random

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

categories = ["cloudy", "desert", "forest", "water"]

#training_set = torch.utils.data.Dataset()

#train_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

def load_data(root_dir):
    images = []
    labels = []
    file_dirs = []
    
    paths = os.walk(root_dir)
    for path in paths:
        for file_name in path[2]:
            images.append(transform(PIL.Image.open(path[0] + "/" + file_name)))
            labels.append(str.split(path[0], "/")[2])
            file_dirs.append(path[0] + "/" + file_name)
            
    print(f"Loaded {len(images)} images")
    
    return images, labels, file_dirs

class Dataset():
    def __init__(self, images=[], labels=[], file_dirs=[]):
        self.images = images
        self.labels = labels
        self.file_dirs = file_dirs
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
    def load_images(self, data_dir, start_index=0, length=999999999):
        self.images = []
        self.labels = []
        self.file_dirs = []
        
        paths = os.walk(data_dir)
        for path in paths:
            for file_name in path[2]:
                self.images.append(transform(PIL.Image.open(path[0] + "/" + file_name)))
                self.labels.append(str.split(path[0], "/")[2])
                self.file_dirs.append(path[0] + "/" + file_name)
                
        print(f"Loaded {len(self.images)} images")
    
trainingdataset = Dataset("./datajpg/")