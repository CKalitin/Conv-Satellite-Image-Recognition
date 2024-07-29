# Dataset source: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification?resource=download

import os
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import time

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
    
    i = 1000
    print(images[i])
    print(file_dirs[i])
    image = PIL.Image.open(file_dirs[i])
    image.show()
    plt.imshow(images[i].permute(1,2,0))
    plt.show()
    
load_data("./datapng/")