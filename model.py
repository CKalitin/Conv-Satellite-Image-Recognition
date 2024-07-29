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
    
    #plt.imshow(images[4000].permute(1,2,0))
    #plt.show()
    #print(file_dirs[4000])
    
    correct_answers = 0
    for i in range(0, 50):
        index = random.randint(0, len(images))
        plt.imshow(images[index].permute(1,2,0))
        plt.show(block=False)
        guess = input("Guess: ")
        plt.close()
        if guess == labels[index]: correct_answers += 1
        print(labels[index])
    print(f"p = {correct_answers / 50}")
    
load_data("./datajpg/")