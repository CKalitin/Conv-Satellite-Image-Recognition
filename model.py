# Dataset source: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification?resource=download
# Tutorial: https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/
# Human: First attempt: 72%, Second attempt: 88%, Third attempt: 96% <- score to beat, or at least approach

import os
import torch
import torch.nn as nn
import torchvision
import datetime
import PIL
import matplotlib.pyplot as plt

label_dict = { 0: "cloudy", 1: "desert", 2: "forest", 3: "water"}
label_dict_opposite = { "cloudy": 0, "desert": 1, "forest": 2, "water": 3} # There's a better way to do this, study

# Pytorch Dataloader can take any class as a Dataset as long as it has __len__ and __getitem__ (to use square brackets []) https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class Dataset():
    def __init__(self, images=[], labels=[], file_dirs=[], data_dir=""):
        self.images = images # tensor
        self.labels = labels
        self.file_dirs = file_dirs
        if data_dir != "": self.load_images(data_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def load_images(self, data_dir, start_index=0, length=999999):
        self.images = []
        self.labels = []
        self.file_dirs = []
        
        paths = os.walk(data_dir)
        for path in paths:
            # Min so you don't go out of range. file_dir is completely dependent on the file naming convention, this is stupid, well only requires 1 preliminary step
            for i in range(start_index, min(length, len(path[2])-start_index)+start_index):
                file_dir = f"{path[0]}/{str.split(path[0], "/")[2]}_{i}.jpg"
                transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                
                self.images.append(transform(PIL.Image.open(file_dir)))
                self.labels.append(label_dict_opposite[str.split(path[0], "/")[2]])
                self.file_dirs.append(file_dir)
                
        print(f"Loaded {len(self.images)} images")

class Model(nn.Module):
    def __init__(self):
        super().__init__() # Makes this class a delegate of torch.nn.Module
        
        # Input = 3x64x64, Output = 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool2d(2)
        
        # Input = 16x32x32, Output = 16x16x16
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool2d(2)
        
        self.flat = nn.Flatten()
        
        self.fc3 = nn.Linear(4096, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        
        x = self.act2(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)
        
        x = self.flat(x)
        
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        
        x = self.fc4(x)
        
        return x

training_set = Dataset()
eval_set = Dataset()

eval_set_length = 100
training_set.load_images("./datajpg/", eval_set_length, 1400)
eval_set.load_images("./datajpg/", 0, eval_set_length)

batch_size = 32
train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(f"Model Parameters: { sum(p.numel() for p in model.parameters()) }") # numel returns the num elements in the given tensor

n_epochs = 15
accuracies = []
for epoch in range(n_epochs):
    for images, labels in train_loader:
        # forward, backward, and then weight update
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad() # Resets gradients
        loss.backward()
        optimizer.step()
    
    accuracy = 0
    count = 0
    for inputs, labels in eval_loader:
        y_pred = model(inputs)
        accuracy += (torch.argmax(y_pred, 1) == labels).float().sum() # Array of correct answers to float (bc. acc is float) then sum all elements
        count += len(labels)
    accuracy /= count
    accuracies.append(float(accuracy))
    print(f"Epoch: {epoch}, Accuracy: {round(float(accuracy) * 100, 0)}%")

avg_end_accuracy = round(sum(accuracies[-5:])/5*1000)
torch.save(model.state_dict(), f"./Models/model_{datetime.datetime.now().strftime("%m:%d:%H:%M:%S").replace(":", ".")}_a{avg_end_accuracy}.pth")