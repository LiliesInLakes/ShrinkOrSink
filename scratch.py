import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
print(device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # Flip images left-to-right
    transforms.RandomRotation(15),               # Rotate by up to 15 degrees
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), # Zoom in slightly
    transforms.ColorJitter(brightness=0.2),      # Change brightness
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standardize
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_set = torchvision.datasets.STL10(root='./data', split= 'train', download=True, transform=train_transform)
test_set = torchvision.datasets.STL10(root='./data', split= 'test', download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

print('Number of images in the training dataset:', len(train_set))
print('Number of images in the testing dataset:', len(test_set))
found_images = {}
for image, label in train_set:
    if label not in found_images:
        found_images[label] = image

print(f"Shape of the images in the training dataset: {train_loader.dataset[0][0].shape}")

# fig, axes = plt.subplots(1, 10, figsize=(24, 6))
# for i in range(10):
#     image = train_loader.dataset[i][0].permute(1, 2, 0)
#     denormalized_image= image / 2 + 0.5
#     axes[i].imshow(denormalized_image)
#     axes[i].set_title(classes[train_loader.dataset[i][1]])
#     axes[i].axis('off')
# plt.show()
fig, axes = plt.subplots(1, 10, figsize=(24, 6))
for i in range(10):
    image = found_images[i].permute(1, 2, 0)
    #turn to grayscale
    denormalized_image= image / 2 + 0.5
    axes[i].imshow(denormalized_image)
    axes[i].set_title(classes[i])
    axes[i].axis('off')
plt.show()

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(128,256, 3)
        # self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, stride=2)

        # self.fc1 = nn.Linear(128 * 12 * 12, 128, bias= True)
        # self.fc2 = nn.Linear(120, 84, bias= True)
        # self.fc3 = nn.Linear(84, 10, bias =True)
        self.fc1 = nn.Linear(128*10*10, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)


    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.log_softmax(self.fc3(x), dim=1)
    #     return x
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # x = x.view(-1, 128 * 12 * 12) # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = ConvNeuralNet()
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 400 == 1:
            print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')

#check model size
param_size = 0
for param in net.parameters():
    param_size += param.nelement() * param.element_size()

buffer_size = 0
# for buffer in next.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print(f'Model size: {size_all_mb:.3f}MB')

def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    image = image.permute(1, 2, 0)
    denormalized_image= image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()



correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')