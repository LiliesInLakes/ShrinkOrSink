
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.transforms import v2
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import CosineAnnealingLR

accuracy=0
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
print(device)

#SEEDING

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
print(f"Manual seed set to {SEED}")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # Flip images left-to-right
    transforms.RandomRotation(15),               # Rotate by up to 15 degrees
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), # Zoom in slightly
    transforms.ColorJitter(brightness=0.2),      # Change brightness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standardize
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Add this where you load train_set and test_set
unlabeled_set = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=train_transform)

# Use a larger batch size for unlabeled data to speed things up
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=160, shuffle=True)

train_set = torchvision.datasets.STL10(root='./data', split= 'train', download=True, transform= transform)
train_set_aug= torchvision.datasets.STL10(root='./data', split= 'train', download=True, transform=train_transform)
test_set = torchvision.datasets.STL10(root='./data', split= 'test', download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
train_loader_aug = torch.utils.data.DataLoader(train_set_aug, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=False)


classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use more channels but fewer linear parameters
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), # 48x48
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), # 24x24
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), # 12x12
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 10) # Much smaller than 128*10*10!
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
    
net = ConvNeuralNet()
net.to(device)
     
loss_function = nn.CrossEntropyLoss(label_smoothing= 0.1)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

cutmix = v2.CutMix(num_classes=10)
mixup = v2.MixUp(num_classes=10)
epochs_plain = 20
epochs_aug= 20
epochs_cutmix= 20
best_val_loss = float('inf')
patience = 10
counter = 0
for epoch in range(epochs_plain):

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        if i % 30 == 1:
            print(f'[{epoch + 1}/{epochs_plain}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        optimizer.step()
    scheduler.step()

        
        # Inside your epoch loop:
    # if loss.item() < best_val_loss:
    #     best_val_loss = loss.item()
    #     torch.save(net.state_dict(), 'best_model.pth') # Save the "Sweet Spot"
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Stopping early to prevent overfitting!")
    #         break

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network after plain test images: {100 * correct // total} %')

for epoch in range(epochs_aug):

    running_loss = 0.0
    for i, data in enumerate(train_loader_aug):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        if i % 30 == 1:
            print(f'[{epoch + 1}/{epochs_aug}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        optimizer.step()
    scheduler.step()

        
        # Inside your epoch loop:
    # if loss.item() < best_val_loss:
    #     best_val_loss = loss.item()
    #     torch.save(net.state_dict(), 'best_model.pth') # Save the "Sweet Spot"
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Stopping early to prevent overfitting!")
    #         break

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network after aug test images: {100 * correct // total} %')

for epoch in range(epochs_cutmix):

    running_loss = 0.0
    for i, data in enumerate(train_loader_aug):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        if np.random.rand() < 0.5:
            
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            inputs, labels = cutmix_or_mixup(inputs, labels)
            # outputs = net(inputs)
            # if i==0:
            #     img = inputs[0].permute(1, 2, 0).cpu().numpy() 
            #     plt.imshow(img)
            #     plt.title("Lam")
            #     plt.show()
    # <rest of the training loop here>


        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        if i % 30 == 1:
            print(f'[{epoch + 1}/{epochs_cutmix}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        optimizer.step()
    scheduler.step()

        
        # Inside your epoch loop:
    # if loss.item() < best_val_loss:
    #     best_val_loss = loss.item()
    #     torch.save(net.state_dict(), 'best_model.pth') # Save the "Sweet Spot"
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Stopping early to prevent overfitting!")
    #         break

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network after aug and cutmix test images: {100 * correct // total} %')
print('Finished Training')


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
images, _ = next(iter(test_loader))

image = images[3]
batched_image = image.unsqueeze(0).to(device)
with torch.no_grad():
    log_probabilities = net(batched_image)

probabilities = torch.exp(log_probabilities).squeeze().cpu()
view_classification(image, probabilities)
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
# for images, labels in train_loader:
#     outputs = net(images)

#     # Get the index of the highest log-probability (the predicted class)
#     _, predicted = torch.max(outputs.data, 1)

#     # Total number of labels
#     total += labels.size(0)

#     # Total correct predictions
#     correct += (predicted == labels).sum().item()

# # Calculate accuracy percentage
# train_accuracy = 100 * correct / total
# print(f'Training Accuracy for testing underfitting: {train_accuracy:.2f}%')


param_size = 0
for param in net.parameters():
    param_size += param.nelement() * param.element_size()

buffer_size = 0
# for buffer in next.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print(f'Model size: {size_all_mb:.3f}MB')

#SAVING MODEL TO SAVE RUNTIME


threshold = 0.98  # Only "trust" the model if it's 95% sure
unlabeled_iter = iter(unlabeled_loader)

epochs_semi= 60
best_val_loss = float('inf')
patience = 10
counter = 0
for epoch in range(epochs_semi):
    if(epoch%20)==0:
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if 100 * correct // total >accuracy:
            accuracy= 100 * correct // total
            torch.save(net.state_dict(), './marvelmodel2.pth')
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    net.train()
    for i, (l_inputs, l_labels) in enumerate(train_loader):
        l_inputs, l_labels = l_inputs.to(device), l_labels.to(device)
        
        optimizer.zero_grad()

        # 1. SUPERVISED LOSS
        outputs = net(l_inputs)
        supervised_loss = loss_function(outputs, l_labels)

        # 2. SEMI-SUPERVISED LOSS
        try:
            u_inputs, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            u_inputs, _ = next(unlabeled_iter)
        
        u_inputs = u_inputs.to(device)
        
        # Get "Pseudo-Labels" (No Gradients for the guessing part)
        with torch.no_grad():
            u_outputs = net(u_inputs)
            probs = torch.softmax(u_outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs > threshold  # Only keep high confidence
        
        if mask.any():
            # Calculate loss for the unlabeled images we are sure about
            u_outputs_final = net(u_inputs[mask])
            unlabeled_loss = loss_function(u_outputs_final, pseudo_labels[mask])
            
            # Combine losses
            total_loss = supervised_loss + (0.5 * unlabeled_loss)
        else:
            total_loss = supervised_loss
        if i % 30 == 1:
            print(f'[{epoch + 1}/{epochs_semi}, {i + 1:5d}] loss: {supervised_loss.item() :.4f}')

        total_loss.backward()
        optimizer.step()
    scheduler.step()

    #this is to prevent overfitting, it will stop training once loss is no becoming less
    

        # Inside your epoch loop:
    # if total_loss.item() < best_val_loss:
    #     best_val_loss = total_loss.item()
    #     torch.save(net.state_dict(), 'best_model.pth') # Save the "Sweet Spot"
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Stopping early to prevent overfitting!")
    #         break

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        # Prune 20% of connections with the lowest L1-norm
        prune.l1_unstructured(module, name='weight', amount=0.2)
        prune.remove(module, 'weight') # Makes the pruning permanent
param_size = 0
for param in net.parameters():
    param_size += param.nelement() * param.element_size()

buffer_size = 0
# for buffer in next.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print(f'Model size: {size_all_mb:.3f}MB')
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
if 100 * correct // total >accuracy:
    accuracy= 100 * correct // total
    torch.save(net.state_dict(), './marvelmodel2.pth')
print(f'Accuracy of the network after pruning test images: {100 * correct // total} %')
