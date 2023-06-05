import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import model

total_epoch = 50
lr = 0.01

# 이용가능한 GPU 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 데이터셋을 trainset 40000개와 validationset 10000개로 분리
indices = list(range(len(dataset)))
random.shuffle(indices)
train_indices = indices[:40000]
validation_indices = indices[40000:]

trainset =  torch.utils.data.Subset(dataset, train_indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)
validationset =  torch.utils.data.Subset(dataset, validation_indices)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=100, shuffle=False, num_workers=16)
    
# 손실함수, 옵티마이저, 스케쥴러 정의
model = model.AlexNet()
model = model.to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train():
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0   
    return total_loss / len(trainloader), 100 * correct / total
            
def validation():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
    return total_loss / len(validationloader), 100 * correct / total


# 모델 학습
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []
for epoch in tqdm(range(total_epoch)):
    train_loss, train_accuracy = train()
    valid_loss, valid_accuracy = validation()
    scheduler.step()

    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_accuracy)
    
print('Finished Training')

# 그래프 그리기
plt.plot(train_loss_list, label='train loss', marker='.')
plt.plot(valid_loss_list, label='valid loss', marker='.')
plt.legend()
plt.savefig('loss.png')
plt.clf()

plt.plot(train_acc_list, label='train accuracy', marker='.')
plt.plot(valid_acc_list, label='valid accuracy', marker='.')
plt.legend()
plt.savefig('accuracy.png')

# 모델 저장
PATH = './mymodel_cifar10.pth'
torch.save(model.state_dict(), PATH)
