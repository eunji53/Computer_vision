import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm

lr = 0.01

# 이용가능한 GPU 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 데이터 전처리
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# 앙상블할 모델 리스트에 담기
models = []
for i in range(5):
    model = timm.create_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(f"resnet18_cifar10_%dfold.pth" % (i+1)))
    model.eval()
    model = model.to(device)
    models.append(model)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        bs, ncrops, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)
        for model in models:
            outputs += model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))
