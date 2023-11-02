import os

import torch
import torch.nn as nn
import torch.optim as optim
import time

from models import ResNet50
from utils import *

print("Running MTLSign pretrain.py.")


'''Device setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''Train/test dataloader setting'''
print("Using CIFAR10 dataset.")
train_db_path = './data'
test_db_path = './data'
batchsize = 100
num_classes = 10

transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

trainset = torchvision.datasets.CIFAR10(root=train_db_path, train=True, download=True, 
                                        transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=test_db_path,train=False, download=True,
                                        transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                            shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers=4)


'''Get pretrained resnet on cifar10'''
model = ResNet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
max_epochs = 60

for epoch in range(max_epochs):
    '''time count'''
    time_s = time.time()

    adjust_lr_rate(init_lr=0.1, optimizer=optimizer, epoch=epoch, lradj=20)  

    '''Model train'''
    print(f"\n[train] Epoch: {epoch}")
    model.train()
    iteration = 0
    train_loss = 0
    correct = 0
    total = 0
    print_frq = 50
    save_frq = 10

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicts.eq(targets.data).cpu().sum()

        if (iteration % print_frq == 0):
            print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                        % (iteration, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            

    '''Model test and save'''
    if (epoch % 10 == 0 or epoch == (max_epochs - 1)):
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        for idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicts = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += predicts.eq(targets.data).cpu().sum()

        print("[Test] Loss: %.3f | Acc: %.3f%% (%d / %d)\n" 
                % (test_loss / (idx + 1), 100. * test_correct / test_total, test_correct, test_total))
        
        torch.save({'model_state_dict': model.state_dict()}, os.path.join('./checkpoint', 'baseline_' + str(epoch) + '.pth'))