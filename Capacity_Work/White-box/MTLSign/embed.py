import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import ResNet50
from utils import *
from watermark import MTLSign, QRset

import argparse

parser = argparse.ArgumentParser(description='MTLSign embed')
parser.add_argument('--id', default=0, type=int, help='process id')
args = parser.parse_args()

print("Running MTLSign embed.py.")

'''Device setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


'''Load pretrained model'''
checkpoint_path = './checkpoint/pretrain_50.pth'
checkpoint = torch.load(checkpoint_path)
ckp_dict = checkpoint['model_state_dict']
model = ResNet50().to(device)
model.load_state_dict(ckp_dict)


'''QRdataset setting'''
qr_transform = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
wmset = QRset('./data/qrdataset/index.txt', transform=qr_transform)
wmloader = torch.utils.data.DataLoader(
    dataset=wmset,
    batch_size=100,
    shuffle=True
)


MTL = MTLSign()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 0.0001},
    {'params': MTL.wmend.parameters(), 'lr': 0.1}
])


'''Model test'''
print("\nTest pretrained model...")    
test(model, criterion, testloader, device)


'''Fine-tuning model to embed watermark'''
wm_epoch = 20

for epoch in range(wm_epoch):
    print(f"\n[embed] Epoch: {epoch}")

    '''Adjust learning rate'''
    if (epoch + 1) % 10 == 0:
        for param_group in optimizer.param_groups:
            if (param_group['lr'] > 1e-4):
                param_group['lr'] *= 0.1

    '''Model Fine-Tuning'''
    model.train()
    MTL.wmend.train()

    iteration = 0
    train_loss = 0
    correct = 0
    total = 0
    print_frq = 50

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        wm_loss = 0
        for idx, (wm_inputs, wm_targets) in enumerate(wmloader):
            freeze_bn(model)   # freeze bn when verify
            wm_inputs, wm_targets = wm_inputs.to(device), wm_targets.to(device)
            wm_outputs = MTL.verify(model, wm_inputs, device)
            wm_loss = criterion(wm_outputs, wm_targets)

        unfreeze_bn(model)   # unfreeze bn

        loss += 0.0005 * wm_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicts.eq(targets.data).cpu().sum()

        if (iteration % print_frq == 0):
            print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                    % (iteration, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    '''Model test'''    
    print("Test model...")
    test(model, criterion, testloader, device)

    '''MTLSign test'''
    print("Test wm embedding acc...")
    MTL.wmend.eval()
    wm_correct = 0
    wm_total = 0
    for idx, (wm_inputs, wm_targets) in enumerate(wmloader):
        wm_inputs, wm_targets = wm_inputs.to(device), wm_targets.to(device)

        wm_outputs = MTL.verify(model, wm_inputs, device)
        _, predicts = torch.max(wm_outputs.data, 1)
        wm_total += wm_targets.size(0)
        wm_correct += predicts.eq(wm_targets.data).cpu().sum()

    print("wm acc: %.3f%% (%d / %d)"
            % (100. * wm_correct / wm_total, wm_correct, wm_total))
       

'''Model save'''
ckp = {
    'model_state_dict': model.state_dict(),
    'wm_bacend': MTL.wmend.state_dict(),
}

torch.save(ckp, os.path.join('./checkpoint', 'wm8192_id' + str(args.id) + '.pth'))