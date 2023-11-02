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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Watermark capacity test')
parser.add_argument('--wm_lbl', default='./save_dir/wm4096/index.txt', help='wm index')
parser.add_argument('--wm_model', default='./checkpoint/wm8192_id1.pth', help='test model')
parser.add_argument('--id', default=0, type=int, help='process id')
args = parser.parse_args()

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

train_len = len(trainset)
trainset, _ = torch.utils.data.random_split(trainset, [int(0.1*train_len), int(0.9*train_len)])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                            shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers=4)


'''QRdataset setting'''
qr_transform = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
wmset = QRset(args.wm_lbl, transform=qr_transform)
wmloader = torch.utils.data.DataLoader(
    dataset=wmset,
    batch_size=100,
    shuffle=True
)


'''Load pretrained model'''
checkpoint_path = args.wm_model
checkpoint = torch.load(checkpoint_path)
model = ResNet50().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
MTL = MTLSign()
MTL.wmend.load_state_dict(checkpoint['wm_bacend'])
criterion = nn.CrossEntropyLoss()


'''Fine tuning attack'''
ft_epoch = 40
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

logdir = 'mtl_ft4096' + '_id_' + str(args.id)
savedir = os.path.join('log', logdir)
os.makedirs(savedir, exist_ok=True)
logfile = 'epoch_acc_wm.txt'
savepath = os.path.join('log', logdir, logfile)

for epoch in range(ft_epoch):
    '''Model test'''    
    print("Test model...")
    acc = test(model, criterion, testloader, device)

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

    wm_acc = 100. * wm_correct / wm_total
    print("wm acc: %.3f%% (%d / %d)"
            % (100. * wm_correct / wm_total, wm_correct, wm_total))
    
    '''save log'''
    with open(savepath, 'a') as f:
        f.write('%d %.2f %.2f\n' % (epoch, acc.item(), wm_acc.item()))

    '''Model Fine-Tuning'''
    if (epoch + 1) % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    model.train()
    iteration = 0
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    


'''draw data'''
x = []
y1 = []
y2 = []
with open(savepath, 'r') as f:
    for line in f:
        words = line.split()
        x.append(int(words[0]))
        y1.append(float(words[1]))
        y2.append(float(words[2]))

plt.plot(x, y1, color='r', label='testset acc')
plt.plot(x, y2, color='b', label='wmset acc')
plt.legend()
plt.xlabel('adv epoch')
plt.ylabel('accuracy')
plt.show()

draw_file = 'pic.png'
draw_save_path = os.path.join('log', logdir, draw_file)
plt.savefig(draw_save_path)