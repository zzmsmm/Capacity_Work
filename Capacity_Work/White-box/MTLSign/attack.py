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
parser.add_argument('--wm_lbl', default='./save_dir/wm8192/index.txt', help='wm index')
parser.add_argument('--adv_lbl', default='./save_dir/wm8192/attack.txt', help='adv index')
parser.add_argument('--wm_model', default='./checkpoint/wm256.pth', help='test model')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--adv_epoch', default=500, type=int, help='attack epoch')
parser.add_argument('--id', default=0, type=int, help='process id')
args = parser.parse_args()

'''Device setting'''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''Train/test dataloader setting'''
print("Using CIFAR10 dataset.")
test_db_path = './data'
batchsize = 100
num_classes = 10

transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

testset = torchvision.datasets.CIFAR10(root=test_db_path,train=False, download=True,
                                        transform=transform_test)

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

advset = QRset(args.adv_lbl, transform=qr_transform)
advloader = torch.utils.data.DataLoader(
    dataset=advset,
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

'''Model test'''
print("\nTest pretrained model...")    
test(model, criterion, testloader, device)

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


print("Test attack acc...")
MTL.wmend.eval()
adv_correct = 0
adv_total = 0
for idx, (adv_inputs, adv_targets) in enumerate(advloader):
    adv_inputs,adv_targets = adv_inputs.to(device), adv_targets.to(device)

    adv_outputs = MTL.verify(model, adv_inputs, device)
    _, predicts = torch.max(adv_outputs.data, 1)
    adv_total += adv_targets.size(0)
    adv_correct += predicts.eq(adv_targets.data).cpu().sum()

print("adv acc: %.3f%% (%d / %d)"
        % (100. * adv_correct / adv_total, adv_correct, adv_total))


'''Attack'''
wm_num = len(wmset)
adv_epoch = args.adv_epoch
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)

logdir = 'wm_num_' + str(wm_num) + '_id_' + str(args.id)
savedir = os.path.join('log', logdir)
os.makedirs(savedir, exist_ok=True)
logfile = 'attack_epoch_' + str(adv_epoch) + '_lr_' + str(lr) + '.txt'
savepath = os.path.join('log', logdir, logfile)

for epoch in range(adv_epoch):
    print(f"\n[attack] Epoch: {epoch}")
    '''Model Fine-Tuning'''
    model.train()
    optimizer.zero_grad()

    wm_loss = 0
    for idx, (adv_inputs, adv_targets) in enumerate(advloader):
        freeze_bn(model)   # freeze bn when verify
        adv_inputs, adv_targets = adv_inputs.to(device), adv_targets.to(device)
        adv_outputs = MTL.verify(model, adv_inputs, device)
        adv_loss = criterion(adv_outputs, adv_targets)

    unfreeze_bn(model)   # unfreeze bn
    loss = adv_loss
    loss.backward()
    optimizer.step()

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
    
    '''Attack test'''
    print("Test attack acc...")
    MTL.wmend.eval()
    adv_correct = 0
    adv_total = 0
    for idx, (adv_inputs, adv_targets) in enumerate(advloader):
        adv_inputs, adv_targets = adv_inputs.to(device), adv_targets.to(device)

        adv_outputs = MTL.verify(model, adv_inputs, device)
        _, predicts = torch.max(adv_outputs.data, 1)
        adv_total += adv_targets.size(0)
        adv_correct += predicts.eq(adv_targets.data).cpu().sum()

    print("adv acc: %.3f%% (%d / %d)"
        % (100. * adv_correct / adv_total, adv_correct, adv_total))

    with open(savepath, 'a') as f:
        f.write('%d %.2f %.2f\n' % (epoch, acc.item(), wm_acc.item()))


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