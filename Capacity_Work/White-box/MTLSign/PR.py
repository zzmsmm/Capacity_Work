import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import ResNet50
from utils import *
from watermark import MTLSign, QRset
from pruning import prune

import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Watermark capacity test')
parser.add_argument('--wm_lbl', default='./save_dir/wm4096/index.txt', help='wm index')
parser.add_argument('--wm_model', default='./checkpoint/wm256.pth', help='test model')
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

logdir = 'mtl_pr4096_' + 'id_' + str(args.id)
savedir = os.path.join('log', logdir)
os.makedirs(savedir, exist_ok=True)
logfile = 'rate_' + 'acc_' + 'wmacc' + '.txt'
savepath = os.path.join('log', logdir, logfile)


for rate in np.linspace(0, 1, 50):
    '''Load pretrained model'''
    checkpoint_path = args.wm_model
    checkpoint = torch.load(checkpoint_path)
    model = ResNet50().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    MTL = MTLSign()
    MTL.wmend.load_state_dict(checkpoint['wm_bacend'])
    criterion = nn.CrossEntropyLoss()

    '''pruning'''
    prune(model, "resnet50", pruning_rate=rate)

    '''Model test'''    
    print("Test model...")
    acc = test(model, criterion, testloader, device)

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
    
    with open(savepath, 'a') as f:
        f.write('%.2f %.2f %.2f\n' % (rate, acc.item(), wm_acc.item()))


'''draw data'''
x = []
y1 = []
y2 = []
with open(savepath, 'r') as f:
    for line in f:
        words = line.split()
        x.append(float(words[0]))
        y1.append(float(words[1]))
        y2.append(float(words[2]))

plt.plot(x, y1, color='r', label='testset acc')
plt.plot(x, y2, color='b', label='wmset acc')
plt.legend()
plt.xlabel('pruning rate')
plt.ylabel('accuracy')
plt.show()

draw_file = 'pic.png'
draw_save_path = os.path.join('log', logdir, draw_file)
plt.savefig(draw_save_path)