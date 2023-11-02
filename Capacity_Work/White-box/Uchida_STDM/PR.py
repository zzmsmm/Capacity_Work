import argparse
import os

import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import torch.optim as optim

#from models.resnet import ResNet18
from models.resnet import ResNet50
from models import ResNet50_cifar10
from utils import *
from watermark import Uchida
from watermark import STDM

from pruning import prune

parser = argparse.ArgumentParser(description='pruning attack')
parser.add_argument('--train_db_path', default='./data', help='the root path of the trainning data')
parser.add_argument('--test_db_path', default='./data', help='the root path of the testing data')
parser.add_argument('--dataset', default='cifar10', help='tain on [DATASET]')
parser.add_argument('--model', default='checkpoint/model.pth')
parser.add_argument('--wm', action='store_true', help='train with wm')
parser.add_argument('--b', default='save_dir/stdm2048/stdm_b_2048.pth')
parser.add_argument('--x', default='save_dir/stdm2048/stdm_x_2048.pth')
parser.add_argument('--id', type=int, default=0, help='proccess id')
parser.add_argument('--log_dir', default='./log', help='the root path of log')
args = parser.parse_args()

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'stdm_pr2048' + '_id' + str(args.id) + '.txt')

logdir = 'stdm_pr2048' + '_id_' + str(args.id)
savedir = os.path.join(LOG_DIR, logdir)
os.makedirs(savedir, exist_ok=True)
savefile = 'rate_acc_wm'  + '.txt'
savepath = os.path.join(savedir, savefile)

rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 
         0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78,
         0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
         0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

for rate in rates:
    model = ResNet50_cifar10()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.net = model.net.to(device)

    ckp = torch.load(args.model, map_location=device)
    dict_ckp = ckp['model_state_dict']
    model.net.load_state_dict(dict_ckp)

    criterion = nn.CrossEntropyLoss()
    trainset, testset, n_classes = model.getdataset(args.dataset, args.train_db_path, args.test_db_path)
    trainloader, testloader= model.getdataloader(trainset, testset, 100)

    '''uchida = Uchida()
    b_path = os.path.join('save_dir', 'wm1024', 'uchida_b_1024.pth')
    x_path = os.path.join('save_dir', 'wm1024', 'uchida_x_1024.pth')
    uchida.B = torch.load(b_path)
    uchida.X = torch.load(x_path)'''
    stdm = STDM()
    b_path = os.path.join('save_dir', 'stdm2048', 'stdm_b_2048.pth')
    x_path = os.path.join('save_dir', 'stdm2048', 'stdm_x_2048.pth')
    stdm.B = torch.load(b_path)
    stdm.X = torch.load(x_path)

    '''pruning'''
    prune(model.net, "resnet50", pruning_rate=rate)

    '''evaluation'''
    model.net.eval()
    # test
    acc = model.test(criterion, logfile, testloader, device)
    print(f"Test acc: {acc:.2f}%")
    # verify
    '''ex_b = uchida.extract(model.net, uchida.X)
    wm_acc = uchida.verify(ex_b, logfile)'''
    ex_b = stdm.extract(model.net, stdm.X)
    wm_acc = stdm.verify(ex_b, logfile)
    print(f"Verify acc: {wm_acc:.2f}%")

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
draw_save_path = os.path.join(savedir, draw_file)
plt.savefig(draw_save_path)