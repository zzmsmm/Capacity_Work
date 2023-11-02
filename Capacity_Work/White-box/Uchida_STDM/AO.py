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

parser = argparse.ArgumentParser(description='ao attack')
parser.add_argument('--train_db_path', default='./data', help='the root path of the trainning data')
parser.add_argument('--test_db_path', default='./data', help='the root path of the testing data')
parser.add_argument('--dataset', default='cifar10', help='tain on [DATASET]')
parser.add_argument('--model', default='checkpoint/model.pth')
parser.add_argument('--b', default='save_dir/stdm2048/stdm_b_2048.pth')
parser.add_argument('--x', default='save_dir/stdm2048/stdm_x_2048.pth')
parser.add_argument('--id', type=int, default=0, help='proccess id')
parser.add_argument('--log_dir', default='./log', help='the root path of log')
args = parser.parse_args()

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, 'stdm_ao2048' + '_id' + str(args.id) + '.txt')

#uchida = Uchida()
stdm = STDM()
b_path = os.path.join('save_dir', 'stdm2048', 'stdm_b_2048.pth')
x_path = os.path.join('save_dir', 'stdm2048', 'stdm_x_2048.pth')
#uchida.B = torch.load(b_path)
#uchida.X = torch.load(x_path)
stdm.B = torch.load(b_path)
stdm.X = torch.load(x_path)

'''adv_uchida = Uchida()
b, x = adv_uchida.keyGen()
adv_uchida.X = uchida.X'''
adv_stdm = STDM()
b, x = adv_stdm.keyGen()
adv_stdm.X = stdm.X

model = ResNet50_cifar10()
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model.net = model.net.to(device)

ckp = torch.load(args.model, map_location=device)
dict_ckp = ckp['model_state_dict']
model.net.load_state_dict(dict_ckp)

trainset, testset, n_classes = model.getdataset(args.dataset, args.train_db_path, args.test_db_path)
trainloader, testloader= model.getdataloader(trainset, testset, 100)

adv_epoch = 200
lr = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.parameters(), lr=lr)

logdir = 'stdm_ao2048' + '_id_' + str(args.id)
savedir = os.path.join(LOG_DIR, logdir)
os.makedirs(savedir, exist_ok=True)
savefile = 'epoch_acc_wm'  + '.txt'
savepath = os.path.join(savedir, savefile)

for epoch in range(adv_epoch):
    model.net.eval()
    # test
    acc = model.test(criterion, logfile, testloader, device)
    print(f"Test acc: {acc:.2f}%")
    # verify
    #ex_b = uchida.extract(model.net, uchida.X)
    ex_b = stdm.extract(model.net, stdm.X)
    #wm_acc = uchida.verify(ex_b, logfile)
    wm_acc = stdm.verify(ex_b, logfile)
    print(f"Verify acc: {wm_acc:.2f}%")

    '''ex_b = adv_uchida.extract(model.net, adv_uchida.X)
    adv_acc = adv_uchida.verify(ex_b, logfile)'''
    ex_b = adv_stdm.extract(model.net, adv_stdm.X)
    adv_acc = adv_stdm.verify(ex_b, logfile)
    print(f"Adv wm acc: {adv_acc:.2f}%")


    with open(savepath, 'a') as f:
        f.write('%d %.2f %.2f\n' % (epoch, acc.item(), wm_acc.item()))

    model.net.train()
    optimizer.zero_grad()
    #loss = uchida.embed(model.net, adv_uchida.X, adv_uchida.B, device)
    loss = stdm.embed(model.net, adv_stdm.X, adv_stdm.B, device)
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
draw_save_path = os.path.join(savedir, draw_file)
plt.savefig(draw_save_path)