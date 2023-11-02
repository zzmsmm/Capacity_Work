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

'''configuration setting'''
parser = argparse.ArgumentParser(description='Protocol test')
parser.add_argument('--train_db_path', default='./data', help='the root path of the trainning data')
parser.add_argument('--test_db_path', default='./data', help='the root path of the testing data')
parser.add_argument('--dataset', default='cifar10', help='tain on [DATASET]')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lradj', type=int, default=20, help='multiple the lr by 0.1 every [LRADJ] epochs')
parser.add_argument('--max_epochs', type=int, default=60, help='the maximun epochs')
parser.add_argument('--batchsize', type=int, default=100, help='batchsize')
parser.add_argument('--save_dir', default='./checkpoint', help='the root path of saved models')
parser.add_argument('--save_model', default='model.pth', help='test model')
parser.add_argument('--wm', action='store_true', help='train with wm')
parser.add_argument('--log_dir', default='./log', help='the root path of log')
parser.add_argument('--conf', default='config.txt', help='config file')
parser.add_argument('--runname', default='embed1024', help='the running program name')
parser.add_argument('--id', type=int, default=0, help='proccess id')
args = parser.parse_args()

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logfile = os.path.join(LOG_DIR, str(args.runname) + '_id' + str(args.id) + '.txt')
configfile = args.conf

# save the configuration parameters
with open(configfile, 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')


'''model setting'''
model = ResNet50_cifar10()
trainset, testset, n_classes = model.getdataset(args.dataset, args.train_db_path, args.test_db_path)
trainloader, testloader= model.getdataloader(trainset, testset, args.batchsize)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model.net = model.net.to(device)
if device == 'cuda':
    cudnn.benchmark = True   # use cudnn to accelerate conv

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


'''uchida setting'''
uchida = Uchida()
b_path = os.path.join('save_dir', 'wm1024', 'uchida_b_1024.pth')
x_path = os.path.join('save_dir', 'wm1024', 'uchida_x_1024.pth')
uchida.B = torch.load(b_path)
uchida.X = torch.load(x_path)


'''model taining'''
start_epoch = 0
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    # adjust the learning rate in every [lrafj] epoch
    adjust_lr_rate(args.lr, optimizer, epoch, args.lradj)

    # train
    model.train(epoch, criterion, optimizer, logfile, trainloader, device, args.wm, uchida)

    # test
    acc = model.test(criterion, logfile, testloader, device)
    print(f"Test acc: {acc:.2f}%")
    # verify
    ex_b = uchida.extract(model.net, uchida.X)
    wm_acc = uchida.verify(ex_b, logfile)
    print(f"Verify acc: {wm_acc:.2f}%")

    # save checkpoint
    if (epoch == (start_epoch + args.max_epochs - 1)):
        print("Saving...")
        model.save(acc, epoch, device, args.save_dir, args.id)