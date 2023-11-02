import argparse
import os

import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models.resnet import ResNet50
from models import ResNet50_cifar10
from utils import *
from watermark import STDM

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
parser.add_argument('--conf', default='stdm_config.txt', help='config file')
parser.add_argument('--runname', default='stdm4096_embed', help='the running program name')
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

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model.net = model.net.to(device)
if device == 'cuda':
    cudnn.benchmark = True   # use cudnn to accelerate conv

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


'''stdm setting'''
stdm = STDM()
b_path = os.path.join('save_dir', 'stdm4096', 'stdm_b_4096.pth')
x_path = os.path.join('save_dir', 'stdm4096', 'stdm_x_4096.pth')
stdm.B = torch.load(b_path)
stdm.X = torch.load(x_path)


'''model taining'''
start_epoch = 0
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    # adjust the learning rate in every [lrafj] epoch
    adjust_lr_rate(args.lr, optimizer, epoch, args.lradj)

    # train
    print(f"\nEpoch: {epoch}")
    model.net.train()
    iteration = 0
    train_loss = 0
    embed_loss = 0
    correct = 0
    total = 0 
    print_frq = 50

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model.net(inputs)
        loss = criterion(outputs, targets)
        
        wm_loss = stdm.embed(model.net, stdm.X, stdm.B, device)
        
        loss += 0.1 * wm_loss  
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        embed_loss += wm_loss.item()
        _, predict = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predict.eq(targets.data).cpu().sum()
        
        if (iteration % print_frq == 0):
            print("wm loss: %.2f" % (embed_loss / (batch_idx + 1)))
            print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                    % (iteration, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write(f"[train] Epoch: {epoch}\n")
        f.write("[train] Loss: %.3f | Acc: %.3f%% (%d / %d)\n" 
                    % (train_loss / iteration, 100. * correct / total, correct, total))

    # test
    acc = model.test(criterion, logfile, testloader, device)
    print(f"Test acc: {acc:.2f}%")

    # verify watermark
    ex_b = stdm.extract(model.net, stdm.X)
    wm_acc = stdm.verify(ex_b, logfile)
    print(f"Verify acc: {wm_acc:.2f}%")

    
# save model
state = {
    'acc': acc,
    'epoch': epoch,
    'model_state_dict': model.net.state_dict(),
}
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
torch.save(state, os.path.join(args.save_dir, 'stdm_4096_id' + str(args.id) + '.pth'))