import logging
import pickle
import time

import torch

import models
import os
import argparse

import torch.nn as nn
import torch.optim as optim
import numpy as np

from attacks.pruning import prune
from attacks.fine_tuning import fine_tune, fine_tune_whitebox

# set up argument parser
from helpers.loaders import get_data_transforms, get_dataloader, get_dataset, get_wm_transform, get_wm_path
from helpers.utils import set_up_logger, save_results, get_trg_set, save_obj, set_up_optim_sched, re_initializer_layer
from trainer import test, whitebox_wm_acc
from watermarks.exponential_weighting import load_trg_indices

parser = argparse.ArgumentParser(description='Perform attacks on models.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [default: cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='densenet')
parser.add_argument('--attack_type', default='pruning', help='attack type. choices: pruning, fine-tuning')
parser.add_argument('--pruning_rates', nargs='+', default=[0.1], type=float, help='percentages (list) of how many weights to prune')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size for fine-tuning')
parser.add_argument('--wm_batch_size', default=32, type=int, help='batch size for fine-tuning')
parser.add_argument('--num_epochs', default=1, type=int, help='number of epochs for fine-tuning')
parser.add_argument('--patience', default=50, type=int, help='patience for transfer learning')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--bit_length', default=256, type=int)

parser.add_argument('--trg_set_size', default=100, type=int, help='the size of the trigger set (default: 100)')
parser.add_argument('--loadmodel', default=None, help='the model which the attack should be performed on')
parser.add_argument('--save_file', default="save_results_attacks.csv", help='file for saving results')
parser.add_argument('--wm_type', help='e.g. content, noise, unrelated')
parser.add_argument('--method', help='watermarking method')
parser.add_argument('--eps', help='eps for watermarking method')
parser.add_argument('--pattern_size', default=128, help='pattern size or num bits')
parser.add_argument('--save_model', action='store_true', help='save attacked model?')

parser.add_argument('--tunealllayers', action='store_true', help='fine-tune all layers')
parser.add_argument('--reinitll', action='store_true', help='re initialize the last layer')

parser.add_argument('--cuda')

args = parser.parse_args()

device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'  # cuda:0

cwd = os.getcwd()

net = models.__dict__[args.arch](num_classes=args.num_classes)

if device == 'cpu':
    net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth'), map_location=torch.device('cpu')))
else:
    net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth')))

net = net.to(device)

if args.method == 'ExponentialWeighting':
    t = 2.0  # from original implementation
    net.enable_ew(t)
    logging.info('Exponential weighting done')

# set up loss and optimizer
criterion = nn.CrossEntropyLoss()
# set up optimizer and scheduler
optimizer, scheduler = set_up_optim_sched(args, net)

# create log file
cwd = os.getcwd()
log_dir = os.path.join(cwd, 'log')
os.makedirs(log_dir, exist_ok=True)
if args.attack_type == 'pruning':
    info = str(args.attack_type) + '_' + str(args.loadmodel) + '_pruned'
elif args.attack_type == 'fine-tuning':
    info = str(args.attack_type) + '_' + str(args.loadmodel) + '_transfer-learned_' + str(args.lr) + '_tunealllayers_' + str(args.tunealllayers)
elif args.attack_type == 'fine-pruning':
    for pruning_rate in args.pruning_rates:
        info = str(args.attack_type) + '_' + str(args.loadmodel) + '_fine-pruned_' + str(pruning_rate) + '_lr_' + str(args.lr)

logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_' + info + '.txt')
set_up_logger(logfile)

logging.info('Start attack. Attack type: ' + args.attack_type)

# prepare test loader
train_db_path = os.path.join(cwd, 'data')
test_db_path = os.path.join(cwd, 'data')
transform_train, transform_test = get_data_transforms(args.dataset)
train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, 0.1)
_, test_loader, _ = get_dataloader(train_set, test_set, args.batch_size, valid_set, shuffle=True)

# prepare wm loader
if args.method == 'Uchida':
    wm_path = get_wm_path(args.method, args.loadmodel, args.arch, wm_type=args.wm_type)
    b = np.load(os.path.join(wm_path, 'b.npy'))
    X = np.load(os.path.join(wm_path, 'X.npy'))
else:  
    transform = get_wm_transform(args.method, args.dataset)
    wm_path = get_wm_path(args.method, args.loadmodel, args.arch, wm_type=args.wm_type)
    trigger_set = get_trg_set(wm_path, 'labels.txt', args.trg_set_size, transform)
    wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)


if args.attack_type == 'pruning':

    for pruning_rate in args.pruning_rates:
        # reload original model
        net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth')))

        pruning_rate = float(pruning_rate)
        start_time = time.time()
        prune(net, args.arch, pruning_rate)
        attack_time = time.time() - start_time

        if args.method == 'ProtectingIP' and args.wm_type == 'content':
            savepath = os.path.join('capacity_log', args.dataset + '_resnet50', 'content', 'NP')
        elif args.method == 'ProtectingIP' and args.wm_type == 'noise':
            savepath = os.path.join('capacity_log', args.dataset + '_resnet50', 'noise', 'NP')
        elif args.method == 'FrontierStitching':
            savepath = os.path.join('capacity_log', args.dataset + '_resnet50', 'frontier-stitching', 'NP')
        elif args.method == 'ExponentialWeighting':
            savepath = os.path.join('capacity_log', args.dataset + '_resnet50', 'exponential-weighting', 'NP')

        os.makedirs(savepath, exist_ok=True)
        savepath = f'{savepath}/{args.loadmodel}.txt'

        # check test_acc
        logging.info('Testing on test set.')
        test_acc = test(net, criterion, test_loader, device)

        # check wm_acc
        if args.method == 'Uchida':
            wm_acc = whitebox_wm_acc(net, args.arch, b, X, args.bit_length)
        else:
            logging.info('Testing on trigger set.')
            wm_acc = test(net, criterion, wm_loader, device)

        with open(savepath, 'a') as f:
            f.write('%.2f %.2f %.2f\n' % (pruning_rate, test_acc, wm_acc))

        # save results
        # csv_args = [getattr(args, arg) for arg in vars(args)] + [pruning_rate, test_acc.item(), wm_acc.item(), attack_time]
        # save_results(csv_args, os.path.join(cwd, args.save_file))
        if args.save_model:
            logging.info('Saving attacked model.')
            torch.save(net.state_dict(), os.path.join(cwd, 'checkpoint', info + '_' + str(pruning_rate) + '.pth'))

elif args.attack_type == 'fine-tuning':

    # net.unfreeze_model()

    start_time = time.time()

    if args.method == 'Uchida':
        test_acc, val_loss, wm_acc, test_acc_orig, epoch, history = fine_tune_whitebox(net, device, criterion, optimizer,
                                                                                  scheduler, test_loader,
                                                                                  args.dataset, args.batch_size,
                                                                                  args.num_epochs, args.patience,
                                                                                  os.path.join(cwd, 'checkpoint',
                                                                                               info + '.pth'), args.arch, b, X, args.bit_length,
                                                                                  tune_all=args.tunealllayers)
    else:
        test_acc, val_loss, wm_acc, test_acc_orig, epoch, history = fine_tune(net, device, criterion, optimizer,
                                                                                  scheduler, test_loader,
                                                                                  args.dataset, args.batch_size,
                                                                                  args.num_epochs, args.patience,
                                                                                  os.path.join(cwd, 'checkpoint',
                                                                                               info + '.pth'), args.loadmodel, wm_loader,
                                                                                  tune_all=args.tunealllayers, transfer=None)
    attack_time = time.time() - start_time

    #csv_args = [getattr(args, arg) for arg in vars(args)] + [test_acc.item(), test_acc_orig.item(), wm_acc.item(),
    #                                                         val_loss, epoch, attack_time]
    #save_results(csv_args, os.path.join(cwd, args.save_file))
    #save_obj(history, info)

elif args.attack_type == 'fine-pruning':
    for pruning_rate in args.pruning_rates:
        # reload original model
        net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', args.loadmodel + '.pth')))

        pruning_rate = float(pruning_rate)
        start_time = time.time()
        prune(net, args.arch, pruning_rate)
        
        logging.info('Testing after pruning')
        # check test_acc
        logging.info('Testing on test set.')
        test_acc = test(net, criterion, test_loader, device)

        # check wm_acc
        if args.method == 'Uchida':
            wm_acc = whitebox_wm_acc(net, args.arch, b, X, args.bit_length)
        else:
            logging.info('Testing on trigger set.')
            wm_acc = test(net, criterion, wm_loader, device)

        logging.info('Fine-tuning start')

        if args.method == 'Uchida':
            test_acc, val_loss, wm_acc, test_acc_orig, epoch, history = fine_tune_whitebox(net, device, criterion, optimizer,
                                                                                  scheduler, test_loader,
                                                                                  args.dataset, args.batch_size,
                                                                                  args.num_epochs, args.patience,
                                                                                  os.path.join(cwd, 'checkpoint',
                                                                                               info + '.pth'), args.arch, b, X, args.bit_length,
                                                                                  tune_all=args.tunealllayers, transfer=None)
        else:
            test_acc, val_loss, wm_acc, test_acc_orig, epoch, history = fine_tune(net, device, criterion, optimizer,
                                                                                  scheduler, test_loader,
                                                                                  args.dataset, args.batch_size,
                                                                                  args.num_epochs, args.patience,
                                                                                  os.path.join(cwd, 'checkpoint',
                                                                                               info + '.pth'), wm_loader,
                                                                                  tune_all=args.tunealllayers, transfer=None)
        attack_time = time.time() - start_time

        csv_args = [getattr(args, arg) for arg in vars(args)] + [test_acc.item(), test_acc_orig.item(), wm_acc.item(),
                                                             val_loss, epoch, attack_time]
        save_results(csv_args, os.path.join(cwd, args.save_file))
        save_obj(history, info)
       
