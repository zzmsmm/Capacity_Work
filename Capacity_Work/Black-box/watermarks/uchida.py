from watermarks.base import WmMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from trainer import train_whitebox, train_whitebox_overwrite
from helpers.utils import find_tolerance

class Uchida(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'white_box', 'uchida')
        os.makedirs(self.path, exist_ok=True)

        self.bit_length = args.bit_length

    def gen_watermarks(self, net, device):
        # generate b
        b = np.random.randint(0, 2, self.bit_length)
        # print(b)
        # print(self.arch)
        '''
        for name, param in net.named_parameters():
            if 'weight' in name or True:
                print(name)
                param = param.cpu().detach().numpy()
                print(param.shape)
        '''

        if self.arch == 'cnn_mnist' or self.arch == 'cnn_cifar10':
            for name, param in net.named_parameters():
                if 'conv_layer.6.weight' in name:
                    param = param.cpu().detach().numpy()
                    w = np.mean(param.reshape(128, 576), axis=0)
                    wm_col = np.prod(w.shape)
                    print(wm_col)

        elif self.arch == 'resnet18':
            for name, param in net.named_parameters():
                if 'layer2.0.conv1.weight' in name:
                    param = param.cpu().detach().numpy()
                    w = np.mean(param.reshape(128, 576), axis=0)
                    wm_col = np.prod(w.shape)
                    print(wm_col)
        
        X = np.random.randn(self.bit_length, wm_col)

        if self.save_wm:
            path = os.path.join(self.path, self.arch, self.runname)
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, 'b.npy'), b)
            np.save(os.path.join(path, 'X.npy'), X)
        
        print('watermarks generation done')

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):
        if self.embed_type == 'pretrained':
            logging.info("Load model: " + self.loadmodel + ".pth")
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.pth')))
        
        path = os.path.join(self.path, self.arch, self.runname)
        b = np.load(os.path.join(path, 'b.npy'))
        X = np.load(os.path.join(path, 'X.npy'))

        real_acc, wm_acc, val_loss, epoch, self.history = train_whitebox(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, b, X, save_dir, self.save_model,
                                                               self.history, self.arch, self.bit_length)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch


    def overwrite(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):
        if self.embed_type == 'pretrained':
            logging.info("Load model: " + self.loadmodel + ".pth")
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.pth')))
        
        path1 = os.path.join(self.path, self.arch, self.runname)
        b = np.load(os.path.join(path1, 'b.npy'))
        X = np.load(os.path.join(path1, 'X.npy'))

        path2 = os.path.join(self.path, self.arch, self.loadmodel)
        b0 = np.load(os.path.join(path2, 'b.npy'))
        X0 = np.load(os.path.join(path2, 'X.npy'))

        real_acc, wm_acc, val_loss, epoch, self.history = train_whitebox_overwrite(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, b, X, b0, X0, save_dir, self.save_model,
                                                               self.history, self.arch, self.bit_length)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch


    def verify(self, net, device):
        logging.info("Verifying watermark.")

        logging.info("Loading saved model.")
        net.load_state_dict(torch.load(os.path.join('checkpoint', self.save_model + '.pth')))

        path = os.path.join(self.path, self.arch, self.runname)
        b = np.load(os.path.join(path, 'b.npy'))
        X = np.load(os.path.join(path, 'X.npy'))

        false_preds = 0

        if self.arch == 'cnn_mnist' or self.arch == 'cnn_cifar10':
            for name, param in net.named_parameters():
                if 'conv_layer.6.weight' in name:
                    param = param.cpu().detach().numpy()
                    w = np.mean(param.reshape(128, 576), axis=0)
        elif self.arch == 'resnet18':
            for name, param in net.named_parameters():
                if 'layer2.0.conv1.weight' in name:
                    param = param.cpu().detach().numpy()
                    w = np.mean(param.reshape(128, 576), axis=0)

        pre = np.int64(np.dot(X, w) > 0)
        false_preds = self.bit_length - np.sum(pre == b)

        theta = find_tolerance(self.bit_length, self.thresh)

        logging.info("False preds: %d. Watermark verified (tolerance: %d)? %r" % (false_preds, theta,
                                                                                  (false_preds < theta).item()))

        success = false_preds < theta

        return success, false_preds, theta



