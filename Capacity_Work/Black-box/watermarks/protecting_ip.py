"""Protecting Intellectual Property of Deep Neural Networks with Watermarking (Zhang et al., 2018)

- different wm_types: ('content', 'unrelated', 'noise')"""

from watermarks.base import WmMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import image_char, save_triggerset, get_size, find_tolerance, get_trg_set
from helpers.loaders import get_data_transforms, get_wm_transform
from helpers.transforms import EmbedText

from trainer import test, train, train_on_augmented, train_on_augmented_overwrite, train_on_wms


class ProtectingIP(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set', 'protecting_ip')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.wm_type = args.wm_type  # content, unrelated, noise
        self.ao = args.ao
        self.p = None

    def gen_watermarks(self, device):
        logging.info('Generating watermarks. Type = ' + self.wm_type)
        datasets_dict = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100, 'mnist': datasets.MNIST}

        # in original: one trigger label for ALL trigger images. went with label_watermark=lambda w, x: (x + 1) % 10
        # trigger_lbl = 1  # "airplane"

        if self.wm_type == 'content':
            wm_dataset = self.dataset

            if self.dataset == "cifar10" or self.dataset == "cifar100":
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    EmbedText("TEST", (0, 22), 0.5),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            elif self.dataset == "mnist":
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    EmbedText("TEST", (0, 18), 0.5),
                ])


        elif self.wm_type == 'unrelated':
            if self.dataset == 'mnist':
                wm_dataset = 'cifar10'
                # normalize like cifar10, crop like mnist
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ])
            elif self.dataset == 'cifar10':
                wm_dataset = 'mnist'
                # crop like cifar10, make rgb
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        elif self.wm_type == 'noise':
            wm_dataset = self.dataset
            # add gaussian noise to trg images
            transform_train, _ = get_data_transforms(self.dataset)

            if self.dataset == 'mnist':
                transform_watermarked = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + torch.randn_like(x))
                ])

            elif self.dataset == 'cifar10':
                transform_watermarked = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + torch.randn_like(x)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])

        elif self.wm_type == 'origin':
            wm_dataset = self.dataset
            transform_watermarked = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
            ])


        wm_set = datasets_dict[wm_dataset](root='./data', train=True, download=True, transform=transform_watermarked)

        '''
        img, lbl = wm_set[1]
        img = img.to(device)
        trg_lbl = (lbl + 1) % self.num_classes
        self.trigger_set.append((img, torch.tensor(trg_lbl)))
        '''

        for i in random.sample(range(len(wm_set)), len(wm_set)):  # iterate randomly
            img, lbl = wm_set[i]
            img = img.to(device)

            trg_lbl = (lbl + 1) % self.num_classes  # set trigger labels label_watermark=lambda w, x: (x + 1) % 10
            self.trigger_set.append((img, torch.tensor(trg_lbl)))

            if len(self.trigger_set) == self.size:
                break  # break for loop when trigger set has final size

        if self.save_wm:
            save_triggerset(self.trigger_set, os.path.join(self.path, self.arch, self.wm_type), self.runname)
            print('watermarks generation done')

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        # self.gen_watermarks(device) # old
        transform = get_wm_transform('ProtectingIP', self.dataset)

        self.trigger_set = get_trg_set(os.path.join(self.path, self.arch, self.wm_type, self.runname), 'labels.txt', self.size,
                                       transform=transform)

        self.loader()

        if self.embed_type == 'pretrained':
            # load model
            logging.info("Load model: " + self.loadmodel + ".pth")
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.pth')))

        real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model,
                                                               self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch


    def overwrite(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        # self.gen_watermarks(device) # old
        transform = get_wm_transform('ProtectingIP', self.dataset)

        trigger_set0 = get_trg_set(os.path.join(self.path, self.arch, self.wm_type, self.loadmodel), 'labels.txt', self.size,
                                       transform=transform)
        logging.info('Loading Original WM dataset.')
        wm_loader0 = torch.utils.data.DataLoader(trigger_set0, batch_size=self.wm_batch_size,
                                                     shuffle=True)

        self.trigger_set = get_trg_set(os.path.join(self.path, self.arch, self.wm_type, self.runname), 'labels.txt', self.size,
                                       transform=transform)
        logging.info('Loading New WM dataset.')
        self.wm_loader = torch.utils.data.DataLoader(self.trigger_set, batch_size=self.wm_batch_size,
                                                     shuffle=True)  

        if self.embed_type == 'pretrained':
            # load model
            logging.info("Load model: " + self.loadmodel + ".pth")
            net.load_state_dict(torch.load(os.path.join('checkpoint', self.loadmodel + '.pth')))

        if self.ao:
            logging.info("Start AO attack...")
            real_acc, wm_acc, val_loss, epoch, self.history = train_on_wms(self.epochs_w_wm, device, net, optimizer, criterion, scheduler, 
                                                                    self.wm_loader, test_loader, wm_loader0, save_dir, self.save_model, self.history)
        else: 
            real_acc, wm_acc, val_loss, epoch, self.history = train_on_augmented_overwrite(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, wm_loader0, self.wm_loader, save_dir, self.save_model,
                                                               self.history)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch