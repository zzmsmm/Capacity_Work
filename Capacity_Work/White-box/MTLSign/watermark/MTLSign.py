import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data
from PIL import Image
import copy

from models import *

class QRset(data.Dataset):
    def __init__(self, lbl_path, transform=None, target_transform=None):
        imgs = []
        with open(lbl_path, 'r') as f:
            for line in f:
                line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    


class Extract(torch.nn.Module):
    def __init__(self, block, num_blocks):
        super(Extract, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1 )
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        z1 = self.layer1(out)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)

        z1 = z1.reshape(z1.size(0), -1)
        z2 = z2.reshape(z2.size(0), -1)
        z3 = z3.reshape(z3.size(0), -1)
        z4 = z4.reshape(z4.size(0), -1)
        z = torch.cat([z1, z2, z3, z4], dim=1)
        return z



class Backend(torch.nn.Module):
    def __init__(self):
        super(Backend, self).__init__()
        '''input demension: 64*32*32+128*16*16+256*8*8+512*4*4'''
        self.n_input = 122880 * 4
        self.fc = nn.Linear(self.n_input, 2)

    def forward(self, x):
        out = self.fc(x)
        return out



class MTLSign(object):
    def __init__(self):
        super(MTLSign, self).__init__()
        self.copy_layer1 = []
        self.copy_layer2 = []
        self.copy_layer3 = []
        self.copy_layer4 = []
        self.wmend = Backend()
        self.mona = Extract(Bottleneck, [3, 4, 6, 3])

    def copy_init_weight(self, model):
        for param in model.layer1.parameters():
            temp = copy.deepcopy(param)
            self.copy_layer1.append(temp)
        
        for param in model.layer2.parameters():
            temp = copy.deepcopy(param)
            self.copy_layer2.append(temp)

        for param in model.layer3.parameters():
            temp = copy.deepcopy(param)
            self.copy_layer3.append(temp)

        for param in model.layer4.parameters():
            temp = copy.deepcopy(param)
            self.copy_layer4.append(temp)


    def Rfunc(self, model, l=1):
        loss = 0

        if (self.copy_layer1 != []):
            idx = 0
            for param in model.layer1.parameters():
                loss += l * torch.sum((param - self.copy_layer1[idx])**2)
                idx += 1
        if (self.copy_layer2 != []):
            idx = 0
            for param in model.layer2.parameters():
                loss += l * torch.sum((param - self.copy_layer2[idx])**2)
                idx += 1
        if (self.copy_layer3 != []):
            idx = 0
            for param in model.layer3.parameters():
                loss += l * torch.sum((param - self.copy_layer3[idx])**2)
                idx += 1
        if (self.copy_layer4 != []):
            idx = 0
            for param in model.layer4.parameters():
                loss += l * torch.sum((param - self.copy_layer4[idx])**2)
                idx += 1
        
        return loss
    

    def verify(self, model, x, device):
        self.wmend.to(device)
        self.mona.to(device)
        self.mona.conv1 = model.conv1
        self.mona.bn1 = model.bn1
        self.mona.layer1 = model.layer1
        self.mona.layer2 = model.layer2
        self.mona.layer3 = model.layer3
        self.mona.layer4 = model.layer4
        z = self.mona(x)
        y = self.wmend(z)
        
        return y