'''
    helper funtions including:
      - adjust_lr_rate(): adjust learning rate
      - loader_show(): show images in loader
      - test(): test [model] in [loader] on [device] using [criterion]
      - freeze_bn(): freeze batchnormalize layers in [model]
      - unfreeze_bn(): freeze batchnormalize layers in [model]
'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


def adjust_lr_rate(init_lr, optimizer, epoch, lradj):
    '''Set the learning rate to the initial LR decayed by every [lradj=20] epochs'''
    lr = init_lr * (0.1 ** (epoch // lradj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   


def loader_show(loader, mean, std, n=8):
    '''show first [n] images form loader'''
    (images, labels) = next(iter(loader))
    images, labels = images.cpu(), labels.cpu()
    # denormalize
    for i in range(images.shape[1]):
        images[:, i, :, :] = images[:, i, :, :] * std[i] + mean[i]
    # show images
    img_batch = images[0:n]
    print(labels[0:n])
    grid = torchvision.utils.make_grid(img_batch, nrow=4)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.show()



def test(model, criterion, loader, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    for idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        test_total += targets.size(0)
        test_correct += predicts.eq(targets.data).cpu().sum()

    print("[Test] Loss: %.3f | Acc: %.3f%% (%d / %d)" 
            % (test_loss / (idx + 1), 100. * test_correct / test_total, test_correct, test_total))
    
    acc = 100. * test_correct / test_total
    return acc



def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()