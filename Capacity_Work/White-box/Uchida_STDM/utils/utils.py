'''
    helper funtions including:
      - adjust_lr_rate(): adjust learning rate
      - loader_show(): show images in loader
'''
import matplotlib.pyplot as plt
import torch
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