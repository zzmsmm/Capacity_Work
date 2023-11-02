import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from .resnet import ResNet50


class ResNet50_cifar10(object):
    def __init__(self):
        super(ResNet50_cifar10, self).__init__()
        self.net = ResNet50(num_classes=10)


    '''data process related'''
    def _gettransforms(self, datatype):
        transform_train, transform_test = None, None
        if datatype.lower() == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            pass

        return transform_train, transform_test


    def getdataset(self, dataset, train_db_path, test_db_path):
        # get transformations
        transform_train, transform_test = self._gettransforms(datatype=dataset)  
        n_classes = 0

        if dataset.lower() == 'cifar10':
            print("Using CIFAR10 dataset.")
            trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                    train=True, download=True,
                                                    transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                                   train=False, download=True,
                                                   transform=transform_test)
            n_classes = 10
            return trainset, testset, n_classes
        else:
            print("Dataset is not supported.")
            return None, None, None


    def getdataloader(self, trainset, testset, batchsize):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                  shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                                 shuffle=False, num_workers=4)
        return trainloader, testloader


    '''model trainning step in every epoch'''
    def train(self, epoch, criterion, optimizer, logfile, loader, device, wm, uchida=None):
        print(f"\nEpoch: {epoch}")
        self.net.train()
        iteration = 0
        train_loss = 0
        correct = 0
        total = 0 
        print_frq = 50

        for batch_idx, (inputs, targets) in enumerate(loader):
            iteration += 1
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            
            wm_loss = 0
            if wm:
                wm_loss = uchida.embed(self.net, uchida.X, uchida.B, device)
            
            loss += 0.01 * wm_loss  
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum()
            
            if (iteration % print_frq == 0):
                print("wm loss: %.2f" % wm_loss)
                print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                        % (iteration, len(loader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        with open(logfile, 'a') as f:
            f.write(f"[train] Epoch: {epoch}\n")
            f.write("[train] Loss: %.3f | Acc: %.3f%% (%d / %d)\n" 
                        % (train_loss / iteration, 100. * correct / total, correct, total))


    '''acc test'''
    def test(self, criterion, logfile, loader, device):
        self.net.eval()
        iteration = 0
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            iteration += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum()

        with open(logfile, 'a') as f:
            f.write("Test results:\n")
            f.write("Loss: %.3f | Acc: %.3f%% (%d / %d)\n" 
                        % (test_loss / iteration, 100. * correct / total, correct, total))
        
        return 100. * correct / total


    '''checkpoint save'''
    def save(self, acc, epoch, device, save_dir, id):
        state = {
            'acc': acc,
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(state, os.path.join(save_dir, 'uchida_1024_id' + str(id) + '.pth'))