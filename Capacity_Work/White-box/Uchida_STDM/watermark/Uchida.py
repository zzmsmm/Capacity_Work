'''Uchida's watermark method
[1] Uchida, Yusuke, et al. "Embedding watermarks into deep neural networks"
    Proceedings of the 2017 ACM on international conference on multimedia retrieval.[ICMR-2017]
'''
import torch 
import torch.nn as nn  
import torch.nn.functional as F 

'''
method settings:
    target model: ResNet19_cifar10
    target layer: layer2.1.conv1
    weight size: [128, 128, 3, 3] -> [D, L, S, S]
        - B(id), size: T bits
        - X(extract matrix), size: S * S * D
'''
class Uchida(object):
    def __init__(self):
        super(Uchida, self).__init__()
        self.layer_weight = [
            "layer3.1.conv2.weight",
            "layer3.2.conv2.weight",
            "layer3.3.conv2.weight",
            "layer3.4.conv2.weight"
        ]
        self.n = len(self.layer_weight)
        self.B = []
        self.X = []


    '''keyGen() -> X, B'''
    def keyGen(self, T=256, M=3*3*256):
        for i in range(self.n):
            b = torch.randn(T)
            x = torch.randn(T * M).reshape(T, M)
            b = torch.heaviside(b, torch.ones(T))
            self.B.append(b)
            self.X.append(x)
        return self.B, self.X



    '''embed watermark'''
    def embed(self, net, X, B, device):
        r_loss = 0   # acting like a watermark regularizer

        for i in range(self.n):
            X[i], B[i] = X[i].to(device), B[i].to(device)
            if self.layer_weight[i] not in net.state_dict():
                print("Target embedding layer is not supported")
                return r_loss
            else:
                for name, param in net.named_parameters():
                    if name == self.layer_weight[i]:
                        W = param
                        
                w = torch.mean(W, dim=1).reshape(-1)
                Y = torch.sum(X[i] * w, dim=1)
                Y = torch.sigmoid(Y)
                Y = Y.to(device)

                r_loss += nn.BCELoss()(Y, B[i])
        
        return r_loss


    '''extract watermark'''
    def extract(self, net, X):
        ex_B = []
        for i in range(self.n):
            if self.layer_weight[i] not in net.state_dict():
                print("Target embedding layer is not supported")
            else:
                W = net.state_dict()[self.layer_weight[i]].cpu()
                w = torch.mean(W, dim=1).reshape(-1)
                b = torch.sum(X[i].cpu() * w, dim=1)
                ex_b = torch.heaviside(b, torch.ones(b.size(0)))
                ex_B.append(ex_b)
        
        return ex_B
    

    '''wm acc teat'''
    def verify(self, ex_B, logfile):
        total = 0
        for i in range(self.n):
            total += self.B[i].size(0)

        correct = 0
        for i in range(self.n):
            correct += ex_B[i].eq(self.B[i].cpu()).cpu().sum()
            
        acc = 100. * correct / total

        with open(logfile, 'a') as f:
            f.write("[WM] Verify results: %.3f%% (%d / %d)\n" % (acc, correct, total))
            
        return acc