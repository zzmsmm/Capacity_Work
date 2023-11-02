'''
STDM: Spread-Transform Dither Modulation Watermarking of Deep Neural Network
'''
import torch 
import torch.nn as nn  
import torch.nn.functional as F 

'''
uchida + new loss
'''
class STDM(object):
    def __init__(self):
        super(STDM, self).__init__()
        self.layer_weight = [
            "layer2.0.conv2.weight",
            "layer2.1.conv2.weight",
            "layer2.2.conv2.weight",
            "layer2.3.conv2.weight",

            "layer3.0.conv2.weight",
            "layer3.1.conv2.weight",
            "layer3.2.conv2.weight",
            "layer3.3.conv2.weight",
            "layer3.4.conv2.weight",
            "layer3.5.conv2.weight",

        ]
        self.n = len(self.layer_weight)
        self.B = []
        self.X = []


    '''keyGen() -> X, B'''
    def keyGen(self, T1=256, T2=512, M1=3*3*128, M2=3*3*256):
        for i in range(self.n):
            if (i < 4):
                b = torch.randn(T1)
                x = torch.randn(T1 * M1).reshape(T1, M1)
                b = torch.heaviside(b, torch.ones(T1))
                self.B.append(b)
                self.X.append(x)
            else:
                b = torch.randn(T2)
                x = torch.randn(T2 * M2).reshape(T2, M2)
                b = torch.heaviside(b, torch.ones(T2))
                self.B.append(b)
                self.X.append(x)
        return self.B, self.X



    '''embed watermark'''
    def embed(self, net, X, B, device):
        r_loss = 0   # acting like a watermark regularizer

        for i in range(self.n):
            X[i], B[i] = X[i].to(device), B[i].to(device)
            for name, param in net.named_parameters():
                if name == self.layer_weight[i]:
                    W = param
                    break
                    
            w = torch.mean(W, dim=1).reshape(-1)
            Y = torch.sum(X[i] * w, dim=1)
            Y = self.stdm(Y)
            '''Y = 10 * torch.sin(10 * Y)
            Y = 1 - torch.sigmoid(-Y)'''
            Y = Y.to(device)
            r_loss += nn.BCELoss()(Y, B[i])
        
        return r_loss
    
    def stdm(self, x, alpha=10, beta=10):
        y = alpha * torch.sin(beta * x)
        z = 1 - torch.sigmoid(-y)
        return z


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
                b = self.stdm(b) - 0.5
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