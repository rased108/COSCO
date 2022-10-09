import torch
import torch.nn as nn
from .constants import *
from .npn import *

from sys import argv

class energy_50(nn.Module):
    def __init__(self):
        super(energy_50, self).__init__()
        self.name = "energy_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 51, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        return x

class energy_latency_50(nn.Module):
    def __init__(self):
        super(energy_latency_50, self).__init__()
        self.name = "energy_latency_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latencyW_50(nn.Module):
    def __init__(self):
        super(energy_latencyW_50, self).__init__()
        self.name = "energy_latencyW_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Dropout(0.5),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
    

class energy_latency_10(nn.Module):
    def __init__(self):
        super(energy_latency_10, self).__init__()
        self.name = "energy_latency_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.Softplus(), 
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_10(nn.Module):
    def __init__(self):
        super(energy_latency2_10, self).__init__()
        self.name = "energy_latency2_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 14, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_50(nn.Module):
    def __init__(self):
        super(energy_latency2_50, self).__init__()
        self.name = "energy_latency2_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class stochastic_energy_latency_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_50, self).__init__()
        self.name = "stochastic_energy_latency_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 52, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_50, self).__init__()
        self.name = "stochastic_energy_latency2_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 54, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_10, self).__init__()
        self.name = "stochastic_energy_latency_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 12, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_10, self).__init__()
        self.name = "stochastic_energy_latency2_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 14, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s
    
# class energy_latency1w_10(nn.Module):
#     def __init__(self):
#         super(energy_latency2_50, self).__init__()
#         self.name = "energy_latency2_50"
#         self.find = nn.Sequential(
#             nn.Linear(50 * 54, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64), 
#             nn.LeakyReLU(),
#             nn.Linear(64, 2),
#             nn.Dropout(0.5),
#             nn.Sigmoid())

#     def forward(self, x):
#         x = x.flatten()
#         x = self.find(x)
#         if not('train' in argv[0] and 'train' in argv[2]):
#             x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
#         return x
    
class energy_latency2w_50(nn.Module):
    def __init__(self):
        super(energy_latency2w_50, self).__init__()
        self.name = "energy_latency2w_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
class energy_latency2w_10(nn.Module):
    def __init__(self):
        super(energy_latency2w_10, self).__init__()
        self.name = "energy_latency2w_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 14, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x 
    
class energy_latency2w1_50(nn.Module):
    def __init__(self):
        super(energy_latency2w1_50, self).__init__()
        self.name = "energy_latency2w1_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x    
    
    
class energy_latencyw_10(nn.Module):
    def __init__(self):
        super(energy_latencyw_10, self).__init__()
        self.name = "energy_latencyw_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 32), 
            nn.PReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x 
    
class energy_latency1_10(nn.Module):
    def __init__(self):
        super(energy_latency1_10, self).__init__()
        self.name = "energy_latency1_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x     
class energy_latency1_50(nn.Module):
    def __init__(self):
        super(energy_latency1_50, self).__init__()
        self.name = "energy_latency1_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
        
    
    
class energy_latency2w2_50(nn.Module):
    def __init__(self):
        super(energy_latency2w2_50, self).__init__()
        self.name = "energy_latency2w2_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
    
class energy_latency2w2_10(nn.Module):
    def __init__(self):
        super(energy_latency2w2_10, self).__init__()
        self.name = "energy_latency2w2_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 14, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x    
class energy_latencyW(nn.Module):
    def __init__(self):
        super(energy_latencyW, self).__init__()
        self.name = "energy_latencyW"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Dropout(0.5),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
class energy_latencyw_50(nn.Module):
    def __init__(self):
        super(energy_latencyw_50, self).__init__()
        self.name = "energy_latencyw_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 64), 
            nn.PReLU(),
            nn.Linear(64, 32), 
            nn.PReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x        