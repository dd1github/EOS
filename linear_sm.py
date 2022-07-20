
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


class Lin(nn.Module):

    def __init__(self):
        super(Lin, self).__init__()
        
        self.linear = nn.Linear(64, 10)
        
        

    def forward(self, x):
        
        out = self.linear(x)
        
        return out


