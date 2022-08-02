
import torch 
import torch.nn as nn
import torch.nn.functional as F
from bes.nn.es_module import ESModule
from pd.nn.att import DotProductAttention, AdditiveAttention


class MLP(ESModule):
    def __init__(self, input_dim, hidden_dim=64,):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        #self.nf1 = nn.LayerNorm([hidden_dim])
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        #self.nf2 = nn.LayerNorm([hidden_dim])
        
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        #self.nf3 = nn.LayerNorm([hidden_dim])
                
        self.fcout = nn.Linear(in_features=hidden_dim, out_features=1)
    
    def forward(self, h, return_featues=False):
        h = F.selu(self.fc1(h))
        #h = self.nf1(h)
        r = F.selu(self.fc2(h))
        #r = self.nf2(r)
        h = F.selu(self.fc3(h+r))
        #h = self.nf3(h)
        h = h+r
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        
        return torch.sigmoid(self.fcout(h))
    

class MLPAtt(ESModule):
    def __init__(self, input_dim, hidden_dim=64,):
        super(MLPAtt, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        #self.nf1 = nn.LayerNorm([hidden_dim])
        self.att = AdditiveAttention(hidden_dim=hidden_dim)        
        #self.nf2 = nn.LayerNorm([hidden_dim])
        self.fcout = nn.Linear(in_features=hidden_dim, out_features=1)
    
    def forward(self, h, return_featues=False):
        h = F.selu(self.fc1(h))
        #h = self.nf1(h)
        h, att = self.att(h, h)
        #h = self.nf2(h+r) 
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        return torch.sigmoid(self.fcout(h))