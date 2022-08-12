import torch 
import torch.nn as nn
import torch.nn.functional as F
from bes.nn.es_module import ESModule


class Conv(ESModule):

    def __init__(self, input_dim=188, hidden_dim=128, output_dim=1, conv_channels=32, in_channels=13, kernel_size=3):
        super(Conv, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.conv_chanels = conv_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n1 = nn.LayerNorm([conv_channels,input_dim])
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n2 = nn.LayerNorm([conv_channels, input_dim])
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n3 = nn.LayerNorm([conv_channels, input_dim])
        self.conv4 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n4 = nn.LayerNorm([conv_channels, input_dim])
        self.conv5 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n5 = nn.LayerNorm([conv_channels, input_dim])

        self.fc1 = nn.Linear(in_features=conv_channels*input_dim, out_features=hidden_dim)
        self.nf1 = nn.LayerNorm([hidden_dim])
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf2 = nn.LayerNorm([hidden_dim])
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf3 = nn.LayerNorm([hidden_dim])
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf4 = nn.LayerNorm([hidden_dim])
        self.fc5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf5 = nn.LayerNorm([hidden_dim])
        
        self.fcout = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, h, return_featues=False):
        h = F.gelu(self.conv1(h))
        r = self.n1(h)
        h = F.gelu(self.conv2(r))
        h = self.n2(h)
        h = F.gelu(self.conv3(h))
        r = self.n3(h+r)
        h = F.gelu(self.conv4(r))
        h = self.n4(h)
        h = F.gelu(self.conv5(h))
        h = self.n5(h+r)
         
        #h = torch.mean(h, axis=1,)
        h = h.view(-1, self.conv_chanels*self.input_dim)
        h = F.selu(self.fc1(h))
        r = self.nf1(h)
        h = F.selu(self.fc2(r))
        h = self.nf2(h)
        h = F.selu(self.fc3(h))
        r = self.nf3(h+r)
        h = F.selu(self.fc4(r))
        h = self.nf4(h)
        h = F.selu(self.fc5(h))
        h = self.nf5(h+r)
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        
        return torch.sigmoid(self.fcout(h))


class ConvAgg(ESModule):

    def __init__(self, input_dim=178, hidden_dim=128, output_dim=1, conv_channels=32, 
                in_cont_channels=13, in_cat_dim=44, kernel_size=3):
        super(ConvAgg, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.conv_chanels = conv_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=in_cont_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n1 = nn.LayerNorm([conv_channels,input_dim])
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n2 = nn.LayerNorm([conv_channels, input_dim])
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n3 = nn.LayerNorm([conv_channels, input_dim])
        self.conv4 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n4 = nn.LayerNorm([conv_channels, input_dim])
        self.conv5 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n5 = nn.LayerNorm([conv_channels, input_dim])
        
        self.cat_fc1 = nn.Linear(in_features=in_cat_dim, out_features=hidden_dim)
        self.cat_nf1 = nn.LayerNorm([hidden_dim])
        self.cat_fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.cat_nf2 = nn.LayerNorm([hidden_dim])
        

        self.fc1 = nn.Linear(in_features=(conv_channels*input_dim + hidden_dim), out_features=hidden_dim)
        self.nf1 = nn.LayerNorm([hidden_dim])
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf2 = nn.LayerNorm([hidden_dim])
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf3 = nn.LayerNorm([hidden_dim])
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf4 = nn.LayerNorm([hidden_dim])
        self.fc5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf5 = nn.LayerNorm([hidden_dim])
        
        self.fcout = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, cont, cat, return_featues=False):

        h = F.gelu(self.conv1(cont))
        r = self.n1(h)
        h = F.gelu(self.conv2(r))
        h = self.n2(h)
        h = F.gelu(self.conv3(h))
        r = self.n3(h+r)
        h = F.gelu(self.conv4(r))
        h = self.n4(h)
        h = F.gelu(self.conv5(h))
        h = self.n5(h+r)

        # cat vars 
        h_cat = F.selu(self.cat_fc1(cat))
        h_cat = self.nf1(h_cat)
        h_cat = F.selu(self.fc2(h_cat))
        h_cat = self.nf2(h_cat)
         
        #h = torch.mean(h, axis=1,)
        h = h.view(-1, self.conv_chanels*self.input_dim)
        h = torch.cat((h, h_cat), dim=-1)
        h = F.selu(self.fc1(h))
        r = self.nf1(h)
        h = F.selu(self.fc2(r))
        h = self.nf2(h)
        h = F.selu(self.fc3(h))
        r = self.nf3(h+r)
        h = F.selu(self.fc4(r))
        h = self.nf4(h)
        h = F.selu(self.fc5(h))
        h = self.nf5(h+r)
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        
        return torch.sigmoid(self.fcout(h))


class ConvPred(ESModule):

    def __init__(self, input_dim=178, hidden_dim=128, output_dim=1, conv_channels=32, 
                in_cont_channels=13, in_cat_dim=44, kernel_size=3, pred_dim=18):
        super(ConvPred, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.conv_chanels = conv_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=in_cont_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n1 = nn.LayerNorm([conv_channels,input_dim])
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n2 = nn.LayerNorm([conv_channels, input_dim])
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n3 = nn.LayerNorm([conv_channels, input_dim])
        self.conv4 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n4 = nn.LayerNorm([conv_channels, input_dim])
        self.conv5 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=1)
        self.n5 = nn.LayerNorm([conv_channels, input_dim])
        
        self.cat_fc1 = nn.Linear(in_features=in_cat_dim, out_features=hidden_dim)
        self.cat_nf1 = nn.LayerNorm([hidden_dim])
        self.cat_fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.cat_nf2 = nn.LayerNorm([hidden_dim])
        

        self.fc1 = nn.Linear(in_features=(conv_channels*input_dim + hidden_dim), out_features=hidden_dim)
        self.nf1 = nn.LayerNorm([hidden_dim])
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf2 = nn.LayerNorm([hidden_dim])
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf3 = nn.LayerNorm([hidden_dim])
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf4 = nn.LayerNorm([hidden_dim])
        self.fc5 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.nf5 = nn.LayerNorm([hidden_dim])
        
        self.fcpred = nn.Linear(in_features=pred_dim, out_features=pred_dim)
        self.fcout = nn.Linear(in_features=hidden_dim+pred_dim, out_features=1)

    def forward(self, cont, cat, pred, return_featues=False):

        h = F.gelu(self.conv1(cont))
        r = self.n1(h)
        h = F.gelu(self.conv2(r))
        h = self.n2(h)
        h = F.gelu(self.conv3(h))
        r = self.n3(h+r)
        h = F.gelu(self.conv4(r))
        h = self.n4(h)
        h = F.gelu(self.conv5(h))
        h = self.n5(h+r)

        # cat vars 
        h_cat = F.selu(self.cat_fc1(cat))
        h_cat = self.nf1(h_cat)
        h_cat = F.selu(self.fc2(h_cat))
        h_cat = self.nf2(h_cat)
         
        #h = torch.mean(h, axis=1,)
        h = h.view(-1, self.conv_chanels*self.input_dim)
        h = torch.cat((h, h_cat), dim=-1)
        h = F.selu(self.fc1(h))
        r = self.nf1(h)
        h = F.selu(self.fc2(r))
        h = self.nf2(h)
        h = F.selu(self.fc3(h))
        r = self.nf3(h+r)
        h = F.selu(self.fc4(r))
        h = self.nf4(h)
        h = F.selu(self.fc5(h))
        h = self.nf5(h+r)

        pred = self.fcpred(pred)
        h = torch.cat((h, pred), dim=1)
        if return_featues:
            return torch.sigmoid(self.fcout(h)), h
        
        return torch.sigmoid(self.fcout(h))
