import torch 
import torch.nn as nn
import torch.nn.functional as F
from bes.nn.es_module import ESModule


class MLP(ESModule):

    def __init__(self, input_dim, hidden_dim=64, output_dim=1, activation=F.selu, num_hidden_layers=1,
                 output_activation=F.sigmoid):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()

        if type(hidden_dim) == int:
            self.num_layers = num_hidden_layers + 1
            self.layers_dims = num_hidden_layers * [hidden_dim]
        elif type(hidden_dim) == list:
            self.num_layers = len(hidden_dim) + 1
            self.layers_dims = hidden_dim
        else:
            raise NotImplementedError

        self.layers_dims.insert(0, input_dim)
        self.layers_dims.append(output_dim)
        assert self.num_layers == len(self.layers_dims) - 1

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.layers_dims[i], self.layers_dims[i+1]))

    def forward(self, h):
        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)
        return torch.sigmoid(self.layers[-1](h))


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


class Conv2(ESModule):
    # Convolution over the feature dim
    def __init__(self, input_dim=114, hidden_dim=128, output_dim=1, conv_channels=64):
        super(Conv2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.conv_chanells = conv_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=3, padding=1)
        self.n1 = nn.LayerNorm([conv_channels,13])
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.n2 = nn.LayerNorm([conv_channels, 13])
        self.conv3 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.n3 = nn.LayerNorm([conv_channels, 13])
        self.conv4 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.n4 = nn.LayerNorm([conv_channels, 13])
        self.conv5 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.n5 = nn.LayerNorm([conv_channels, 13])

        self.fc1 = nn.Linear(in_features=conv_channels*13, out_features=hidden_dim)
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
        h = h.view(-1, self.conv_chanells*13)
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



class CatEndModel(ESModule):
    def __init__(self, input_dim):
        super(CatEndModel, self).__init__()
        self.conv = Conv(input_dim=114, hidden_dim=64, output_dim=1, conv_channels=13)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x