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

    def __init__(self, hidden_dim=64, output_dim=1, ):
        super(Conv, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=25, kernel_size=5, )
        self.conv2 = nn.Conv1d(in_channels=25, out_channels=5, kernel_size=5, )
        self.conv3 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3, )
        self.fc1 = nn.Linear(in_features=63, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)
    def forward(self, h):
        h = F.selu(self.conv1(h))
        h = F.selu(self.conv2(h))
        h = F.selu(self.conv3(h))
        h = F.selu(self.fc1(h))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


class CatEndModel(nn.Module):
    def __init__(self, input_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x