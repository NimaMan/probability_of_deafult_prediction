
class ConvOverTime(ESModule):
    # Convolution over the feature dim
    def __init__(self, input_dim=114, hidden_dim=128, output_dim=1, conv_channels=64):
        super(ConvOverTime, self).__init__()
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
