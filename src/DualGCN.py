import torch.nn as nn
import torch
# local imports
from layer.chebyshev import ChebyshevConv

class DualGCN(nn.Module):
    def __init__(self, size_in, size_out, hidden_dim, nb_layers, K, enable_bias=False, droprate=None):
        super(DualGCN, self).__init__()

        self.size_in = size_in
        self.size_out = size_out

        self.layers = nn.ModuleList()

        # additional linear layer to handle dimension discrepancies (if any)
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=size_in, out_features=hidden_dim, bias=enable_bias))
        
        # encoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim / 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim

        # decoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim * 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim
        
        # additional linear layer to handle dimension discrepancies (if any) 
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=size_out, bias=enable_bias))

        if droprate != None:
            self.dropout = nn.Dropout(p=droprate)
        else:
            self.dropout = None
        
        self.activation = nn.ReLU()

    def forward_one(self, x, L):
        for i in range(len(self.layers)):
            if type(self.layers[i]) == nn.Linear:
                x = self.layers[i](x)
            else:
                x = self.layers[i](x, L)
                x = self.activation(x)
                if self.dropout != None:
                    x = self.dropout(x)
        return x
        
    def forward(self, x1, x2, L1, L2):
        x1 = self.forward_one(x1, L1)
        x2 = self.forward_one(x2, L2)
        x = torch.cat((x1, x2), dim=1)
        x = nn.Linear(in_features=2*self.size_in, out_features=self.size_out, bias=None).forward(x)
        return x1, x2, x
