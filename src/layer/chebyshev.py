import math
import torch

class ChebyshevConv(torch.nn.Module):
    def __init__(self, K, size_in, size_out, bias=None):
        super(ChebyshevConv, self).__init__()

        self.K = K
        self.weight = torch.nn.Parameter(torch.FloatTensor(K, size_in, size_out))

        if bias != None:
            self.bias = torch.nn.Parameter(torch.FloatTensor(size_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias != None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, L):
        # x_0 = x,
        # x_1 = L * x,
        # x_k = 2 * L * x_{k-1} - x_{k-2},
        # where L = 2 * L / eigv_max - I.

        poly = []
        if self.K < 0:
            raise ValueError('ERROR: K must be non-negative!')
        elif self.K == 0:
            # x_0 = x
            poly.append(x)
        elif self.K == 1:
            # x_0 = x
            poly.append(x)
            if L.is_sparse:
                # x_1 = L * x
                poly.append(torch.sparse.mm(L, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = L * x
                poly.append(torch.mm(L, x))
        else:
            # x_0 = x
            poly.append(x)
            # x_1 = L * x
            poly.append(torch.mm(L, x))
            # x_k = 2 * L * x_{k-1} - x_{k-2}
            for k in range(2, self.K):
                poly.append(torch.mm(2 * L, poly[k - 1]) - poly[k - 2])
        
        feature = torch.stack(poly, dim=0)
        if feature.is_sparse:
            feature = feature.to_dense()
        graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)

        if self.bias != None:
            graph_conv = torch.add(input=graph_conv, other=self.bias, alpha=1)

        return graph_conv