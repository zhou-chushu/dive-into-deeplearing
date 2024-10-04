import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_num, out_num):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_num, out_num))
        self.bias = nn.Parameter(torch.randn(out_num))
    def __repr__(self):
        return f"MyLinear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias=True)"
    
    def forward(self, X):
        return torch.mm(X, self.weight) + self.bias

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fnl1 = MyLinear(7, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.fnl2 = nn.Linear(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fnl3 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fnl1(x))
        x = self.dropout1(x)
        x = F.relu(self.fnl2(x))
        x = self.dropout2(x)
        x = self.fnl3(x)
        return x

print(MLP())
print(MLP().parameters)
print(MLP().state_dict())
print(MLP().named_parameters())
print(MLP()._modules)

net = MLP()
print(*[(name, param.shape) for name, param in net.named_parameters()])

# class MySequential(nn.Module):
#     def __init__(self, *args):
#         super(MySequential, self).__init__()
#         for idx, module in enumerate(args):
#             # self._modules['my' + str(idx)] = module
#             self.add_module('my' + str(idx), module)
    
#     def forward(self, x):
#         for module in self._modules.values():
#             x = module(x)
#         return x

# net = MySequential(nn.Linear(7, 2), nn.ReLU(), nn.Linear(2, 1))
# print(net._modules)
# print(net.state_dict())