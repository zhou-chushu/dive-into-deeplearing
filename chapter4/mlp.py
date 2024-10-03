import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from chapter3 import Accumulator
from chapter3 import accuracy, evaluate_accuracy
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils import data

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=False)

batch_size = 32
worker_num = 16
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=worker_num)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=worker_num)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on', device)
net.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    print(f'epoch {epoch+1} / {num_epochs}')
    metric = Accumulator(3)
    net.train()
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    print(f'loss {metric[0]/metric[2]:.3f}, train acc {metric[1]/metric[2]:.3f}, eval acc {evaluate_accuracy(net, test_iter, device)}')