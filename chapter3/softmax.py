import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
from utils import Accumulator, accuracy, evaluate_accuracy # for runing delete the .

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True)

batch_size = 32
worker_num = 16
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=worker_num)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=worker_num)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)) # nn.Softmax(dim=1) don't need to be added here, because the loss function nn.CrossEntropyLoss() will do this for us. Equals to first pass nn.logSoftmax and use nn.NLLLoss() as loss.

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on', device)
net.to(device)

num_epochs = 10
# flag = 1
for epoch in range(num_epochs):
    print(f'epoch {epoch+1} / {num_epochs}')
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        # if epoch == 0 and flag:
        #     print(y_hat.sum(axis = 1, keepdim = True))
        #     # print(y_hat.shape)
        #     flag = 0
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        metric.add(float(l.sum()), accuracy(y_hat, y, device), y.numel())
    print(f'loss {metric[0]/metric[2]:.3f}, train acc {metric[1]/metric[2]:.3f}, eval acc {evaluate_accuracy(net, test_iter)}')