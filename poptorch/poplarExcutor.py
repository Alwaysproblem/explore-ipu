import torch
import poptorch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, input_size = 3, output_size = 1):
        super(Model, self).__init__()
        self.L = nn.Linear(input_size, output_size, bias = True)
        self.act = nn.Sigmoid()
        self.loss = nn.BCELoss()
    
    def forward(self, X, target = None):
        o = self.act(self.L(X))
        if self.training:
            return o, self.loss(o, target)
        
        return o

sample_n = 10
meana = np.array([1, 1])
cova = np.array([[0.1, 0],[0, 0.1]])

meanb = np.array([2, 2])
covb = np.array([[0.1, 0],[0, 0.1]])

x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

y_red = np.array([1] * sample_n)
y_green = np.array([0] * sample_n)

X = np.concatenate([x_red, x_green])
X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
y = np.concatenate([y_red, y_green])

y = y[:, None]

assert X.shape == (sample_n*2, 3)
assert y.shape == (sample_n*2, 1)

X = torch.Tensor(X)
y = torch.Tensor(y)

model = Model()
opt = Adam(model.parameters())

model.train()
for i in tqdm(range(1000)):
    opt.zero_grad()
    out, loss = model(X, y)
    loss.backward()
    opt.step()

model.eval()

print(*model.named_children())
print(model(X))