"""
Batch Normalization (Ioffe & Szegedy, ICML 2015) from scratch.

  1. Implements BN for fully-connected layers: normalize with mini-batch
     statistics during training, keep running averages for inference, then
     scale and shift with learnable gamma and beta.
  2. Checks the forward pass against torch.nn.BatchNorm1d.
  3. Trains the same deep sigmoid MLP with and without BN on scikit-learn's
     digits dataset, with a learning rate high enough to show the difference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)


class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mu = x.mean(0)
            var = x.var(0, unbiased=False)                        # biased variance for normalization
            with torch.no_grad():                                 # unbiased variance for the running estimate
                n = x.size(0)
                self.running_mean.lerp_(mu, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mu, var = self.running_mean, self.running_var
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ----------------------------------------------------------------------
# 1) Check against PyTorch's implementation
# ----------------------------------------------------------------------
x = torch.randn(64, 10) * 3 + 5
mine, ref = MyBatchNorm1d(10), nn.BatchNorm1d(10)
for _ in range(5):
    mine(x); ref(x)
mine.eval(); ref.eval()
print("max |difference| train-mode stats vs nn.BatchNorm1d:",
      f"{(mine.running_var - ref.running_var).abs().max():.2e}")
print("max |difference| eval-mode output                 :",
      f"{(mine(x) - ref(x)).abs().max():.2e}")

# ----------------------------------------------------------------------
# 2) Deep sigmoid MLP with and without BN
# ----------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

def mlp(use_bn, depth=6, width=64):
    layers, d = [], 64
    for _ in range(depth):
        layers += [nn.Linear(d, width)] + ([MyBatchNorm1d(width)] if use_bn else []) + [nn.Sigmoid()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 10))

def run(use_bn, lr=0.2, epochs=30):
    torch.manual_seed(0)
    model = mlp(use_bn)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 32):
            i = perm[b:b + 32]
            loss = F.cross_entropy(model(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            history.append((model(Xte).argmax(1) == yte).float().mean().item())
    return history

plain, bn = run(False), run(True)
print("\nTest accuracy of a 6-layer sigmoid MLP (SGD, lr=0.2)")
print("epoch   without BN   with BN")
for ep in (0, 4, 9, 19, 29):
    print(f"{ep + 1:5d}   {plain[ep]:10.3f}   {bn[ep]:7.3f}")
