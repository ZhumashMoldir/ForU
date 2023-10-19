import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs

class SVMT(nn.Module):
    def __init__(self):
        super(SVMT, self).__init__()
        self.w = nn.Parameter(torch.randn(2, 1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self.w + self.b

def svmt_loss(output, y, w, c):
    hinge_loss = torch.mean(torch.clamp(1 - y * output, min=0))
    regularization_term = 0.5 * (w.t() @ w)
    return hinge_loss + c * regularization_term

def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.w.squeeze()

            loss = svmt_loss(output, y, weight, args.c)
            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))

def visualize(X, Y, model):
    W = model.w.squeeze().detach().cpu().numpy()
    b = model.b.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = np.vstack([x.ravel(), y.ravel()]).T

    z = (xy @ W + b).reshape(x.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, levels=[-1, 0, 1], colors=['Red', 'Blue', 'red'], alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral, s=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    X = (X - X.mean()) / X.std()
    Y[np.where(Y == 0)] = -1

    model = SVMT()
    model.to(args.device)

    train(X, Y, model, args)
    visualize(X, Y, model)