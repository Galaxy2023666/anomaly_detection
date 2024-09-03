import torch
from torch import nn

from torch.utils.data import DataLoader

from tools.time2Vec.utility import *

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2VecOnRNN(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2VecOnRNN, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)

        self.fc1 = nn.Linear(hiddem_dim, 2)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x

class ToyDataset(Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()

        df = pd.read_csv('../../data/experiment_data/toy_dataset.csv')
        self.x = df["x"].values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model

    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset()
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        num_epochs = 100

        for ep in range(num_epochs):
            for x, y in dataloader:
                optimizer.zero_grad()

                y_pred = self.model(x.unsqueeze(1).float())
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()

                print("epoch: {}, loss:{}".format(ep, loss.item()))

    def preprocess(self, x):
        return x

    def decorate_output(self, x):
        return x

if __name__ == "__main__":
    x = list(range(1, 7000))
    y = list()

    for item in x:
        if item % 7 == 0:
            y.append(1)
        else:
            y.append(0)

    df = pd.DataFrame(list(zip(x, y)), columns=["x", "y"])
    df.to_csv('../../data/toy_dataset.csv', index=False)

    dataset = ToyDataset()
    print(dataset[6])

    pipe = ToyPipeline(Time2VecOnRNN("sin", 42))
    pipe.train()

    #pipe = ToyPipeline(Model("cos", 12))
    #pipe.train()
