import torch.nn as nn

class PriceNet(nn.Module):
    def __init__(self, n_features, hidden1=128):
        super().__init__()
        # первый скрытый блок
        self.fc1   = nn.Linear(n_features, hidden1)
        self.bn1   = nn.BatchNorm1d(hidden1)
        self.drop1 = nn.Dropout(0.2)
        # второй скрытый блок
        hidden2     = max(hidden1 // 2, 1)
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.bn2   = nn.BatchNorm1d(hidden2)
        self.drop2 = nn.Dropout(0.2)
        # третий скрытый блок
        self.fc3   = nn.Linear(hidden2, 32)
        self.bn3   = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.2)
        # выход
        self.out   = nn.Linear(32, 1)
        self.act   = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        return self.out(x)