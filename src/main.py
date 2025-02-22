# main.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_model():
    """
    簡単なモデルのトレーニングを行う関数
    """
    # データセットの準備
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    dataset = SimpleDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # モデルの定義
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # トレーニングループ
    for epoch in range(5):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def main():
    """
    メイン関数
    """
    print("LLMを活用したプロジェクト1の実行")
    train_model()

if __name__ == "__main__":
    main()