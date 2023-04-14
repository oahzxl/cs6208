import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import Graphormer


if __name__ == "__main__":
    dataset = TUDataset(root="./data/TUDataset", name="MUTAG", use_edge_attr=True)
    model = Graphormer(dim=64, head_num=4, layer_num=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = dataset.shuffle()
    train_dataset = dataset[: int(len(dataset) * 0.7)]
    test_dataset = dataset[int(len(dataset) * 0.7) :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for epoch in range(30):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(
                data.x.cuda(),
                data.edge_index.cuda(),
                data.edge_attr.cuda(),
                data.batch.cuda(),
            )
            loss = criterion(out, data.y.cuda())
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            train_acc += (pred == data.y.cuda()).sum() / len(data.y)
            train_loss += loss.item()
        train_acc /= len(train_loader)
        train_loss /= len(train_loader)

        model.eval()
        test_acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                out = model(
                    data.x.cuda(),
                    data.edge_index.cuda(),
                    data.edge_attr.cuda(),
                    data.batch.cuda(),
                )
                pred = out.argmax(dim=1)
                loss = criterion(out, data.y.cuda())
                test_acc += (pred == data.y.cuda()).sum() / len(data.y)
                test_loss += loss.item()
            test_acc = test_acc / len(test_loader)
            test_loss /= len(test_loader)
        print(
            f"Epoch: {epoch + 1:2d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )
