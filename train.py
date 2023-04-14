import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


if __name__ == "__main__":
    dataset = TUDataset(root='./data/TUDataset', name='MUTAG') 
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    torch.manual_seed(42) 
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(30):
        model.train()
        train_acc = 0.
        train_loss = 0.
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            pred = out.argmax(dim=1)
            train_acc += (pred == data.y).sum() / len(data.y)
            train_loss += loss.item()
        train_acc /= len(train_loader)
        train_loss /= len(train_loader)

        model.eval()
        test_acc = 0.
        test_loss = 0.
        with torch.no_grad():
            for data in test_loader:  # Iterate in batches over the training/test dataset.
                out = model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                loss = criterion(out, data.y)
                test_acc += (pred == data.y).sum() / len(data.y)  # Check against ground-truth labels.
                test_loss += loss.item()
            test_acc = test_acc / len(test_loader)  # Derive ratio of correct predictions.
            test_loss /= len(test_loader)
        print(f'Epoch: {epoch + 1:2d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
