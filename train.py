import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data
from model import Graphormer


def main():
    train_loader, test_loader = get_data()
    model = Graphormer(64, 4, 2).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "Epoch: %2d, Step: %5d / %5d, Train loss: %.4f"
                    % (epoch + 1, i + 1, len(train_loader), running_loss / 2000)
                )
                running_loss = 0.0

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "Epoch: %2d, Test loss: %.4f" % (epoch + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

    print("Finished Training")


if __name__ == "__main__":
    main()
