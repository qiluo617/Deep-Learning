# Qi Luo
# A02274095

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

batch_size = 32

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())

test_dataset = datasets.MNIST('./data',
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(100, 80)
        self.relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.softmax(self.fc3(x), dim=1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00000045)
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate():
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)

    print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(validation_loader.dataset), accuracy))
    return accuracy


def test():
    model.eval()
    val_loss, correct = 0, 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)

    print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    return accuracy


epochs = 50

for epoch in range(1, epochs + 1):
    train()
    print("Train Epoch: {}".format(epoch))
    acc = validate()

torch.save(model.state_dict(), 'model.ckpt')
