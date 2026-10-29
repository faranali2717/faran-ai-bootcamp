import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 0.1307 / 0.3081 = mnist mean and std
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('data', train=True, download=True, transform=tf)
test_set = datasets.MNIST('data', train=False, download=True, transform=tf)
train_set, val_set = torch.utils.data.random_split(full_train, [50000, 10000])

train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
val_dl = DataLoader(val_set, batch_size=256)
test_dl = DataLoader(test_set, batch_size=256)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def accuracy(model, dl):
    model.eval()
    right = 0
    n = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            right += (model(xb).argmax(1) == yb).sum().item()
            n += len(yb)
    return 100 * right / n


model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)

epochs = 20
losses = []

for ep in range(epochs):
    model.train()
    running = 0.0

    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)

        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item() * len(yb)

    sched.step()
    losses.append(running / len(train_set))
    print(f"epoch {ep + 1}/{epochs}  loss {losses[-1]:.4f}  val acc {accuracy(model, val_dl):.2f}%")

print(f"test acc: {accuracy(model, test_dl):.2f}%")

plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('cross-entropy')
plt.grid(True)
plt.tight_layout()
plt.show()

here = os.path.dirname(os.path.abspath(__file__))
torch.save(model.state_dict(), os.path.join(here, 'mlp_mnist.pth'))
print("saved weights in", here)