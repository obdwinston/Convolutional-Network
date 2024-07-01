import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 1e-3
epochs = 20

# +===========+
# | Load Data |
# +===========+

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

root = './data/pytorch'
train_data = datasets.MNIST(
    root=root,
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root=root,
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print(f'Train data size: {len(train_dataloader.dataset)}')
print(f'Test data size: {len(test_dataloader.dataset)}')

for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    break

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

# +==============+
# | Define Model |
# +==============+

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # every nn.Module subclass implements operations on input data in forward method
        x = F.tanh(self.conv1(x))   # layer 1: convolution + activation
        x = F.avg_pool2d(x, 2)      # layer 2: pooling
        x = F.tanh(self.conv2(x))   # layer 3: convolution + activation
        x = F.avg_pool2d(x, 2)      # layer 4: pooling
        x = x.view(-1, 16 * 5 * 5)  # flatten
        x = F.tanh(self.fc1(x))     # layer 5: fully-connected + activation
        x = F.tanh(self.fc2(x))     # layer 6: fully-connected + Activation
        x = self.fc3(x)             # layer 7: fully-connected (output)
        return x

# +====================+
# | Train & Test Model |
# +====================+

device = (
    'cuda' # parallel computing for NVIDIA GPUs
    if torch.cuda.is_available()
    else 'mps' # parallel computing for Apple Silicon
    if torch.backends.mps.is_available()
    else 'cpu' # general-purpose processor
)
print(f'Device: {device}')

model = LeNet5().to(device)
print(model)

loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_function, optimiser):
    size = len(dataloader.dataset)

    model.train() # set model to training mode

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward propagation
        pred = model(X) # performs forward pass and builds computation graph
        loss = loss_function(pred, y) # computes loss and adds loss to computation graph

        # backward propagation (auto-differentiation w.r.t. computation graph)
        loss.backward() # compute gradients and store gradient as attribute in respective parameter
        optimiser.step() # adjust weights using computed gradients
        optimiser.zero_grad() # reset gradients before next forward and backward pass

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)

    model.eval() # set model to evaluation mode

    test_loss, correct = 0, 0
    with torch.no_grad():
    # temporarily disables gradient calculation for all parameters (i.e. requires_grad=False)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= n_batches
    correct /= size
    print(f'Test Error: \n  Accuracy: {(100*correct):>0.1f}%, Average Loss: {test_loss:>8f} \n')

for t in range(epochs):
    print(f'Epoch {t+1}\n==============================')
    train(train_dataloader, model, loss_function, optimiser)
    test(test_dataloader, model, loss_function)
print('Done!')

# +============+
# | Save Model |
# +============+

torch.save(model.state_dict(), 'params.pth')
print('Saved PyTorch Model State to params.pth')

# +======================+
# | Load & Predict Model |
# +======================+

model = LeNet5().to(device) # re-instantiate model
model.load_state_dict(torch.load('params.pth')) # load parameters
print('Loaded PyTorch Model State from params.pth')

model.eval()

import random
index = random.randint(0, len(test_data) - 1)
x, y = test_data[index][0], test_data[index][1] # test_data[index] -> tuple (x, y)

with torch.no_grad():
    x = x.to(device)
    pred = model(x) # pred size (1, 10)

    predicted = pred.argmax(1).item()
    # return maximum value along dimension 1
    # extract item from tensor (i.e. integer scalar)
    actual = y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
