import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch.optim as optim
import torch

from utils import ctloss
from model import ConvNet
from torchvision.datasets import MNIST
from mnist_clip import MNIST_clip

transform = transforms.Compose([
    transforms.Resize(36),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])
dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = dataloader.DataLoader(dataset, batch_size=64, shuffle=True)

model = ConvNet(1, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

i = 0
while True:
    torch.save(model.state_dict(), './data/model_{}.pth'.format(i))
    dataiter = iter(dataloader)
    for t, data in enumerate(dataiter):
        imgs, labels = data
        output = model(imgs)

        optimizer.zero_grad()
        loss = ctloss(output, labels)
        loss.backward()
        optimizer.step()

        if t % 1000 == 0:
            print('Epoch : {}     Iter : {}      {:.3f}'.format(i, t, loss.data.item()))

    i += 1