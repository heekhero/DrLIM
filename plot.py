import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch.optim as optim
import torch

from tqdm import tqdm
from utils import ctloss
from model import ConvNet
from torchvision.datasets import MNIST
from mnist_clip import MNIST_clip
from matplotlib import pyplot as plt


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])
dataset = MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = dataloader.DataLoader(dataset, batch_size=128, shuffle=False)

model = ConvNet(1, 2)


model.load_state_dict(torch.load('./data/model_1281.pth'))
dataiter = iter(dataloader)

X = None
y = None
for t, data in tqdm(enumerate(dataiter)):
    imgs, labels = data
    output = model(imgs)

    if X is None:
        X = output
        y = labels
    else:
        X = torch.cat([X, output], dim=0)
        y = torch.cat([y, labels], dim=0)

X = X.detach().numpy()
y = y.detach().numpy()
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('result.png')