import torch.nn as nn
import torch.nn.init as init

class ConvNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True)

        self.activation = nn.ReLU(inplace=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.map = nn.Linear(32, out_channel, bias=False)

        # self._init()

    def forward(self, x):
        n = x.size(0)
        x = self.norm1(self.activation(self.conv1(x)))
        x = self.norm2(self.activation(self.conv2(x)))

        x = self.pool(x).reshape(n, -1)

        out = self.map(x)
        return out

    def _init(self):
        def _init_weight(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight, 0, 0.1)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.constant_(m.weight, 0)

        self.apply(_init_weight)