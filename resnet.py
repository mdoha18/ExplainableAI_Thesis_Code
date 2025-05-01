import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetAudio(nn.Module):
    """Modified ResNet18 for single-channel audio input."""
    def __init__(self, num_classes=35, pretrained=True):
        super(ResNetAudio, self).__init__()
        if pretrained:
            self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet18 = resnet18(weights=None)

        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)
