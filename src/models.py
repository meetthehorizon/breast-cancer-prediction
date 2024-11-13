import torch
import torch.nn as nn
import torchvision.models as models

import torch

from src.logging_config import setup_logger

logger = setup_logger(__name__)


class FCNNBreastCancer(nn.Module):
    def __init__(self):
        super(FCNNBreastCancer, self).__init__()
        self.fc1 = nn.Linear(1 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 224 * 224)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class CNNBreastCancer(nn.Module):
    def __init__(self):
        logger.debug("Initializing BreastCancerCNN")

        super(CNNBreastCancer, self).__init__()
        nn.Conv2d()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        logger.info("BreastCancerCNN initialized")

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ResNetBreastCancer(nn.Module):
    def __init__(self):
        super(ResNetBreastCancer, self).__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    pass
