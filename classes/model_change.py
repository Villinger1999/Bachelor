import torch.nn as nn
from torchvision import models
import sys

def get_model(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class LeNet(nn.Module):
    def __init__(
        self,
        channel: int = 3,
        hidden: int = 768,
        num_classes: int = 10,
        activation: str = "relu",
    ):
        super().__init__()

        self.act = self._get_activation(activation)

        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=2, stride=2),
            self.act,
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            self.act,
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            self.act,
        )

        self.fc = nn.Linear(hidden, num_classes)

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()

        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif name == "prelu":
            return nn.PReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "softmax":
            # NOTE: usually applied only to output layer
            return nn.Softmax(dim=1)
        elif name == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
