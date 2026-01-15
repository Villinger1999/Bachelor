import torch.nn as nn

class LeNet(nn.Module):
    def __init__(
        self,
        channel: int = 3,
        hidden: int = 768,
        num_classes: int = 10,
        activation: str = "sigmoid",
    ):
        super(LeNet, self).__init__()
    
        def act():
            return self._get_activation(activation)
            
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
        )

        self.fc = nn.Sequential(nn.Linear(hidden, num_classes))

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif name == "prelu":
            return nn.PReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softmax":
            return nn.Softmax(dim=1)
        elif name == "tanh":
            return nn.Tanh()
        elif name == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
