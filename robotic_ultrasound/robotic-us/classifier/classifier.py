import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.LeakyReLU(),
        )
        self.model = nn.Sequential(
            self.encoder,
            nn.Linear(256, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.softmax(x, dim=1)
        x = torch.argmax(x, dim=1)
        return x
