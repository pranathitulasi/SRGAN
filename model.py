from torch import nn


class SRGenerator(nn.Module):
    def __init__(self):
        super(SRGenerator, self).__init__()

        self.upscale = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),  # Initial feature extraction
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # Up-sampling Block (Pixel Shuffle)
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),     # performs 2x up-scaling
            nn.PReLU(),

            # nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            # nn.PixelShuffle(2),
            # nn.PReLU(),

            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),  # Final reconstruction
            nn.Tanh()
        )

    def forward(self, x):
        return self.upscale(x)


class SRDiscriminator(nn.Module):
    def __init__(self):
        super(SRDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
