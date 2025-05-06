from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upscale = nn.Sequential(
            # initial feature extraction layer for global features
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            # second convolutional layer to extract base-level features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # 2 upsampling blocks for 4x upscaling
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            # final convolutional layer to bring the upscaled image to 1 output channel, producing final SR image
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.upscale(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # initial convolutional layer with LeakyReLU to keep small gradients
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            # sequential block of 3 convolutional layers to downsample image
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # feature maps are deepened by doubling output channels
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # global average pooling reduces each feature map to 1 value
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()        # outputs a final probability of the input being a real image(1) or not(0)
        )

    def forward(self, x):
        return self.model(x)
