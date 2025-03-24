import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Dataset path
dataset_dir = r"C:/Users/pt5898p/OneDrive - University of Greenwich/Documents/Year3/COMP1682_FYP/FYPtrial/SRGAN/Dataset_modified"

transform = transforms.Compose([
    transforms.Resize((64, 64)),      # Downscale to low resolution
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
full_train_set = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)

# Select 100 random samples for quick testing
indices = torch.randperm(len(full_train_set))[:1000]  # Randomly pick 100 samples
train_set = Subset(full_train_set, indices)

# Define DataLoader
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Display some low-resolution images
real_samples, _ = next(iter(train_loader))
print(real_samples.shape)

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = real_samples[i].squeeze(0)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
plt.show()


class SRGenerator(nn.Module):
    def __init__(self):
        super(SRGenerator, self).__init__()

        self.upscale = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),  # Initial feature extraction
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # Upsampling Block (Pixel Shuffle)
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),

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

# Instantiate Models
generator = SRGenerator()
discriminator = SRDiscriminator()

# Loss Functions
adversarial_loss = nn.BCELoss()  # For Discriminator
content_loss = nn.MSELoss()      # For Generator

# Optimizers
lr = 0.0002
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

num_epochs = 50

for epoch in range(num_epochs):
    for n, (low_res_images, _) in enumerate(train_loader):
        batch_size_actual = low_res_images.size(0)

        # Generate Super-Resolved Images
        high_res_fake = generator(low_res_images)

        # Create Labels
        real_labels = torch.ones((batch_size_actual, 1))
        fake_labels = torch.zeros((batch_size_actual, 1))

        # Train Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(low_res_images)
        fake_outputs = discriminator(high_res_fake.detach())  # Detach to avoid training G

        loss_real = adversarial_loss(real_outputs, real_labels)
        loss_fake = adversarial_loss(fake_outputs, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(high_res_fake)  # Discriminator evaluates fake images
        loss_G_adversarial = adversarial_loss(fake_outputs, real_labels)  # Want to fool D
        # loss_G_content = content_loss(high_res_fake, low_res_images)  # Content loss
        low_res_images_resized = torch.nn.functional.interpolate(low_res_images, size=(256, 256), mode="bilinear", align_corners=False)
        loss_G_content = content_loss(high_res_fake, low_res_images_resized)
        loss_G = loss_G_adversarial + 0.01 * loss_G_content  # Combined loss
        loss_G.backward()
        optimizer_G.step()

        # Print loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch + 1} | Loss D: {loss_D.item()} | Loss G: {loss_G.item()}")

test_samples, _ = next(iter(train_loader))
super_resolved = generator(test_samples).detach()

# Display Original and Super-Resolved Images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    # Low-resolution (Input)
    axes[0, i].imshow(test_samples[i].squeeze(0), cmap='gray')
    axes[0, i].set_title("Low-Res")
    axes[0, i].axis('off')

    # Super-resolved (Output)
    axes[1, i].imshow(super_resolved[i].squeeze(0), cmap='gray')
    axes[1, i].set_title("Super-Res")
    axes[1, i].axis('off')

plt.show()
