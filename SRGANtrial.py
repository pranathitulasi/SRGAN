import os
import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

# Dataset path
low_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"
model_save_path = "models5/"

os.makedirs(model_save_path, exist_ok=True)

transform_low_res = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_high_res = transforms.Compose([
    transforms.Resize((64, 64)),      # Downscale to low resolution
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])


class SuperResDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform_lr, transform_hr):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.low_res_files = sorted(os.listdir(low_res_dir))
        self.high_res_files = sorted(os.listdir(high_res_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return min(len(self.low_res_files), len(self.high_res_files))

    def __getitem__(self, index):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_files[index])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_files[index])

        low_res_image = Image.open(low_res_path).convert("L")
        high_res_image = Image.open(high_res_path).convert("L")

        low_res_image = self.transform_lr(low_res_image)
        high_res_image = self.transform_hr(high_res_image)

        return low_res_image, high_res_image


# Load Dataset
batch_size = 32
dataset = SuperResDataset(low_res_dir, high_res_dir, transform_low_res, transform_high_res)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

# Find the latest saved model
generator_path = None
discriminator_path = None

if os.path.exists(model_save_path):
    generator_files = [f for f in os.listdir(model_save_path) if f.startswith("generator_epoch_")]
    discriminator_files = [f for f in os.listdir(model_save_path) if f.startswith("discriminator_epoch_")]

    if generator_files:
        generator_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        generator_path = os.path.join(model_save_path, generator_files[-1])

    if discriminator_files:
        discriminator_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        discriminator_path = os.path.join(model_save_path, discriminator_files[-1])

if generator_path:
    generator.load_state_dict(torch.load(generator_path))
    print(f"Loaded generator weights from {generator_path}")

if discriminator_path:
    discriminator.load_state_dict(torch.load(discriminator_path))
    print(f"Loaded discriminator weights from {discriminator_path}")

# Loss Functions
adversarial_loss = nn.BCELoss()  # For Discriminator
content_loss = nn.MSELoss()      # For Generator

# Optimizers
lr = 0.00001
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

ssim_scores = []
psnr_scores = []

# Resume training from the loaded epoch, or start from 0
start_epoch = 0
if generator_path:
    start_epoch = int(generator_path.split("_")[-1].split(".")[0])

num_epochs = 250

for epoch in range(num_epochs):
    total_ssim, total_psnr, num_batches = 0, 0, 0

    for n, (low_res_images, high_res_images) in enumerate(train_loader):
        batch_size_actual = low_res_images.size(0)

        # Generate Super-Resolved Images
        high_res_fake = generator(low_res_images)

        # Create Labels
        real_labels = torch.ones((batch_size_actual, 1))
        fake_labels = torch.zeros((batch_size_actual, 1))

        # Train Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(high_res_images)
        fake_outputs = discriminator(high_res_fake.detach())  # Detach to avoid training G

        loss_real = adversarial_loss(real_outputs, real_labels)
        loss_fake = adversarial_loss(fake_outputs, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(high_res_fake)     # Discriminator evaluates fake images
        loss_G_adversarial = adversarial_loss(fake_outputs, real_labels)    # Want to fool D
        loss_G_content = content_loss(high_res_fake, high_res_images)       # Content loss

        loss_G = loss_G_adversarial + 0.01 * loss_G_content     # Combined loss
        loss_G.backward()
        optimizer_G.step()

        for i in range(batch_size_actual):
            real_img = high_res_images[i].squeeze(0).detach().numpy()
            fake_img = high_res_fake[i].squeeze(0).detach().numpy()

            total_ssim += ssim(real_img, fake_img, data_range=1.0)
            total_psnr += psnr(real_img, fake_img, data_range=1.0)
            num_batches += 1

            # Store SSIM & PSNR scores
        avg_ssim = total_ssim / num_batches
        avg_psnr = total_psnr / num_batches
        ssim_scores.append(avg_ssim)
        psnr_scores.append(avg_psnr)

    print(
        f"Epoch {epoch + 1}: Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

    # saves model every 50 epochs and prints comparison of low res vs high-res
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), os.path.join(model_save_path, f"generator_epoch_{epoch + 1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_save_path, f"discriminator_epoch_{epoch + 1}.pth"))
        print(f"Model saved at epoch {epoch + 1}")

        # Display output images
        test_samples, test_high_res = next(iter(train_loader))
        super_resolved = generator(test_samples).detach()

        def denormalize(img):
            return (img + 1) / 2

        fig, axes = plt.subplots(3, 5, figsize=(10, 6))

        for i in range(5):
            # Low-resolution (Input)
            axes[0, i].imshow(denormalize(test_samples[i].squeeze(0)), cmap='gray')
            axes[0, i].set_title("Low-Res")
            axes[0, i].axis('off')

            # High-resolution (Ground Truth)
            axes[1, i].imshow(denormalize(test_high_res[i].squeeze(0)), cmap='gray')
            axes[1, i].set_title("High-Res")
            axes[1, i].axis('off')

            # Super-resolved (Generated)
            axes[2, i].imshow(denormalize(super_resolved[i].squeeze(0)), cmap='gray')
            axes[2, i].set_title("Generated")
            axes[2, i].axis('off')
        plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), ssim_scores, label="SSIM", color="blue")
plt.plot(range(1, num_epochs + 1), psnr_scores, label="PSNR", color="red")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("SSIM & PSNR Over Training")
plt.legend()
plt.show()
