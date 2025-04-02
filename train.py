import os
import torch
import torch.optim as optim
import torch.nn as nn
from transform_data import get_dataloader
from model import SRGenerator, SRDiscriminator

# Directories
low_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"
model_save_path = "models2/"
os.makedirs(model_save_path, exist_ok=True)


def load_model(model, model_save_path, model_name):
    model_path = None
    if os.path.exists(model_save_path):
        model_files = [f for f in os.listdir(model_save_path) if f.startswith(model_name)]
        if model_files:
            model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            model_path = os.path.join(model_save_path, model_files[-1])

    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded {model_name} weights from {model_path}")
    return model


# Hyperparameters
batch_size = 32
lr = 0.00001
num_epochs = 250

# Load DataLoader
train_loader = get_dataloader(low_res_dir, high_res_dir, batch_size)

generator = SRGenerator()
discriminator = SRDiscriminator()

# Loss functions
adversarial_loss = nn.BCELoss()
content_loss = nn.MSELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Training Loop
for epoch in range(num_epochs):
    for low_res_images, high_res_images in train_loader:
        batch_size_actual = low_res_images.size(0)
        real_labels = torch.ones((batch_size_actual, 1))
        fake_labels = torch.zeros((batch_size_actual, 1))

        # Train Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(high_res_images)
        fake_images = generator(low_res_images).detach()
        fake_outputs = discriminator(fake_images)
        loss_real = adversarial_loss(real_outputs, real_labels)
        loss_fake = adversarial_loss(fake_outputs, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(generator(low_res_images))
        loss_G_adversarial = adversarial_loss(fake_outputs, real_labels)
        loss_G_content = content_loss(generator(low_res_images), high_res_images)
        loss_G = loss_G_adversarial + 0.01 * loss_G_content
        loss_G.backward()
        optimizer_G.step()

print(f"Epoch {epoch + 1}: Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

# Save models2 every 50 epochs
if (epoch + 1) % 50 == 0:
    torch.save(generator.state_dict(), os.path.join(model_save_path, f"generator_epoch_{epoch + 1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_save_path, f"discriminator_epoch_{epoch + 1}.pth"))
    print(f"Model saved at epoch {epoch + 1}")
