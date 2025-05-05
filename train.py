import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from transform_data import get_dataloader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import Generator, Discriminator

low_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"
model_save_path = "4x_models7/"
os.makedirs(model_save_path, exist_ok=True)

batch_size = 16
num_epochs = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator_path = None
discriminator_path = None

def load_model(model_save_path):
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
        print(f"Loaded generator weights from {generator_path}")

    if discriminator_path:
        print(f"Loaded discriminator weights from {discriminator_path}")

    return generator_path, discriminator_path


def train():
    train_loader = get_dataloader(low_res_dir, high_res_dir, batch_size)
    #print(f"Training loader size: {len(train_loader)}")

    generator = Generator()
    discriminator = Discriminator()

    # loading latest weights
    generator_path, discriminator_path = load_model(model_save_path)

    if generator_path:
        generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
    if discriminator_path:
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device('cpu')))

    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

    ssim_scores = []
    psnr_scores = []
    avg_ssim_values = []
    avg_psnr_values = []

    for epoch in range(num_epochs):
        total_ssim, total_psnr, num_batches = 0, 0, 0

        for n, (low_res_images, high_res_images) in enumerate(train_loader):
            batch_size_actual = low_res_images.size(0)

            # generate SR images
            generated_images = generator(low_res_images)

            real_labels = torch.ones(batch_size_actual)
            fake_labels = torch.zeros(batch_size_actual)

            # train discriminator
            optimizer_D.zero_grad()
            real_outputs = discriminator(high_res_images)
            fake_outputs = discriminator(generated_images.detach())

            loss_real = adversarial_loss(real_outputs, real_labels)
            loss_fake = adversarial_loss(fake_outputs, fake_labels)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()
            fake_outputs = discriminator(generated_images)
            loss_G_adversarial = adversarial_loss(fake_outputs, real_labels)
            loss_G_content = content_loss(generated_images, high_res_images)

            # combined loss of generator
            loss_G = loss_G_adversarial + 0.01 * loss_G_content
            loss_G.backward()
            optimizer_G.step()

            for i in range(batch_size_actual):
                real_img = high_res_images[i].squeeze(0).detach().numpy()
                fake_img = generated_images[i].squeeze(0).detach().numpy()

                total_ssim += ssim(real_img, fake_img, data_range=1.0)
                total_psnr += psnr(real_img, fake_img, data_range=1.0)
                num_batches += 1

        # store SSIM & PSNR scores
        avg_ssim = total_ssim / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim_values.append(avg_ssim)
        avg_psnr_values.append(avg_psnr)

        print(f"Epoch {epoch + 1}: Loss D: {loss_D:.4f}, Loss G: {loss_G :.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

        # saves models every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(model_save_path, f"generator_epoch_{epoch + 1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_save_path, f"discriminator_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

            test_samples, test_high_res = next(iter(train_loader))
            super_resolved = generator(test_samples).detach()

            def denormalize(img):
                return (img + 1) / 2

            fig, axes = plt.subplots(3, 5, figsize=(10, 6))

            for i in range(5):
                axes[0, i].imshow(denormalize(test_samples[i].squeeze(0)), cmap='gray')
                axes[0, i].set_title("Low-Res")
                axes[0, i].axis('off')

                axes[1, i].imshow(denormalize(test_high_res[i].squeeze(0)), cmap='gray')
                axes[1, i].set_title("High-Res")
                axes[1, i].axis('off')

                axes[2, i].imshow(denormalize(super_resolved[i].squeeze(0)), cmap='gray')
                axes[2, i].set_title("Generated")
                axes[2, i].axis('off')

            plt.show()

    plt.figure(figsize=(10, 5))

    # SSIM plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), avg_ssim_values, label="SSIM", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM Score")
    plt.title("SSIM Over Training")
    plt.legend()

    # PSNR plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), avg_psnr_values, label="PSNR", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR Score")
    plt.title("PSNR Over Training")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
