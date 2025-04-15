import torch
import matplotlib.pyplot as plt
from transform_data import get_dataloader
from model import Generator

# Directories
low_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\pt5898p\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"
model_save_path = "models4/"

# Load DataLoader
test_loader = get_dataloader(low_res_dir, high_res_dir, batch_size=5)

generator = Generator()
generator.load_state_dict(torch.load("models4/generator_epoch_150.pth"))
generator.eval()


def denormalize(img):
    return (img + 1) / 2


# Get test samples
test_samples, test_high_res = next(iter(test_loader))
super_resolved = generator(test_samples).detach()

# Display output images
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
