import torch
import matplotlib.pyplot as plt
from transform_data import get_dataloader
from model import Generator

low_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"

# loads dataset
test_loader = get_dataloader(low_res_dir, high_res_dir, batch_size=5)

generator = Generator()
generator.load_state_dict(torch.load("4x_models7/generator_epoch_250.pth", map_location=torch.device('cpu')))
generator.eval()


def denormalize(img):
    return (img + 1) / 2


# get test samples
test_samples, test_high_res = next(iter(test_loader))
super_resolved = generator(test_samples).detach()

# displays output images
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
