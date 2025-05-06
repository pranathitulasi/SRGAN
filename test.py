import torch
import matplotlib.pyplot as plt
from transform_data import get_dataloader
from model import Generator

low_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
high_res_dir = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output"

# loads a batch of 5 images from dataset
test_loader, val_loader = get_dataloader(low_res_dir, high_res_dir, batch_size=5)

# creates a new instance of the generator model and loads pre-trained weights
generator = Generator()
generator.load_state_dict(torch.load("4x_models7/generator_epoch_250.pth", map_location=torch.device('cpu')))
generator.eval()

# converts the [-1,1] normalised image to [0,1] for displaying
def denormalize(img):
    return (img + 1) / 2

# get test samples from the val set and passes them through the generator to produce SR images
test_samples, test_high_res = next(iter(val_loader))
super_resolved = generator(test_samples).detach()

# displays output images along with input and ground truth
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
