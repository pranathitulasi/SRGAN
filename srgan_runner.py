import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import Generator

# Load the pretrained SRGAN generator
model = Generator()
model.load_state_dict(torch.load("4x_models7/generator_epoch_250.pth", map_location="cpu"))
model.eval()

# Normalization used during training (to [-1, 1])
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def denormalize(img_tensor):
    return (img_tensor + 1.0) / 2.0

def run_srgan(image_np):
    # image_np: 2D grayscale numpy array (H, W)
    pil_img = Image.fromarray(image_np.astype(np.uint8))
    input_tensor = to_tensor(pil_img).unsqueeze(0)  # Shape: (1, 1, H, W)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = denormalize(output_tensor.squeeze(0)).clamp(0, 1)  # Shape: (1, H, W)
    output_np = output_tensor.squeeze(0).numpy() * 255.0
    return output_np.astype(np.uint8)
