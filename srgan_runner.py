import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import Generator

# loads the pretrained generator weights
model = Generator()
model.load_state_dict(torch.load("4x_models7/generator_epoch_250.pth", map_location="cpu"))
model.eval()

# converts image to tensor (C, H, W) and normalises to match the train samples
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# denormalises tensor to convert into an image
def denormalize(img_tensor):
    return (img_tensor + 1.0) / 2.0

# initialises the function to run the trained model on the input image
def run_srgan(image_np):
    # converts the NumPy array into a PIL image with (0, 255) pixel values
    pil_img = Image.fromarray(image_np.astype(np.uint8))
    input_tensor = to_tensor(pil_img).unsqueeze(0)

    # disables gradient calculations to speed up computation
    with torch.no_grad():
        output_tensor = model(input_tensor)     # passes the input image to the model

    # denormalises the image and converts the tensor back into a NumPy array
    output_tensor = denormalize(output_tensor.squeeze(0)).clamp(0, 1)
    output_np = output_tensor.squeeze(0).numpy() * 255.0

    return output_np.astype(np.uint8)
