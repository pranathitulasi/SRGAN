import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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


transform_low_res = transforms.Compose([
    transforms.Resize((110, 110)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_high_res = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_dataloader(low_res_dir, high_res_dir, batch_size=16):
    dataset = SuperResDataset(low_res_dir, high_res_dir, transform_low_res, transform_high_res)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
