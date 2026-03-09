from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebAGray32(Dataset):
    """Loads CelebA jpg images, converting to img_size × img_size grayscale tensors."""

    def __init__(self, root, img_size):
        self.paths = sorted(Path(root).glob("*.jpg"))
        self.img_size = img_size
        if not self.paths:
            raise FileNotFoundError(f"No jpg images found in {root}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),      # downsize
            transforms.Grayscale(num_output_channels=1),  # single-channel grey
            transforms.ToTensor(),                        # -> [1, img_size, img_size] float in [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5]),  # normalization to [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))