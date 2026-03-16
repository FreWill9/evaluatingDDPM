from sklearn.datasets import load_digits
import torch
import torch.nn.functional as F
import os

if __name__ == "__main__":
    save_dir = "data"
    img_size = 16

    os.makedirs(save_dir, exist_ok=True)

    data = load_digits()
    # (1797, 8, 8) -> normalize to [-1, 1] directly
    all_tensors = torch.tensor(data.images, dtype=torch.float32) / 16.0  # [0, 1]
    all_tensors = all_tensors * 2.0 - 1.0  # [-1, 1]
    all_tensors = all_tensors.unsqueeze(1)  # [1797, 1, 8, 8]

    # Upsample to 16×16 using bilinear interpolation
    all_tensors = F.interpolate(all_tensors, size=(img_size, img_size), mode='bilinear', align_corners=False)

    print(all_tensors.shape)
    print(all_tensors.dtype)

    torch.save(all_tensors, f"{save_dir}/digits_gray{img_size}_{len(all_tensors)}.pt")
    print("Saved preprocessed digits dataset to disk")