import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dropout_p: float, C: int, num_groups: int):
        super().__init__()
        self.groupNorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.groupNorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p, inplace=True)

    def forward(self, x, embeddings):
        # x = x + t, x.shape = (B, C, H, W)
        x = x + embeddings[:, :x.shape[1], :, :]

        # r = Conv2(Relu(GN2(dropOut(Conv1(Relu(GN1(x)))))))
        r1 = self.conv1(self.relu(self.groupNorm1(x)))
        r2 = self.dropout(r1)
        r3 = self.conv2(self.relu(self.groupNorm2(r2)))

        # Add residual
        return r3 + x


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class UNetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class SinusoidalEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
