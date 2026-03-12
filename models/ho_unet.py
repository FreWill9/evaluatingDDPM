"""
PyTorch port of the DDPM U-Net from Ho et al. (2020)
"Denoising Diffusion Probabilistic Models"
Adapted for 32×32 single-channel (grayscale) images.

Original TF code: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py

Porting notes (TF v1 → PyTorch):
  - The original imports a custom `nn` helper module (diffusion_tf/nn.py) that
    wraps TF v1 ops with variable-scope management and weight initializers.
    PyTorch's nn.Module handles this natively, so no equivalent wrapper is needed:
      • nn.conv2d(x, num_units, filter_size)  → nn.Conv2d(in_ch, out_ch, kernel)
      • nn.nin(x, num_units)  (1×1 conv)      → nn.Conv2d(channels, channels, 1)
      • nn.dense(x, num_units)                 → nn.Linear(in_features, out_features)
      • nn.get_timestep_embedding(t, dim)      → standalone get_timestep_embedding()
  - init_scale=0. (zero-init on residual output convs) is replicated via
    nn.init.zeros_() on conv2 in ResnetBlock, proj_out in AttnBlock, and conv_out.
  - tf_contrib.layers.group_norm → nn.GroupNorm(32, channels)
  - tf.nn.swish → F.silu() (identical: x · σ(x))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embeddings (same as Ho et al. / Vaswani et al.).
    timesteps: (B,) int tensor
    Returns:   (B, embedding_dim) float tensor
    """
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nonlinearity(x):
    """Swish / SiLU activation — used throughout the original."""
    return F.silu(x)


def normalize(x, num_groups=32):
    """Group Normalization (Wu & He, 2018). The original uses GN, not BN."""
    return F.group_norm(x, num_groups=min(num_groups, x.shape[1]))


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class Upsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    """Pre-activation ResNet block with timestep embedding injection."""

    def __init__(self, in_ch, out_ch, temb_ch, dropout):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.temb_proj = nn.Linear(temb_ch, out_ch)

        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Shortcut (1×1 conv) when channel counts differ
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_conv = nn.Identity()

        # Zero-init last conv (like init_scale=0. in the original)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip_conv(x) + h


class AttnBlock(nn.Module):
    """Single-head self-attention (QKV via 1×1 conv), as in Ho et al."""

    def __init__(self, channels):
        super().__init__()
        self.C = channels
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

        # Zero-init output projection
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, H * W)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W)

        # Attention weights: (B, HW, HW)
        w = torch.bmm(q.permute(0, 2, 1), k) * (C ** -0.5)
        w = F.softmax(w, dim=-1)

        # Attend to values
        h = torch.bmm(v, w.permute(0, 2, 1)).reshape(B, C, H, W)
        h = self.proj_out(h)

        return x + h


# ──────────────────────────────────────────────
# Full U-Net
# ──────────────────────────────────────────────

class UNet(nn.Module):
    """
    DDPM U-Net (Ho et al. 2020), ported to PyTorch.

    Default config for 32×32 grayscale:
        ch=128, ch_mult=(1,2,2,2), num_res_blocks=2,
        attn_resolutions=(16,), dropout=0.1
    These match the CIFAR-10 config from the paper, adjusted for 1 channel.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
        resamp_with_conv: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        temb_ch = ch * 4

        # ---- Timestep embedding MLP ----
        self.temb_dense0 = nn.Linear(ch, temb_ch)
        self.temb_dense1 = nn.Linear(temb_ch, temb_ch)

        # ---- Downsampling ----
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch_block = ch
        current_res = 32  # input spatial resolution
        self.skip_channels = [ch]  # track channels for skip connections

        for i_level in range(self.num_resolutions):
            out_ch_block = ch * ch_mult[i_level]
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(ResnetBlock(in_ch_block, out_ch_block, temb_ch, dropout))
                if current_res in attn_resolutions:
                    level_attns.append(AttnBlock(out_ch_block))
                else:
                    level_attns.append(nn.Identity())
                in_ch_block = out_ch_block
                self.skip_channels.append(in_ch_block)

            self.down_blocks.append(nn.ModuleDict({
                "blocks": level_blocks,
                "attns": level_attns,
            }))

            if i_level != self.num_resolutions - 1:
                self.down_samples.append(Downsample(in_ch_block, resamp_with_conv))
                current_res //= 2
                self.skip_channels.append(in_ch_block)
            else:
                self.down_samples.append(None)

        # ---- Middle ----
        self.mid_block1 = ResnetBlock(in_ch_block, in_ch_block, temb_ch, dropout)
        self.mid_attn = AttnBlock(in_ch_block)
        self.mid_block2 = ResnetBlock(in_ch_block, in_ch_block, temb_ch, dropout)

        # ---- Upsampling ----
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            out_ch_block = ch * ch_mult[i_level]
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()

            for i_block in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                level_blocks.append(ResnetBlock(in_ch_block + skip_ch, out_ch_block, temb_ch, dropout))
                if current_res in attn_resolutions:
                    level_attns.append(AttnBlock(out_ch_block))
                else:
                    level_attns.append(nn.Identity())
                in_ch_block = out_ch_block

            self.up_blocks.append(nn.ModuleDict({
                "blocks": level_blocks,
                "attns": level_attns,
            }))

            if i_level != 0:
                self.up_samples.append(Upsample(in_ch_block, resamp_with_conv))
                current_res *= 2
            else:
                self.up_samples.append(None)

        # ---- Output ----
        self.norm_out = nn.GroupNorm(min(32, in_ch_block), in_ch_block)
        self.conv_out = nn.Conv2d(in_ch_block, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, timesteps):
        """
        x:         (B, 1, 32, 32) noised image
        timesteps: (B,) integer timestep indices
        """
        assert x.ndim == 4

        # ---- Timestep embedding ----
        temb = get_timestep_embedding(timesteps, self.ch)
        temb = nonlinearity(self.temb_dense0(temb))
        temb = self.temb_dense1(temb)

        # ---- Downsampling path ----
        h = self.conv_in(x)
        hs = [h]

        for i_level in range(self.num_resolutions):
            blocks = self.down_blocks[i_level]["blocks"]
            attns = self.down_blocks[i_level]["attns"]

            for block, attn in zip(blocks, attns):
                h = block(h, temb)
                h = attn(h)
                hs.append(h)

            if self.down_samples[i_level] is not None:
                h = self.down_samples[i_level](h)
                hs.append(h)

        # ---- Middle ----
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        # ---- Upsampling path ----
        for i_level in range(self.num_resolutions):
            blocks = self.up_blocks[i_level]["blocks"]
            attns = self.up_blocks[i_level]["attns"]

            for block, attn in zip(blocks, attns):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
                h = attn(h)

            if self.up_samples[i_level] is not None:
                h = self.up_samples[i_level](h)

        assert not hs

        # ---- Output ----
        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        return h