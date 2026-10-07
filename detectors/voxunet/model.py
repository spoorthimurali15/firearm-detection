"""3D U-Net that finds where a device is inside an occupancy volume.

Input: an occupancy volume of shape (B, 1, R, R, R). The last axis is the
build direction (z), the order in which a printer lays down layers.

Output: two things. The first is one score per mesh, shape (B,), computed as
the mean of the k highest voxel scores on or next to the material. The second
is the full-resolution map of voxel scores, shape (B, 1, R, R, R), which can
be shown as a heatmap.

The mesh score is built from the strongest voxels, so to call a mesh malign
the network has to point to some region of it. This pushes it to learn where
the device is, not just whether one is present. During training, the detector
can also give it exact per-voxel labels (see detector.py).

With bottleneck="mamba", the middle of the network, at 1/8 resolution, uses a
bidirectional Mamba block that reads the voxels in print order, layer by
layer. This needs an NVIDIA GPU and "pip install mamba-ssm causal-conv1d". If
Mamba is not installed, the detector switches to bottleneck="conv"
automatically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def amp_enabled(device):
    """bf16 autocast only where the GPU supports it (A100/L4 yes, T4 no -> fp32)."""
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class BiMambaZ(nn.Module):
    """Residual bidirectional Mamba over voxels flattened in print order (z, then x, y)."""

    def __init__(self, dim, depth=2):
        super().__init__()
        from mamba_ssm import Mamba          # imported lazily: optional dependency
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.fwd = nn.ModuleList([Mamba(d_model=dim) for _ in range(depth)])
        self.bwd = nn.ModuleList([Mamba(d_model=dim) for _ in range(depth)])

    def forward(self, x):                              # x: (B, C, X, Y, Z)
        B, C, X, Y, Z = x.shape
        s = x.permute(0, 4, 2, 3, 1).reshape(B, Z * X * Y, C)   # z-major sequence
        for n, f, b in zip(self.norms, self.fwd, self.bwd):
            h = n(s)
            s = s + f(h) + b(h.flip(1)).flip(1)
        return s.reshape(B, Z, X, Y, C).permute(0, 4, 2, 3, 1).contiguous()


class VoxUNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128), k=64, bottleneck="conv"):
        super().__init__()
        self.k = k
        self.enc = nn.ModuleList([Block(1, ch[0])] +
                                 [Block(ch[i - 1], ch[i]) for i in range(1, len(ch))])
        self.ssm = BiMambaZ(ch[-1]) if bottleneck == "mamba" else nn.Identity()
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i], ch[i - 1], 2, stride=2)
                                 for i in range(len(ch) - 1, 0, -1)])
        self.dec = nn.ModuleList([Block(2 * ch[i - 1], ch[i - 1])
                                  for i in range(len(ch) - 1, 0, -1)])
        self.head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, occ):
        x, skips = occ, []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else Fn.max_pool3d(x, 2))
            skips.append(x)
        x = self.ssm(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        vox = self.head(x).float()                               # (B,1,R,R,R)
        # only let voxels in/next to material vote (evidence must be ON the object)
        near = Fn.max_pool3d(occ.float(), 3, stride=1, padding=1) > 0
        masked = vox.masked_fill(~near, -1e4).flatten(1)
        k = min(self.k, masked.shape[1])
        return masked.topk(k, dim=1).values.mean(1), vox
