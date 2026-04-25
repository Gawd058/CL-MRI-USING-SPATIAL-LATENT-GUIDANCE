import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class ContrastiveFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 32, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            ResBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            ResBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            ResBlock(base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=1)


class DataConsistencyBlock(nn.Module):
    def __init__(self, lambda_init: float = 0.1):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x_dl, y=None, mask=None):
        if mask is None:
            return x_dl
        xc = torch.view_as_complex(x_dl.permute(0, 2, 3, 1).contiguous())
        yc = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())
        Xk = torch.fft.fftshift(torch.fft.fft2(xc), dim=(-2, -1))
        Yk = torch.fft.fftshift(torch.fft.fft2(yc), dim=(-2, -1))
        lam = self.lam.clamp(0.01, 10.0)
        mask_b = mask.squeeze(1)
        Xk_dc = torch.where(mask_b.bool(), (Xk + lam * Yk) / (1 + lam), Xk)
        xc_dc = torch.fft.ifft2(torch.fft.ifftshift(Xk_dc, dim=(-2, -1)))
        return torch.view_as_real(xc_dc).permute(0, 3, 1, 2).contiguous()


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 2, base_ch: int = 32):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, base_ch)
        self.enc2 = UNetBlock(base_ch, base_ch * 2)
        self.enc3 = UNetBlock(base_ch * 2, base_ch * 4)
        self.enc4 = UNetBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(base_ch * 8, base_ch * 16)
        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, 2)
        self.dec4 = UNetBlock(base_ch * 16, base_ch * 8)
        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = UNetBlock(base_ch * 8, base_ch * 4)
        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = UNetBlock(base_ch * 4, base_ch * 2)
        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = UNetBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    @staticmethod
    def _pad_to_match(upsampled, skip):
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        if diff_h != 0 or diff_w != 0:
            upsampled = F.pad(upsampled, [0, diff_w, 0, diff_h])
        return upsampled

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._pad_to_match(self.up4(b),  e4), e4], 1))
        d3 = self.dec3(torch.cat([self._pad_to_match(self.up3(d4), e3), e3], 1))
        d2 = self.dec2(torch.cat([self._pad_to_match(self.up2(d3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._pad_to_match(self.up1(d2), e1), e1], 1))
        return self.out_conv(d1)


class ConvBlock5(nn.Module):
    def __init__(self, in_ch=2, ch=64, out_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x) + x


class D5C5(nn.Module):
    def __init__(self, in_ch: int = 2, ch: int = 64, n_cascades: int = 5):
        super().__init__()
        self.cascades  = nn.ModuleList([ConvBlock5(in_ch, ch, in_ch) for _ in range(n_cascades)])
        self.dc_blocks = nn.ModuleList([DataConsistencyBlock() for _ in range(n_cascades)])

    def forward(self, x, y=None, mask=None):
        for conv, dc in zip(self.cascades, self.dc_blocks):
            x = conv(x)
            x = dc(x, y, mask)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.avg(x).view(b, c))
        return x * w.view(b, c, 1, 1)


class MICCANBlock(nn.Module):
    def __init__(self, ch: int = 64, in_ch: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.ca  = ChannelAttention(ch)
        self.out = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x):
        f = self.conv(x)
        f = self.ca(f)
        return self.out(f) + x


class MICCAN(nn.Module):
    def __init__(self, in_ch: int = 2, ch: int = 64, n_cascades: int = 5):
        super().__init__()
        self.cascades  = nn.ModuleList([MICCANBlock(ch, in_ch) for _ in range(n_cascades)])
        self.dc_blocks = nn.ModuleList([DataConsistencyBlock() for _ in range(n_cascades)])

    def forward(self, x, y=None, mask=None):
        for blk, dc in zip(self.cascades, self.dc_blocks):
            x = blk(x)
            x = dc(x, y, mask)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.ws   = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        pH = math.ceil(H / ws) * ws
        pW = math.ceil(W / ws) * ws
        xp = F.pad(x, (0, pW - W, 0, pH - H))
        _, _, H2, W2 = xp.shape
        xw = xp.reshape(B, C, H2 // ws, ws, W2 // ws, ws)
        xw = xw.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        xw2, _ = self.attn(xw, xw, xw)
        xw2 = self.norm(xw + xw2)
        xw2 = xw2.reshape(B, H2 // ws, W2 // ws, ws, ws, C)
        xw2 = xw2.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H2, W2)
        return xw2[:, :, :H, :W]


class TransformerBlock(nn.Module):
    def __init__(self, ch: int = 32, in_ch: int = 2, num_heads: int = 4):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, ch, 1)
        self.attn     = WindowAttention(ch, num_heads)
        self.ff       = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 2, ch, 3, padding=1),
        )
        self.proj_out = nn.Conv2d(ch, in_ch, 1)
        self.norm     = nn.GroupNorm(1, ch)

    def forward(self, x):
        h = self.proj_in(x)
        h = self.attn(h) + h
        h = self.ff(self.norm(h)) + h
        return self.proj_out(h) + x


class ReconFormer(nn.Module):
    def __init__(self, in_ch: int = 2, ch: int = 32, n_cascades: int = 5, num_heads: int = 4):
        super().__init__()
        self.transformer = TransformerBlock(ch, in_ch, num_heads)
        self.dc_blocks   = nn.ModuleList([DataConsistencyBlock() for _ in range(n_cascades)])
        self.n_cascades  = n_cascades

    def forward(self, x, y=None, mask=None):
        for i in range(self.n_cascades):
            x = self.transformer(x)
            x = self.dc_blocks[i](x, y, mask)
        return x


class CLReconModel(nn.Module):
    def __init__(self, latent_dim: int, backbone: nn.Module, in_ch: int = 2):
        super().__init__()
        self.z_proj  = nn.Linear(latent_dim, in_ch)
        self.backbone = backbone

    def forward(self, xu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xu.shape
        z_spatial = self.z_proj(z).view(B, C, 1, 1).expand(B, C, H, W)
        return self.backbone(torch.cat([xu, z_spatial], dim=1))


def build_reconstruction_model(
    name: str,
    latent_dim: int,
    target_h: int = None,
    target_w: int = None,
    **kwargs,
) -> CLReconModel:
    name = name.lower()
    if name == "unet":
        backbone = UNet(in_ch=4, out_ch=2, **kwargs)
    elif name == "d5c5":
        backbone = D5C5(in_ch=4, **kwargs)
    elif name == "miccan":
        backbone = MICCAN(in_ch=4, **kwargs)
    elif name == "reconformer":
        backbone = ReconFormer(in_ch=4, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
    return CLReconModel(latent_dim, backbone)