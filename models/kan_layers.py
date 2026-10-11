
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    """
    KAN层：支持任意维度输入，将最后一维作为特征维度进行变换
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order


        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.scale_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.scale_bias = nn.Parameter(torch.randn(out_features) * 0.1)


        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [*, in_features]  任意维度，最后一维是特征
        输出: [*, out_features]
        """
        original_shape = x.shape
        features = original_shape[-1]
        x_flat = x.reshape(-1, features)
        out_flat = self._forward_2d(x_flat)
        new_shape = original_shape[:-1] + (self.out_features,)
        return out_flat.reshape(new_shape)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        spline_basis = self._compute_spline_basis(x, self.grid)

        spline_out = torch.einsum('bik, oik -> bo', spline_basis, self.spline_weight)

        scale_out = torch.einsum('bi, oi -> bo', x, self.scale_weight) + self.scale_bias
        return spline_out + scale_out

    def _compute_spline_basis(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        G = self.grid_size
        K = self.spline_order

        grid_ext = torch.cat([
            grid[0:1] - torch.arange(K, 0, -1, device=grid.device).float(),
            grid,
            grid[-1:] + torch.arange(1, K + 1, device=grid.device).float()
        ])

        basis = torch.zeros(B, D, G + K, device=x.device)
        for i in range(G + K):
            left = grid_ext[i]
            right = grid_ext[i + 1]
            basis[:, :, i] = ((x >= left) & (x < right)).float()

        for k in range(1, K + 1):
            new_basis = torch.zeros(B, D, G + K - k, device=x.device)
            for i in range(G + K - k):
                left = grid_ext[i]
                right = grid_ext[i + k + 1]
                denom = right - left
                denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                coef1 = (x - left) / denom
                coef2 = (right - x) / denom
                new_basis[:, :, i] = coef1 * basis[:, :, i] + coef2 * basis[:, :, i + 1]
            basis = new_basis

        return basis

class KANConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.kan = KANLayer(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.kan(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        return F.relu(x)
