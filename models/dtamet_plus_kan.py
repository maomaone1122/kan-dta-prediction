
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from models.kan_layers import KANConv1D, KANLayer

class KFEB_KAN(nn.Module):
    """Kernel-Diverse Feature Extraction Block (using KANConv1D)"""
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            self.branches.append(
                KANConv1D(in_channels, out_channels, ks, padding=padding)
            )
        self.fusion = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        return self.relu(out + self.residual(x))


class CSIM(nn.Module):
    """Cross-Scale Interaction Module"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.fc = nn.Linear(channels, channels)
        self.kan = KANLayer(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, small_scale: torch.Tensor, large_scale: torch.Tensor) -> torch.Tensor:

        s = small_scale.permute(0, 2, 1)
        l = large_scale.permute(0, 2, 1)
        attn_out, _ = self.attn(s, l, l)
        attn_out = self.fc(attn_out)
        attn_out = self.kan(attn_out)
        out = self.norm(s + attn_out)
        return out.permute(0, 2, 1)


class DTIM(nn.Module):
    """Drug-Target Interaction Module"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.fc_drug = nn.Linear(channels, channels)
        self.fc_target = nn.Linear(channels, channels)
        self.kan = KANLayer(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = drug_feat.permute(0, 2, 1)
        t = target_feat.permute(0, 2, 1)

        d2t, attn_weights = self.cross_attn(d, t, t)
        d2t = self.fc_drug(d2t)
        d2t = self.kan(d2t)
        d_enhanced = self.norm(d + d2t)

        t2d, _ = self.cross_attn(t, d, d)
        t2d = self.fc_target(t2d)
        t2d = self.kan(t2d)
        t_enhanced = self.norm(t + t2d)

        return d_enhanced.permute(0, 2, 1), t_enhanced.permute(0, 2, 1), attn_weights


class MFM(nn.Module):
    """Multi-modal Fusion Module"""
    def __init__(self, channels: int, fusion_dim: int = 256):
        super().__init__()
        self.gate_drug = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        self.gate_target = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, fusion_dim),
            KANLayer(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, drug_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        d_pool = drug_feat.mean(dim=2)
        t_pool = target_feat.mean(dim=2)

        g_d = self.gate_drug(drug_feat)
        g_t = self.gate_target(target_feat)

        d_weighted = d_pool * g_d
        t_weighted = t_pool * g_t

        combined = torch.cat([d_weighted, t_weighted], dim=1)
        return self.fusion(combined).squeeze(-1)


class DTANetPlusKAN(nn.Module):
    """DTANet+ with KAN Integration"""
    def __init__(
        self,
        drug_max_len: int = 85,
        target_max_len: int = 1200,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        drug_kernels: List[int] = [3, 5, 7],
        target_kernels: List[int] = [5, 7, 9],
        drug_vocab_size: int = 75,
        target_vocab_size: int = 25
    ):
        super().__init__()


        self.drug_embed = nn.Embedding(drug_vocab_size, embed_dim, padding_idx=0)
        self.target_embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=0)


        self.drug_encoder = nn.Sequential(
            KFEB_KAN(embed_dim, hidden_dim, drug_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, drug_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, drug_kernels)
        )
        self.target_encoder = nn.Sequential(
            KFEB_KAN(embed_dim, hidden_dim, target_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, target_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, target_kernels)
        )


        self.downsample_drug = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.downsample_target = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)


        self.csim_drug = CSIM(hidden_dim)
        self.csim_target = CSIM(hidden_dim)


        self.dtim = DTIM(hidden_dim)


        self.decoder_drug = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=0),
            KFEB_KAN(hidden_dim, hidden_dim, drug_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, drug_kernels)
        )
        self.decoder_target = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=0),
            KFEB_KAN(hidden_dim, hidden_dim, target_kernels),
            KFEB_KAN(hidden_dim, hidden_dim, target_kernels)
        )


        self.mfm = MFM(hidden_dim, fusion_dim=256)

    def forward(self, drug_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:

        d_emb = self.drug_embed(drug_seq).permute(0, 2, 1)
        t_emb = self.target_embed(target_seq).permute(0, 2, 1)


        d_feat = self.drug_encoder(d_emb)
        t_feat = self.target_encoder(t_emb)


        d_small = self.downsample_drug(d_feat)
        t_small = self.downsample_target(t_feat)


        d_enhanced = self.csim_drug(d_small, d_feat)
        t_enhanced = self.csim_target(t_small, t_feat)


        d_interacted, t_interacted, _ = self.dtim(d_enhanced, t_enhanced)


        d_decoded = self.decoder_drug(d_interacted)
        t_decoded = self.decoder_target(t_interacted)


        if d_decoded.size(2) != d_feat.size(2):
            d_decoded = F.interpolate(d_decoded, size=d_feat.size(2), mode='linear', align_corners=False)
        if t_decoded.size(2) != t_feat.size(2):
            t_decoded = F.interpolate(t_decoded, size=t_feat.size(2), mode='linear', align_corners=False)


        d_final = d_decoded + d_feat
        t_final = t_decoded + t_feat


        pred = self.mfm(d_final, t_final)
        return pred
