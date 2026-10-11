
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.preprocessing import DTAPreprocessor, DTADataset
from models.kan_layers import KANLayer
from tqdm import tqdm
import time


DRUG_MAX_LEN = 100
TARGET_MAX_LEN = 300
BATCH_SIZE = 16

print("=" * 70)
print("真实内存压力测试 - 跑100个batch的前向+反向")
print("=" * 70)


preprocessor = DTAPreprocessor(DRUG_MAX_LEN, TARGET_MAX_LEN)
drugs, targets, affinities = preprocessor.load_kiba_data("data/kiba.csv")
drugs_enc, targets_enc, aff_arr = preprocessor.prepare_data(drugs, targets, affinities)

print(f"全量样本数: {len(drugs_enc)}")


class StableKANConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_ch)
        self.kan = KANLayer(out_ch, out_ch, grid_size=4, spline_order=2)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.final_norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.kan(out)
        out = out.permute(0, 2, 1)
        out = out + self.residual(x)
        out = self.final_norm(out)
        return torch.tanh(out)

class MultiKernelKANBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes):
        super().__init__()
        self.branches = nn.ModuleList([
            StableKANConv(in_ch, out_ch, ks) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv1d(out_ch * len(kernel_sizes), out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        return torch.tanh(out + self.residual(x))

class KANDTANet(nn.Module):
    def __init__(self, drug_max_len=100, target_max_len=300, embed_dim=96,
                 hidden_dim=96, drug_kernels=[3, 5], target_kernels=[5, 7],
                 drug_vocab_size=75, target_vocab_size=25):
        super().__init__()
        self.drug_embed = nn.Embedding(drug_vocab_size, embed_dim, padding_idx=0)
        self.target_embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=0)

        self.drug_encoder = nn.Sequential(
            MultiKernelKANBlock(embed_dim, hidden_dim, drug_kernels),
            MultiKernelKANBlock(hidden_dim, hidden_dim, drug_kernels),
        )
        self.target_encoder = nn.Sequential(
            MultiKernelKANBlock(embed_dim, hidden_dim, target_kernels),
            MultiKernelKANBlock(hidden_dim, hidden_dim, target_kernels),
        )

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, drug_seq, target_seq):
        d_emb = self.drug_embed(drug_seq).permute(0, 2, 1)
        t_emb = self.target_embed(target_seq).permute(0, 2, 1)

        d_feat = self.drug_encoder(d_emb)
        t_feat = self.target_encoder(t_emb)

        d_pool = d_feat.mean(dim=2)
        t_pool = t_feat.mean(dim=2)

        d_seq = d_feat.permute(0, 2, 1)
        t_seq = t_feat.permute(0, 2, 1)
        attn_out, _ = self.cross_attn(d_seq, t_seq, t_seq)
        d_attended = self.norm(d_seq + attn_out).mean(dim=1)

        combined = torch.cat([d_pool + d_attended, t_pool], dim=1)
        return self.fusion(combined).squeeze(-1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = KANDTANet().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()


full_dataset = DTADataset(drugs_enc, targets_enc, aff_arr)
full_loader = torch.utils.data.DataLoader(
    full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

print(f"\n开始跑 100 个 batch 的前向+反向...")
print(f"（如果内存不够，会在前几个 batch 就崩溃）\n")

start = time.time()
max_memory = 0

for i, (drug_seq, target_seq, aff) in enumerate(tqdm(full_loader, desc="Training test")):
    drug_seq = drug_seq.to(device)
    target_seq = target_seq.to(device)
    aff = aff.to(device)

    optimizer.zero_grad()
    pred = model(drug_seq, target_seq)
    loss = criterion(pred, aff)
    loss.backward()
    optimizer.step()


    if torch.backends.mps.is_available():
        mem = torch.mps.current_allocated_memory() / 1e9
        if mem > max_memory:
            max_memory = mem

    if i >= 99:
        break

elapsed = time.time() - start
print(f"\n100个batch耗时: {elapsed:.1f}秒")
print(f"平均每个batch: {elapsed/100:.3f}秒")
print(f"完整1轮预计时间: {elapsed / 100 * len(full_loader) / 60:.1f}分钟")
print(f"20轮预计时间: {elapsed / 100 * len(full_loader) * 20 / 3600:.1f}小时")
print(f"\n峰值 MPS 内存: {max_memory:.2f} GB")
print("=" * 70)


if max_memory < 6:
    print("✅ 内存健康，可以跑全量")
elif max_memory < 10:
    print("⚠️ 内存偏高，可以跑但注意监控")
else:
    print("❌ 内存危险，建议用 5 万样本")

if elapsed / 100 * len(full_loader) / 60 < 20:
    print("✅ 速度可接受")
else:
    print("⚠️ 速度较慢，跑 20 轮可能超过 5 小时")
