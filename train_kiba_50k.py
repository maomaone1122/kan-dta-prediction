
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data.preprocessing import DTAPreprocessor, DTADataset
from models.kan_layers import KANLayer
from utils.metrics import evaluate_metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os
import json
import pandas as pd


DRUG_MAX_LEN = 100
TARGET_MAX_LEN = 300
N_SAMPLES = 50000
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-4
EMBED_DIM = 96
HIDDEN_DIM = 96

print("=" * 70)
print("DTANet+ with KAN - KIBA (50K Samples)")
print(f"Samples: {N_SAMPLES}")
print(f"Drug len: {DRUG_MAX_LEN}, Target len: {TARGET_MAX_LEN}")
print(f"Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
print("=" * 70)


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


class DualLoss(nn.Module):
    def __init__(self, mse_w=1.0, rank_w=0.5, margin=0.5):
        super().__init__()
        self.mse_w = mse_w
        self.rank_w = rank_w
        self.margin = margin

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)
        mask = (target_diff.abs() > 0.05).float()
        sign_loss = F.relu(self.margin - torch.sign(target_diff) * pred_diff) * mask
        rank_loss = sign_loss.sum() / (mask.sum() + 1e-8)
        return self.mse_w * mse_loss + self.rank_w * rank_loss


preprocessor = DTAPreprocessor(DRUG_MAX_LEN, TARGET_MAX_LEN)
drugs, targets, affinities = preprocessor.load_kiba_data("data/kiba.csv")
drugs_enc, targets_enc, aff_arr = preprocessor.prepare_data(drugs, targets, affinities)

np.random.seed(42)
n = min(N_SAMPLES, len(drugs_enc))
idx = np.random.choice(len(drugs_enc), n, replace=False)
drugs_enc, targets_enc, aff_arr = drugs_enc[idx], targets_enc[idx], aff_arr[idx]

print(f"Loaded {len(drugs_enc)} samples")

all_indices = np.arange(len(drugs_enc))
train_idx, val_idx = train_test_split(all_indices, test_size=0.15, random_state=42)

train_loader = torch.utils.data.DataLoader(
    DTADataset(drugs_enc[train_idx], targets_enc[train_idx], aff_arr[train_idx]),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    DTADataset(drugs_enc[val_idx], targets_enc[val_idx], aff_arr[val_idx]),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = KANDTANet(
    drug_max_len=DRUG_MAX_LEN,
    target_max_len=TARGET_MAX_LEN,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    drug_kernels=[3, 5],
    target_kernels=[5, 7],
    drug_vocab_size=preprocessor.drug_vocab_size,
    target_vocab_size=preprocessor.target_vocab_size
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")


optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = DualLoss(mse_w=1.0, rank_w=0.5)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

history = {"train_loss": [], "val_loss": [], "ci": [], "mse": [], "mae": []}
best_ci = 0
best_state = None
no_improve = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for drug_seq, target_seq, aff in pbar:
        drug_seq = drug_seq.to(device)
        target_seq = target_seq.to(device)
        aff = aff.to(device)
        optimizer.zero_grad()
        pred = model(drug_seq, target_seq)
        loss = criterion(pred, aff)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_targets = [], []
    val_loss_total = 0
    with torch.no_grad():
        for drug_seq, target_seq, aff in val_loader:
            drug_seq = drug_seq.to(device)
            target_seq = target_seq.to(device)
            aff = aff.to(device)
            pred = model(drug_seq, target_seq)
            val_loss_total += F.mse_loss(pred, aff).item()
            all_preds.append(pred.cpu())
            all_targets.append(aff.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = evaluate_metrics(all_preds, all_targets)
    avg_val_loss = val_loss_total / len(val_loader)
    scheduler.step()

    history["train_loss"].append(round(avg_train_loss, 4))
    history["val_loss"].append(round(avg_val_loss, 4))
    history["ci"].append(round(metrics['CI'], 4))
    history["mse"].append(round(metrics['MSE'], 4))
    history["mae"].append(round(metrics['MAE'], 4))

    if metrics['CI'] > best_ci:
        best_ci = metrics['CI']
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    elapsed = (time.time() - start_time) / 60
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
          f"CI: {metrics['CI']:.4f} | MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f} | {elapsed:.1f} min")

    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history, "best_ci": best_ci
    }, "checkpoints/kiba_50k_checkpoint.pt")

    if best_state is not None:
        torch.save(best_state, "checkpoints/best_kiba_50k_model.pt")

    if device.type == "mps":
        torch.mps.empty_cache()

    if no_improve >= 8:
        print(f"\nEarly stopping. Best CI = {best_ci:.4f}")
        break

print("\n" + "=" * 70)
print(f"Training completed. Best CI: {best_ci:.4f}")
print(f"Total time: {(time.time() - start_time) / 60:.1f} min")
print("=" * 70)

pd.DataFrame({
    "Epoch": list(range(1, len(history["ci"]) + 1)),
    "Train_Loss": history["train_loss"],
    "Val_Loss": history["val_loss"],
    "CI": history["ci"],
    "MSE": history["mse"],
    "MAE": history["mae"]
}).to_csv("results/kiba_50k_training_curve.csv", index=False)

with open("results/kiba_50k_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print("Saved:")
print("  results/kiba_50k_training_curve.csv")
print("  checkpoints/best_kiba_50k_model.pt")
