
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def concordance_index(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """计算Concordance Index (CI)"""
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    n = len(y_true)
    concordant = 0
    total = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    
    return concordant / total if total > 0 else 0.5

def rm2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """计算r_m^2指标"""
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    y_pred_mean = np.mean(y_pred)
    ss_res0 = np.sum((y_true - y_pred_mean) ** 2)
    r0_2 = 1 - (ss_res0 / ss_tot)
    
    return r2 * (1 - np.sqrt(abs(r2 - r0_2)))

def evaluate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """计算所有评估指标"""
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    ci = concordance_index(y_pred, y_true)
    rm2_score = rm2(y_pred, y_true)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'CI': ci,
        'r_m^2': rm2_score
    }

def compute_auc_aupr(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 10.0) -> dict:
    """计算AUC和AUPR (用于分类任务)"""
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    y_binary = (y_true_np > threshold).astype(np.float32)
    
    if len(np.unique(y_binary)) < 2:
        return {'AUC': 0.5, 'AUPR': 0.5}
    
    auc = roc_auc_score(y_binary, y_pred_np)
    aupr = average_precision_score(y_binary, y_pred_np)
    
    return {'AUC': auc, 'AUPR': aupr}
