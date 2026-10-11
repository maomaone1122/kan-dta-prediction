
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False


def plot_all(csv_path, suffix, title_prefix):
    """通用画图函数
    csv_path: 训练曲线CSV路径
    suffix: 输出文件名后缀，如 '_20k'
    title_prefix: 图标题前缀，如 '2万样本'
    """
    df = pd.read_csv(csv_path)


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Epoch"], df["Train_Loss"], marker='o', label='Train Loss', linewidth=2, markersize=5)
    ax.plot(df["Epoch"], df["Val_Loss"], marker='s', label='Val Loss', linewidth=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{title_prefix} - Training and Validation Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"results/loss_curve{suffix}.png", dpi=200)
    plt.close()
    print(f"已生成: results/loss_curve{suffix}.png")


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Epoch"], df["CI"], marker='o', color='green', linewidth=2, markersize=6, label='CI')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    ax.axhline(y=0.764, color='red', linestyle='--', alpha=0.7, label='Paper Baseline (0.764)')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Concordance Index (CI)", fontsize=12)
    ax.set_title(f"{title_prefix} - Concordance Index", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"results/ci_curve{suffix}.png", dpi=200)
    plt.close()
    print(f"已生成: results/ci_curve{suffix}.png")


    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color1, fontsize=12)
    ax1.plot(df["Epoch"], df["Train_Loss"], marker='o', color=color1, linewidth=2, label='Train Loss')
    ax1.plot(df["Epoch"], df["Val_Loss"], marker='s', color=color1, linestyle='--', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('CI', color=color2, fontsize=12)
    ax2.plot(df["Epoch"], df["CI"], marker='^', color=color2, linewidth=2.5, markersize=7, label='CI')
    ax2.axhline(y=0.764, color='red', linestyle=':', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    plt.title(f"{title_prefix} - Loss and CI vs Epoch", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"results/combined_curve{suffix}.png", dpi=200)
    plt.close()
    print(f"已生成: results/combined_curve{suffix}.png")


if __name__ == "__main__":
    print("=" * 50)
    print("生成 2万样本版本图...")
    print("=" * 50)
    plot_all("results/kiba_training_curve.csv", "_20k", "KIBA 20K Samples")

    print()
    print("=" * 50)
    print("生成 5万样本版本图...")
    print("=" * 50)
    plot_all("results/kiba_50k_training_curve.csv", "_50k", "KIBA 50K Samples")

    print()
    print("=" * 50)
    print("生成 5万 baseline 版本图...")
    print("=" * 50)
    plot_all("results/kiba_50k_baseline_training_curve.csv", "_baseline", "5K Baseline (No KAN)")

    print()
    print("=" * 50)
    print("生成 full baseline 版本图...")
    print("=" * 50)
    plot_all("results/kiba_full_baseline_training_curve.csv", "_full_baseline", "Full Baseline (No KAN)")

    print()
    print("=" * 50)
    print("生成 full KAN 版本图（新增）...")
    print("=" * 50)
    plot_all("results/kiba_full_training_curve.csv", "_full", "KIBA Full Dataset (KAN)")

    print()
    print("=" * 50)
    print("全部完成！")
    print("=" * 50)
