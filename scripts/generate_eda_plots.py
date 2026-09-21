from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDS_ROOT = PROJECT_ROOT / "data" / "UNIFIED" / "builds"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 200,
    }
)


def plot_cleaning_and_noise() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Cleaning Funnel / Filtering Breakdown
    languages = ["Chagatai (chg)", "South Uzbek (uzs)", "Uyghur (uig)"]
    raw_counts = [4369, 35865, 277624]
    dropped_noise = [33, 472, 39280]  # noise + duplicates + empty
    cleaned_counts = [4336, 35393, 238344]
    balanced_counts = [3035, 3035, 3035]

    y_pos = np.arange(len(languages))
    bar_height = 0.22

    ax1.barh(
        y_pos + bar_height * 1.5,
        [np.log10(v) for v in raw_counts],
        height=bar_height,
        label="Raw Sentences / Segments",
        color="#64748b",
        alpha=0.85,
    )
    ax1.barh(
        y_pos + bar_height * 0.5,
        [np.log10(v) for v in dropped_noise],
        height=bar_height,
        label="Dropped (Noise & Duplicates)",
        color="#ef4444",
        alpha=0.85,
    )
    ax1.barh(
        y_pos - bar_height * 0.5,
        [np.log10(v) for v in cleaned_counts],
        height=bar_height,
        label="Cleaned Sources (Full)",
        color="#3b82f6",
        alpha=0.85,
    )
    ax1.barh(
        y_pos - bar_height * 1.5,
        [np.log10(v) for v in balanced_counts],
        height=bar_height,
        label="Balanced Train Budget",
        color="#10b981",
        alpha=0.85,
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(languages, fontweight="medium")
    ax1.set_xlabel("Количество предложений (log10 scale)")
    ax1.set_title("Воронка фильтрации и балансировки по языкам")
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    ax1.set_xticklabels(["10", "100", "1K", "10K", "100K", "1M"])
    ax1.legend(loc="lower right", framealpha=0.95)

    # Annotate exact numbers
    for i in range(len(languages)):
        ax1.text(
            np.log10(raw_counts[i]) + 0.05,
            y_pos[i] + bar_height * 1.5,
            f"{raw_counts[i]:,}",
            va="center",
            fontsize=8.5,
            color="#475569",
        )
        ax1.text(
            np.log10(cleaned_counts[i]) + 0.05,
            y_pos[i] - bar_height * 0.5,
            f"{cleaned_counts[i]:,}",
            va="center",
            fontsize=8.5,
            color="#1e40af",
        )

    # 2. Noise Reasons Breakdown
    reasons = [
        "Latin Script\n(URLs, headers, tags)",
        "Low Arabic Ratio\n(< 90% letters)",
        "No Letters\n(numbers / symbols)",
        "URL / Bibliography\n(http, doi, .com)",
        "CJK Ideographs\n(Chinese text)",
        "Cyrillic Script\n(Russian notes in UZS)",
    ]
    noise_counts = [14754, 10695, 5064, 1312, 481, 120]
    colors = ["#f43f5e", "#f97316", "#eab308", "#06b6d4", "#8b5cf6", "#6366f1"]

    bars = ax2.barh(reasons[::-1], noise_counts[::-1], color=colors[::-1], alpha=0.85)
    ax2.set_xlabel("Количество отфильтрованных строк")
    ax2.set_title("Причины отсева некачественных строк (Uyghur + UZS)")

    for bar, count in zip(bars, noise_counts[::-1]):
        ax2.text(
            bar.get_width() + 250,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({count / sum(noise_counts) * 100:.1f}%)",
            va="center",
            fontsize=9,
            fontweight="medium",
        )
    ax2.set_xlim(0, 18000)

    plt.tight_layout()
    out_path = ASSETS_DIR / "eda_cleaning_noise.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_token_distributions() -> Path:
    sources = pd.read_csv(
        BUILDS_ROOT / "chagatai_uzs_uyghur_balanced" / "source_sentences.csv"
    )
    train = pd.read_csv(BUILDS_ROOT / "chagatai_uzs_uyghur_balanced" / "train.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Source Sentence Lengths KDE / Histogram
    lang_palette = {"chg": "#10b981", "uig": "#3b82f6", "uzs": "#8b5cf6"}
    lang_names = {
        "chg": "Chagatai (med=13, mean=15.5)",
        "uig": "Uyghur (med=11, mean=13.9)",
        "uzs": "South Uzbek (med=10, mean=11.7)",
    }

    for lang in ["chg", "uig", "uzs"]:
        sub = sources[sources["language"] == lang]
        sns.kdeplot(
            data=sub["num_tokens"],
            ax=ax1,
            label=lang_names[lang],
            color=lang_palette[lang],
            linewidth=2.2,
            clip=(1, 70),
        )

    ax1.set_xlim(0, 60)
    ax1.set_xlabel("Число слов в предложении (Tokens per sentence)")
    ax1.set_ylabel("Плотность распределения (Density)")
    ax1.set_title("Распределение длин исходных предложений")
    ax1.axvline(13, color="#10b981", linestyle="--", alpha=0.5, label="Chagatai median (13)")
    ax1.legend(loc="upper right", framealpha=0.95)

    # 2. Augmented Train Sequences Boxplot
    method_palette = {
        "partial": "#f59e0b",
        "sequential": "#0284c7",
        "random": "#9333ea",
    }
    method_labels = {
        "partial": "Partial Merge\n(хвост + голова, 1 EOS)\nmed: 11-14 токенов",
        "sequential": "Sequential\n(2-4 соседних предл.)\nmed: 36-42 токена",
        "random": "Random Reordered\n(2-5 перемешанных)\nmed: 48-55 токенов",
    }

    train_plot = train.copy()
    train_plot["method_desc"] = train_plot["method"].map(method_labels)

    order_methods = ["partial", "sequential", "random"]
    order_labels = [method_labels[m] for m in order_methods]
    palette = [method_palette[m] for m in order_methods]

    sns.boxplot(
        data=train_plot,
        x="method_desc",
        y="num_tokens",
        order=order_labels,
        ax=ax2,
        palette=palette,
        hue="method_desc",
        legend=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        fliersize=2,
    )
    ax2.set_ylim(0, 160)
    ax2.set_xlabel("Метод аугментации в Train")
    ax2.set_ylabel("Число токенов в последовательности")
    ax2.set_title("Длины последовательностей по методам аугментации")

    # Annotate stats
    for idx, m in enumerate(order_methods):
        sub = train[train["method"] == m]["num_tokens"]
        ax2.text(
            idx,
            sub.median() + 5,
            f"med={sub.median():.0f}\nmean={sub.mean():.1f}",
            ha="center",
            fontweight="bold",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.tight_layout()
    out_path = ASSETS_DIR / "eda_token_distributions.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    plot_cleaning_and_noise()
    plot_token_distributions()


if __name__ == "__main__":
    main()
