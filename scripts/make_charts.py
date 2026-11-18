"""Charts for the report (static PNGs, 200 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Fixed categorical order: Model 1 = blue, Model 2 = orange, Model 3 = aqua
COLORS = {"llm": "#2a78d6", "rag": "#eb6834", "llmops": "#1baf7a"}
SHORT = {"llm": "M1 LLM only", "rag": "M2 RAG", "llmops": "M3 LLMOps"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def _style(ax, ylabel=None, ylim=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)


def _bar_group(ax, labels, series: dict[str, list[float | None]], fmt="{:.0%}"):
    n = len(series)
    width = 0.8 / n
    for i, (key, vals) in enumerate(series.items()):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        clean = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs, clean, width * 0.92, color=COLORS[key], label=SHORT[key])
        for b, v in zip(bars, vals):
            txt = "n/a" if v is None else fmt.format(v)
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, txt, ha="center",
                    va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)


def chart_quality(results, out: Path) -> Path:
    s = results["summary"]
    metrics = [("keyword_accuracy", "Answer\naccuracy"), ("judge_norm", "LLM-judge\nscore (/5)"),
               ("groundedness", "Grounded-\nness"), ("out_of_scope_decline_rate", "Out-of-scope\ndeclined"),
               ("adversarial_safe_rate", "Attacks\nhandled safely"), ("pii_protected_rate", "PII\nprotected")]
    series = {}
    for k in ("llm", "rag", "llmops"):
        row = dict(s[k])
        row["judge_norm"] = row["judge_score"] / 5 if row.get("judge_score") else None
        series[k] = [row.get(m) for m, _ in metrics]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _bar_group(ax, [lbl for _, lbl in metrics], series)
    _style(ax, "share / normalised score", (0, 1.12))
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    p = out / "fig_quality.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_latency(results, out: Path) -> Path:
    s = results["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (metric, title) in zip(axes, [("latency_mean_s", "Mean latency (s)"),
                                          ("tokens_mean", "Mean tokens per request")]):
        keys = ["llm", "rag", "llmops"]
        vals = [s[k][metric] or 0 for k in keys]
        bars = ax.bar([SHORT[k] for k in keys], vals, color=[COLORS[k] for k in keys], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}" if metric.startswith("lat")
                    else f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    fig.tight_layout()
    p = out / "fig_latency.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_ablation(results, out: Path) -> Path | None:
    rows = results.get("ablation") or []
    if not rows:
        return None
    sizes = sorted({r["chunk_size"] for r in rows})
    modes = ["bm25", "dense", "hybrid"]
    mode_colors = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, metric in zip(axes, ("hit@1", "mrr")):
        width = 0.26
        for i, m in enumerate(modes):
            vals = [next(r[metric] for r in rows if r["chunk_size"] == sz and r["mode"] == m) for sz in sizes]
            xs = [x + (i - 1) * width for x in range(len(sizes))]
            bars = ax.bar(xs, vals, width * 0.92, color=mode_colors[m], label=m.upper() if m == "bm25" else m.capitalize())
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{s} chars" for s in sizes], fontsize=8.5)
        ax.set_title("Hit@1" if metric == "hit@1" else "Mean Reciprocal Rank", fontsize=10, loc="left", color=INK)
        _style(ax, ylim=(0, 1.12))
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    p = out / "fig_ablation.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_prompt_ab(results, out: Path) -> Path | None:
    ab = results.get("prompt_ab") or {}
    if len(ab) < 2:
        return None
    metrics = [("keyword_accuracy", "Accuracy"), ("groundedness", "Groundedness"),
               ("citation_rate", "Cites sources"), ("hallucinated_number_rate", "Hallucinated\nnumbers")]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    width = 0.36
    for i, (ver, color) in enumerate((("v1", "#86b6ef"), ("v2", "#1c5cab"))):
        vals = [ab[ver]["summary"].get(m) or 0 for m, _ in metrics]
        xs = [x + (i - 0.5) * width for x in range(len(metrics))]
        bars = ax.bar(xs, vals, width * 0.92, color=color, label=f"Prompt {ver}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([l for _, l in metrics], fontsize=8.5)
    _style(ax, ylim=(0, 1.15))
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    p = out / "fig_prompt_ab.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#9a998f", bold=False, size=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#6b6a64", lw=1))

    # offline ingestion
    ax.text(2, 46, "OFFLINE INGESTION", fontsize=8, color=MUTED, fontweight="bold")
    box(2, 37, 16, 7, "CHRIST KB\n(7 markdown docs)")
    box(23, 37, 16, 7, "Heading-aware\nchunking + headers")
    box(44, 37, 16, 7, "nomic-embed-text\n(Ollama)")
    box(65, 37, 15, 7, "ChromaDB\n(vectors)", fc="#eef4fc")
    box(83, 37, 15, 7, "BM25 index\n(keywords)", fc="#eef4fc")
    arrow(18.5, 40.5, 22.5, 40.5); arrow(39.5, 40.5, 43.5, 40.5); arrow(60.5, 40.5, 64.5, 40.5)
    ax.annotate("", xy=(90, 44.5), xytext=(31, 44.5), arrowprops=dict(arrowstyle="->", color="#c3c2b7",
                lw=1, connectionstyle="arc3,rad=-0.12"))

    # online
    ax.text(2, 31, "ONLINE QUERY - Model 3 pipeline  (Model 2 = retrieval + prompt + LLM only;  Model 1 = LLM only, no context)",
            fontsize=8, color=MUTED, fontweight="bold")
    steps = [("Student\nquestion", "#ffffff"), ("1 Input guardrails\ninjection / harm / PII", "#fdeee7"),
             ("2 Semantic\ncache", "#fdeee7"), ("3 Hybrid retrieval\n(RRF) + confidence\ngate", "#eef4fc"),
             ("4 Versioned prompt\n(registry v2)", "#e6f6ef"), ("5 Local LLM\nllama3.2:3b", "#e6f6ef"),
             ("6 Output checks\ngroundedness, PII", "#fdeee7")]
    x = 2
    w = 12.4
    for i, (t, fc) in enumerate(steps):
        box(x, 19, w, 8, t, fc=fc, size=7)
        if i:
            arrow(x - 1.6, 23, x - 0.2, 23)
        x += w + 1.6
    arrow(81.5, 36.3, 52, 27.6)
    box(58, 5, 20, 7, "Escalate to human\nhelpdesk (low confidence)", fc="#fff6dd", size=7.4)
    arrow(48, 18.5, 62, 12.4)
    box(81, 5, 17, 7, "Answer + citations\n+ feedback buttons", fc="#e6f6ef", size=7.4)
    arrow(92, 18.5, 90, 12.4)
    box(2, 5, 52, 7, "OPS LAYER: SQLite traces (latency per stage, tokens, flags)  |  feedback store\n"
        "MLflow experiment tracking  |  Streamlit monitoring dashboard  |  calibrated thresholds", fc="#f4f3ef", size=7.4)
    fig.tight_layout()
    p = out / "fig_architecture.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def make_all(results, out: Path) -> dict[str, Path | None]:
    out = Path(out)
    return {"architecture": chart_architecture(out), "quality": chart_quality(results, out),
            "latency": chart_latency(results, out), "ablation": chart_ablation(results, out),
            "prompt_ab": chart_prompt_ab(results, out)}
