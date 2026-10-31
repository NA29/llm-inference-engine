"""Render results/*.json into results/plots/*.png (run locally: python -m bench.plots)."""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
OUT = RESULTS / "plots"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def gpu_note(data) -> str:
    dev = data["env"].get("device", {})
    return f"{dev.get('name', '?')} · torch {data['env'].get('torch', '?')} · commit {data['env'].get('git_sha', '?')}"


def bars(ax, labels, values, fmt):
    rects = ax.bar(labels, values, color=SERIES[:len(values)], width=0.55, edgecolor=SURFACE, linewidth=2)
    for r, v in zip(rects, values):
        ax.annotate(fmt(v), (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK)
    ax.grid(axis="x", visible=False)


def max_rate_under(data, slo_ms: float) -> float:
    """Highest measured rate whose p99 TTFT met slo_ms (0 if none did)."""
    return max((r["rate"] for r in data["runs"] if r["ttft_ms"]["p99"] <= slo_ms), default=0.0)


def throughput():
    ours, vllm = load("throughput_ours"), load("throughput_vllm")
    if not (ours and vllm):
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars(ax, ["this engine", f"vLLM {vllm['env'].get('vllm', '')}"],
         [ours["output_tok_per_s"], vllm["output_tok_per_s"]], lambda v: f"{v:,.0f}")
    w = ours["workload"]
    ax.set_title(f"Output throughput: {ours['output_tok_per_s'] / vllm['output_tok_per_s']:.0%} of vLLM")
    ax.set_ylabel("output tokens / s")
    ax.set_xlabel(f"{w['num_prompts']} requests, {w['input_len']} in / {w['output_len']} out, greedy, bf16\n"
                  + gpu_note(ours), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160)


def serving():
    cont, static = load("serving_continuous"), load("serving_static")
    if not (cont and static):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for data, color, label in ((cont, SERIES[0], "continuous batching"), (static, SERIES[1], "static batching")):
        runs = sorted(data["runs"], key=lambda r: r["rate"])
        best = data["max_rate_under_slo"]
        note = f"max {best:.3g} req/s under SLO" if best > 0 else "misses SLO at every tested rate"
        ax.plot([r["rate"] for r in runs], [r["ttft_ms"]["p99"] for r in runs], color=color, linewidth=2,
                marker="o", markersize=5, label=f"{label} ({note})")
    slo = cont["workload"]["slo_ms"]
    ax.axhline(slo, color=INK2, linestyle="--", linewidth=1)
    ax.annotate(f"{slo:.0f} ms SLO", (0.01, slo), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="left", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if static["max_rate_under_slo"] > 0:
        ax.set_title(f"p99 TTFT vs request rate "
                     f"({cont['max_rate_under_slo'] / static['max_rate_under_slo']:.1f}x the rate under the SLO)")
    else:
        ax.set_title("p99 time-to-first-token vs request rate")
    ax.set_xlabel("Poisson request rate (req/s)\n" + gpu_note(cont), fontsize=8)
    ax.set_ylabel("p99 TTFT (ms, log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.tight_layout()
    fig.savefig(OUT / "serving_slo.png", dpi=160)


def memory():
    data = load("memory")
    if not data:
        return
    r = data["results"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars(ax, ["preallocate\nmax_model_len", "preallocate\nprompt + max_tokens\n(oracle)", "paged blocks"],
         [r["prealloc"]["mean_concurrent"], r["prealloc_oracle"]["mean_concurrent"], r["paged"]["mean_concurrent"]],
         lambda v: f"{v:.0f}")
    gb = data["kv_budget_bytes"] / 2**30
    ax.set_title(f"Concurrent requests in {gb:.0f} GB of KV cache ({data['mean_concurrency_ratio']:.1f}x)")
    ax.set_ylabel("running requests (time-weighted mean)")
    ax.set_xlabel(gpu_note(data), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "memory.png", dpi=160)


def latency():
    data = load("latency")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (name, row), color in zip(data["decode_step_ms"].items(), SERIES):
        bs = sorted(row, key=int)
        ax.plot([int(b) for b in bs], [row[b] for b in bs], color=color, linewidth=2, marker="o",
                markersize=5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(b) for b in bs], [str(b) for b in bs])
    ax.minorticks_off()
    sp = data["speedup_vs_eager"]
    ax.set_title(f"Decode step latency ({min(sp.values()):.1f}x-{max(sp.values()):.1f}x faster than eager)")
    ax.set_xlabel("batch size (sequences decoding together)\n" + gpu_note(data), fontsize=8)
    ax.set_ylabel("ms per decode step")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (throughput, serving, memory, latency):
        fn()
    print(sorted(p.name for p in OUT.glob("*.png")))
