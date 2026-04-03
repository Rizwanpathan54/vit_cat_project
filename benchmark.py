import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import psutil
import os
import platform

# ── Optional GPU power monitoring ────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("pynvml found — GPU power & temperature will be measured.")
except Exception:
    NVML_AVAILABLE = False
    print("pynvml not found — skipping GPU power/temperature.")
    print("Install with: pip install pynvml\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {device}")
print(f"CPU     : {platform.processor()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ─────────────────────────────────────────────
#  SHARED HYPERPARAMETERS
# ─────────────────────────────────────────────
IMAGE_SIZE     = 224
PATCH_SIZE     = 16
CHANNELS       = 3
NUM_CLASSES    = 2
DROPOUT        = 0.0    # always 0 for inference benchmarking
NUM_EXPERTS    = 4
TOP_K          = 2

# Vanilla ViT + Original MoE ViT dims
EMBED_DIM_BASE  = 256
MLP_DIM_BASE    = 512
NUM_HEADS_BASE  = 4
NUM_LAYERS_BASE = 6

# Optimized MoE ViT dims (same safe settings for RTX 3050)
EMBED_DIM_OPT   = 256
MLP_DIM_OPT     = 512
NUM_HEADS_OPT   = 4
NUM_LAYERS_OPT  = 6


# ─────────────────────────────────────────────
#  MODEL 1 — VANILLA VIT
#  File: vit_cat_detector.pth
# ─────────────────────────────────────────────
class PatchEmbeddingsVanilla(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.proj      = nn.Conv2d(CHANNELS, EMBED_DIM_BASE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, EMBED_DIM_BASE))
        self.pos_embed  = nn.Parameter(torch.randn(1, self.num_patches + 1, EMBED_DIM_BASE))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat((cls, x), dim=1) + self.pos_embed

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbeddingsVanilla()
        enc = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM_BASE, nhead=NUM_HEADS_BASE,
            dim_feedforward=MLP_DIM_BASE, dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=NUM_LAYERS_BASE)
        self.head = nn.Sequential(nn.LayerNorm(EMBED_DIM_BASE), nn.Linear(EMBED_DIM_BASE, NUM_CLASSES))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.head(x[:, 0])


# ─────────────────────────────────────────────
#  SHARED MOE BLOCK FACTORY
# ─────────────────────────────────────────────
class MoEBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.top_k       = TOP_K
        self.router  = nn.Linear(embed_dim, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(mlp_dim, embed_dim), nn.Dropout(DROPOUT)
            ) for _ in range(NUM_EXPERTS)
        ])
        self.register_buffer("expert_usage", torch.zeros(NUM_EXPERTS))
        self.last_routing = None

    def forward(self, x):
        orig  = x.shape
        xf    = x.view(-1, orig[-1])
        gl    = self.router(xf)
        w     = F.softmax(gl, dim=-1)
        tw, ti = torch.topk(w, self.top_k, dim=-1)
        tw    = tw / (tw.sum(-1, keepdim=True) + 1e-6)
        rp    = w.mean(0)
        df    = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            df[i] = (ti == i).float().mean()
        aux = self.num_experts * (rp * df).sum()
        for i in range(self.num_experts):
            self.expert_usage[i] += (ti == i).sum()
        self.last_routing = ti.detach().cpu()
        out = torch.zeros_like(xf)
        for i, exp in enumerate(self.experts):
            mask = (ti == i)
            tm   = mask.any(-1)
            if not tm.any(): continue
            ew   = (tw * mask).sum(-1, keepdim=True)
            out[tm] += ew[tm] * exp(xf[tm])
        return out.view(orig), aux


# ─────────────────────────────────────────────
#  MODEL 2 — ORIGINAL MOE VIT
#  File: vit_moe_cat_detector_moe.pth
# ─────────────────────────────────────────────
# ── Original MoE ViT uses a SIMPLER MoE block (with router bias, no aux loss) ──
class MoEBlockOrig(nn.Module):
    """Matches the ORIGINAL vit_moe_cat_detector_moe.pth exactly:
       - router has bias=True
       - no attn_scale / moe_scale
       - no patch_embed.norm
       - single Linear head
       - no aux loss returned
    """
    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.top_k       = TOP_K
        self.router  = nn.Linear(EMBED_DIM_BASE, NUM_EXPERTS)   # bias=True (default)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EMBED_DIM_BASE, MLP_DIM_BASE), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(MLP_DIM_BASE, EMBED_DIM_BASE), nn.Dropout(DROPOUT)
            ) for _ in range(NUM_EXPERTS)
        ])
        self.register_buffer("expert_usage", torch.zeros(NUM_EXPERTS))
        self.last_routing = None

    def forward(self, x):
        orig  = x.shape
        xf    = x.view(-1, orig[-1])
        gl    = self.router(xf)
        w     = F.softmax(gl, dim=-1)
        tw, ti = torch.topk(w, self.top_k, dim=-1)
        tw    = tw / (tw.sum(-1, keepdim=True) + 1e-6)
        for i in range(self.num_experts):
            self.expert_usage[i] += (ti == i).sum()
        self.last_routing = ti.detach().cpu()
        out = torch.zeros_like(xf)
        for i, exp in enumerate(self.experts):
            mask = (ti == i)
            tm   = mask.any(-1)
            if not tm.any(): continue
            ew   = (tw * mask).sum(-1, keepdim=True)
            out[tm] += ew[tm] * exp(xf[tm])
        return out.view(orig)

class PatchEmbOrigMoE(nn.Module):
    """No LayerNorm — matches original file."""
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.proj      = nn.Conv2d(CHANNELS, EMBED_DIM_BASE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, EMBED_DIM_BASE))
        self.pos_embed  = nn.Parameter(torch.randn(1, self.num_patches + 1, EMBED_DIM_BASE))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat((cls, x), dim=1) + self.pos_embed

class LayerOrigMoE(nn.Module):
    """No attn_scale / moe_scale params — matches original file."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM_BASE)
        self.attn  = nn.MultiheadAttention(EMBED_DIM_BASE, NUM_HEADS_BASE, dropout=DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(EMBED_DIM_BASE)
        self.moe   = MoEBlockOrig()

    def forward(self, x):
        n = self.norm1(x); a, _ = self.attn(n, n, n)
        x = x + a
        return x + self.moe(self.norm2(x))

class VisionTransformerOrigMoE(nn.Module):
    """Single-linear head — matches original file."""
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbOrigMoE()
        self.encoder     = nn.ModuleList([LayerOrigMoE() for _ in range(NUM_LAYERS_BASE)])
        # Original head: just LayerNorm + Linear (no intermediate layer)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM_BASE),
            nn.Linear(EMBED_DIM_BASE, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.encoder:
            x = layer(x)
        return self.head(x[:, 0])


# ─────────────────────────────────────────────
#  MODEL 3 — OPTIMIZED MOE VIT
#  File: vit_moe_optimized.pth
#  Adds: AdamW, label smoothing, cosine LR,
#        grad clip, stronger augmentation
# ─────────────────────────────────────────────
class PatchEmbOptMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.proj      = nn.Conv2d(CHANNELS, EMBED_DIM_OPT, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, EMBED_DIM_OPT) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, self.num_patches + 1, EMBED_DIM_OPT) * 0.02)
        self.norm       = nn.LayerNorm(EMBED_DIM_OPT)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return self.norm(torch.cat((cls, x), dim=1) + self.pos_embed)

class LayerOptMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1      = nn.LayerNorm(EMBED_DIM_OPT)
        self.attn       = nn.MultiheadAttention(EMBED_DIM_OPT, NUM_HEADS_OPT, dropout=DROPOUT, batch_first=True)
        self.norm2      = nn.LayerNorm(EMBED_DIM_OPT)
        self.moe        = MoEBlock(EMBED_DIM_OPT, MLP_DIM_OPT)
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.moe_scale  = nn.Parameter(torch.ones(1))

    def forward(self, x):
        n = self.norm1(x); a, _ = self.attn(n, n, n)
        x = x + self.attn_scale * a
        mo, aux = self.moe(self.norm2(x))
        return x + self.moe_scale * mo, aux

class VisionTransformerOptMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbOptMoE()
        self.encoder     = nn.ModuleList([LayerOptMoE() for _ in range(NUM_LAYERS_OPT)])
        self.norm        = nn.LayerNorm(EMBED_DIM_OPT)
        self.head        = nn.Sequential(
            nn.Linear(EMBED_DIM_OPT, EMBED_DIM_OPT // 2), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(EMBED_DIM_OPT // 2, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        aux_total = torch.tensor(0.0, device=x.device)
        for layer in self.encoder:
            x, aux = layer(x); aux_total += aux
        return self.head(self.norm(x)[:, 0]), aux_total


# ─────────────────────────────────────────────
#  LOAD ALL THREE MODELS
# ─────────────────────────────────────────────
VANILLA_PT  = "vit_cat_detector.pth"
ORIG_MOE_PT = "vit_moe_cat_detector_moe.pth"
OPT_MOE_PT  = "vit_moe_optimized.pth"

print("Loading models...")
vanilla_model  = VisionTransformer().to(device)
vanilla_model.load_state_dict(torch.load(VANILLA_PT, map_location=device))
vanilla_model.eval()
print("  ✓ Vanilla ViT         —", VANILLA_PT)

orig_moe_model = VisionTransformerOrigMoE().to(device)
orig_moe_model.load_state_dict(torch.load(ORIG_MOE_PT, map_location=device))
orig_moe_model.eval()
print("  ✓ Original MoE ViT    —", ORIG_MOE_PT)

opt_moe_model  = VisionTransformerOptMoE().to(device)
opt_moe_model.load_state_dict(torch.load(OPT_MOE_PT, map_location=device))
opt_moe_model.eval()
print("  ✓ Optimized MoE ViT   —", OPT_MOE_PT)
print()


# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
def get_gpu_stats():
    if not NVML_AVAILABLE: return None, None, None
    pw  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    tmp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return pw, tmp, mem.used / 1e9


# ─────────────────────────────────────────────
#  BENCHMARK FUNCTION
# ─────────────────────────────────────────────
def benchmark_model(model, model_name, batch_sizes=[1, 8, 16, 32], n_runs=50, warmup=10):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    results = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad():
            for _ in range(warmup): model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies, powers, temps, cpus = [], [], [], []

        with torch.no_grad():
            for _ in range(n_runs):
                cpus.append(psutil.cpu_percent(interval=None))
                pw, tmp, _ = get_gpu_stats()
                if pw:  powers.append(pw)
                if tmp: temps.append(tmp)

                if device == "cuda":
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(); model(dummy); e.record()
                    torch.cuda.synchronize()
                    latencies.append(s.elapsed_time(e))
                else:
                    t0 = time.perf_counter()
                    model(dummy)
                    latencies.append((time.perf_counter() - t0) * 1000)

        vram  = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        ram   = psutil.Process(os.getpid()).memory_info().rss / 1e9
        lm    = np.mean(latencies)
        ls    = np.std(latencies)
        tput  = (bs / lm) * 1000

        results[bs] = {
            "latency_mean_ms" : lm,   "latency_std_ms"  : ls,
            "throughput_img_s": tput, "vram_peak_gb"    : vram,
            "gpu_power_w"     : np.mean(powers) if powers else 0,
            "gpu_temp_c"      : np.mean(temps)  if temps  else 0,
            "cpu_pct"         : np.mean(cpus),  "ram_gb" : ram,
        }

        print(f"\n  BS={bs}  |  {lm:.2f}±{ls:.2f} ms  |  {tput:.0f} img/s  |  VRAM {vram:.3f}GB", end="")
        if powers: print(f"  |  {np.mean(powers):.0f}W", end="")
        if temps:  print(f"  |  {np.mean(temps):.0f}°C", end="")
        print(f"  |  CPU {np.mean(cpus):.0f}%")

        del dummy
        if device == "cuda": torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────
#  MODEL SIZE
# ─────────────────────────────────────────────
def model_stats(model, name):
    total   = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  {name:<24} {total:>12,} params   {size_mb:>7.1f} MB")
    return total, size_mb

print("=" * 60)
print("  MODEL SIZE COMPARISON")
print("=" * 60)
v_p,  v_s  = model_stats(vanilla_model,  "Vanilla ViT")
om_p, om_s = model_stats(orig_moe_model, "Original MoE ViT")
op_p, op_s = model_stats(opt_moe_model,  "Optimized MoE ViT")
print(f"\n  Orig MoE / Vanilla  : {om_p/v_p:.2f}x params")
print(f"  Opt MoE  / Vanilla  : {op_p/v_p:.2f}x params")


# ─────────────────────────────────────────────
#  RUN BENCHMARKS
# ─────────────────────────────────────────────
BATCH_SIZES = [1, 8, 16, 32]

vanilla_r  = benchmark_model(vanilla_model,  "Vanilla ViT",       BATCH_SIZES)
orig_moe_r = benchmark_model(orig_moe_model, "Original MoE ViT",  BATCH_SIZES)
opt_moe_r  = benchmark_model(opt_moe_model,  "Optimized MoE ViT", BATCH_SIZES)


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
print("\nGenerating plots...")

COLORS = ["#4C9BE8", "#E8714C", "#4CAF50"]   # blue / orange / green
LABELS = ["Vanilla ViT", "Original MoE ViT", "Optimized MoE ViT"]

metrics = [
    ("latency_mean_ms",  "Latency (ms)",         "lower is better ↓"),
    ("throughput_img_s", "Throughput (imgs/sec)", "higher is better ↑"),
    ("vram_peak_gb",     "Peak VRAM (GB)",        "lower is better ↓"),
    ("gpu_power_w",      "GPU Power Draw (W)",    "lower is better ↓"),
    ("gpu_temp_c",       "GPU Temperature (°C)",  "lower is better ↓"),
    ("cpu_pct",          "CPU Usage (%)",         "lower is better ↓"),
]

fig = plt.figure(figsize=(20, 13))
fig.suptitle(
    "Vanilla ViT  vs  Original MoE ViT  vs  Optimized MoE ViT\n"
    "Performance Benchmark — RTX 3050",
    fontsize=14, fontweight='bold'
)
gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

for ax, (key, ylabel, note) in zip(axes, metrics):
    all_vals = [
        [vanilla_r[bs][key]  for bs in BATCH_SIZES],
        [orig_moe_r[bs][key] for bs in BATCH_SIZES],
        [opt_moe_r[bs][key]  for bs in BATCH_SIZES],
    ]
    x = np.arange(len(BATCH_SIZES))
    w = 0.25
    offsets = [-w, 0, w]

    for vals, color, label, offset in zip(all_vals, COLORS, LABELS, offsets):
        bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                        f"{h:.1f}", ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={bs}" for bs in BATCH_SIZES], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel}\n{note}", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved benchmark_results.png")


# ─────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────
bs  = 32
vr  = vanilla_r[bs]
omr = orig_moe_r[bs]
opr = opt_moe_r[bs]

def fmt(val): return f"{val:.1f}" if val else "N/A"

print("\n" + "=" * 78)
print("  SUMMARY TABLE  (Batch Size = 32)")
print("=" * 78)
print(f"{'Metric':<26} {'Vanilla ViT':>14} {'Orig MoE ViT':>14} {'Opt MoE ViT':>14}")
print("-" * 78)

rows = [
    ("Parameters",        f"{v_p:,}",          f"{om_p:,}",         f"{op_p:,}"),
    ("Model Size (MB)",   f"{v_s:.1f}",         f"{om_s:.1f}",       f"{op_s:.1f}"),
    ("Latency (ms)",      f"{vr['latency_mean_ms']:.2f}", f"{omr['latency_mean_ms']:.2f}", f"{opr['latency_mean_ms']:.2f}"),
    ("Throughput (img/s)",f"{vr['throughput_img_s']:.1f}", f"{omr['throughput_img_s']:.1f}", f"{opr['throughput_img_s']:.1f}"),
    ("VRAM Peak (GB)",    f"{vr['vram_peak_gb']:.3f}", f"{omr['vram_peak_gb']:.3f}", f"{opr['vram_peak_gb']:.3f}"),
    ("GPU Power (W)",     fmt(vr['gpu_power_w']), fmt(omr['gpu_power_w']), fmt(opr['gpu_power_w'])),
    ("GPU Temp (°C)",     fmt(vr['gpu_temp_c']),  fmt(omr['gpu_temp_c']),  fmt(opr['gpu_temp_c'])),
    ("CPU Usage (%)",     f"{vr['cpu_pct']:.1f}", f"{omr['cpu_pct']:.1f}", f"{opr['cpu_pct']:.1f}"),
    ("RAM Usage (GB)",    f"{vr['ram_gb']:.2f}",  f"{omr['ram_gb']:.2f}",  f"{opr['ram_gb']:.2f}"),
]

for label, a, b, c in rows:
    print(f"  {label:<24} {a:>14} {b:>14} {c:>14}")

print("=" * 78)
print("\nDone. benchmark_results.png saved.")