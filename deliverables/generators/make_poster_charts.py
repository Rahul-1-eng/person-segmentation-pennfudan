import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F3A5F"
BLUE = "#2E6FB7"
ORANGE = "#E8940A"
GREEN = "#1B9E77"
GREY = "#9AA3AE"
INK = "#20304A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

OUT = "/app/outputs/poster"

# ---- (a) Test mean per-image IoU -------------------------------------------
models = ["pixel RF", "U-Net baseline", "U-Net + mild aug.", "ResNet34-U-Net", "DeepLab 0-shot", "DeepLab ft"]
iou = [0.2713, 0.6893, 0.7164, 0.7895, 0.7567, 0.7973]
std = [0.0091, 0.0146, 0.0015, 0.0094, 0.0, 0.0017]
colors = [GREY, BLUE, BLUE, ORANGE, GREEN, GREEN]

fig, ax = plt.subplots(figsize=(11.5, 4.0), dpi=300)
y = np.arange(len(models))[::-1]
bars = ax.barh(y, iou, xerr=std, height=0.62, color=colors, error_kw=dict(ecolor=INK, lw=1.4, capsize=3))
for yi, v in zip(y, iou):
    ax.text(v + 0.012, yi, f"{v:.4f}", va="center", ha="left", fontsize=15, color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=14)
ax.set_xlim(0, 0.92)
ax.set_xlabel("Test mean per-image IoU")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/poster_iou.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- (b) Distractor false-positive rate ------------------------------------
conds = ["floor", "mirror", "cast", "detach", "animal", "dark"]
fp_plain = [0.0520, 0.6209, 0.0183, 0.0633, 0.0063, 0.0275]
fp_distr = [0.0025, 0.0714, 0.0143, 0.0485, 0.0049, 0.0220]

x = np.arange(len(conds))
w = 0.38
fig, ax = plt.subplots(figsize=(11.5, 4.0), dpi=300)
ax.bar(x - w / 2, fp_plain, w, label="ResNet34-U-Net", color="#F2B84B", edgecolor=INK, linewidth=0.8)
ax.bar(x + w / 2, fp_distr, w, label="+ distractor training", color="#C95B1B", edgecolor=INK, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=13)
ax.set_ylabel("False-positive rate  FP$_D$")
ax.set_ylim(0, 0.72)
ax.legend(frameon=False, fontsize=14, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/poster_fpd.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("charts written")
