"""
Figure 4: Checkpoint selection diagnostic workflow.

A practical decision framework derived from the paper's findings.
Steps 1-2 are empirically grounded in §3; Step 3 is a logical
consequence tested in §5.2 (argmax vs soup within Δ_min noise).

Usage:
    python fig4_diagnostic_flow.py

Output:
    fig4_diagnostic_flow.png (300 DPI)
    fig4_diagnostic_flow.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Layout
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Color palette
BOX_COLOR    = "#2c7bb6"
BOX_COLOR2   = "#fdae61"
BOX_YES      = "#1b9e77"
BOX_NO       = "#d7191c"
SUGGEST_COLOR = "#999999"

def draw_box(ax, cx, cy, w, h, text, subtext="",
             facecolor="#2c7bb6", edgecolor="white",
             linestyle="-", fontsize=10, sub_fontsize=7,
             alpha=0.9, zorder=3):
    """Draw a rounded rectangle box with centered text."""
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=2.5, linestyle=linestyle,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(cx, cy + 0.05, text, fontsize=fontsize,
            fontweight="bold", color="white",
            ha="center", va="center", zorder=zorder+1)
    if subtext:
        ax.text(cx, cy - h/2 - 0.15, subtext, fontsize=sub_fontsize,
                color="#666", ha="center", va="top",
                style="italic", zorder=zorder+1)
    return rect


def draw_diamond(ax, cx, cy, size, text, subtext="",
                 facecolor="#f0f0f0", edgecolor="#666",
                 fontsize=10, sub_fontsize=7):
    """Draw a diamond (decision node) with centered text."""
    diamond = mpatches.Polygon(
        np.array([
            [cx, cy + size],
            [cx + size, cy],
            [cx, cy - size],
            [cx - size, cy],
        ]),
        closed=True,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=2.0, zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(cx, cy + 0.05, text, fontsize=fontsize,
            fontweight="bold", color="#333",
            ha="center", va="center", zorder=4)
    if subtext:
        ax.text(cx, cy - size - 0.2, subtext, fontsize=sub_fontsize,
                color="#666", ha="center", va="top",
                style="italic", zorder=4)
    return diamond


def draw_arrow(ax, x1, y1, x2, y2, color="#666", lw=1.5,
               style="-", label=""):
    """Draw a labelled arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw, linestyle=style,
            connectionstyle="arc3,rad=0",
            shrinkA=5, shrinkB=5,
        ),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, fontsize=8, color=color,
                ha="left", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.8))


import numpy as np

# ============================================================
# Helper: add a small evidence-strength badge
# ============================================================
def add_badge(ax, cx, cy, w, h, text, color="#333", bg="#e8f4fd"):
    """Add a small badge at the top-right corner of a box."""
    bx = cx + w / 2 - 0.15
    by = cy + h / 2 - 0.15
    ax.text(bx, by, text, fontsize=5.5, color=color,
            ha="right", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor=bg,
                      edgecolor="none", alpha=0.8))

# ============================================================
# Nodes
# ============================================================

# Step 1 — Compute Δ_min
draw_box(ax, 5.0, 8.5, 4.0, 0.9,
         "Step 1: Compute Δ_min",
         "Δ_min = z·√(2p̄(1-p̄)/N)   (§3.4)",
         facecolor=BOX_COLOR)
add_badge(ax, 5.0, 8.5, 4.0, 0.9, "Verified",
          color="#1b6b3a", bg="#d4f0dc")

# Arrow 1→2
draw_arrow(ax, 5.0, 8.0, 5.0, 6.8)

# Step 2 — Estimate observed range
draw_box(ax, 5.0, 6.3, 4.0, 0.9,
         "Step 2: Estimate checkpoint range",
         "Run a few checkpoints, record min-to-max spread",
         facecolor=BOX_COLOR)
add_badge(ax, 5.0, 6.3, 4.0, 0.9, "Verified",
          color="#1b6b3a", bg="#d4f0dc")

# Arrow 2→decision
draw_arrow(ax, 5.0, 5.8, 5.0, 4.8)

# Decision diamond
draw_diamond(ax, 5.0, 3.8, 1.0,
             "Range < Δ_min?",
             "At your evaluation budget N",
             facecolor="#fff8e1", edgecolor="#f0b429")

# YES branch
draw_arrow(ax, 4.0, 3.8, 2.5, 2.5, label="Yes", color=BOX_YES)

# Step 3 — suggested (dashed, lighter)
draw_box(ax, 2.5, 1.5, 3.6, 1.0,
         "Step 3: Use noise-\nrobust aggregation",
         "e.g. uniform averaging, model soup",
         facecolor=SUGGEST_COLOR, edgecolor="#999",
         linestyle="dashed", fontsize=9, alpha=0.7)
add_badge(ax, 2.5, 1.5, 3.6, 1.0, "Tested §5.2",
          color="#666", bg="#f0f0f0")

# test result annotation
ax.annotate("⋄ tested in §5.2: gap below Δ_min\n   (argmax 50.0% vs soup 48.5%)",
            xy=(2.5, 0.45),
            fontsize=6, color="#999", ha="center", va="top",
            style="italic")

# NO branch
draw_arrow(ax, 6.0, 3.8, 7.5, 2.5, label="No", color=BOX_NO)

draw_box(ax, 7.5, 1.5, 3.6, 1.0,
         "Argmax may be viable",
         "But verify statistical power first",
         facecolor=BOX_NO, edgecolor="white",
         fontsize=9)

# ============================================================
# Reference note at bottom
# ============================================================
ax.text(5.0, 0.1,
        "Derived from §3 (Steps 1-2) and §5.2 (Step 3) of this paper",
        fontsize=6.5, color="#aaa", ha="center", va="bottom",
        style="italic")

# ============================================================
# Save
# ============================================================
plt.tight_layout()
fig.savefig("fig4_diagnostic_flow.png", dpi=300)
fig.savefig("fig4_diagnostic_flow.pdf")
plt.show()
