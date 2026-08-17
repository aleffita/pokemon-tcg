#!/usr/bin/env python3
"""
Generate High-Density Visual Research Dashboard for Pokémon TCG AI Challenge
Combines:
1. Autoresearch Policy Evolution (AR-015 to AR-027-retry)
2. Deck Saliency on Frozen Weights (Stage 4 baseline vs Deck variants)
3. AR-028 Opponent Breakdown
4. Hypergeometric Setup Curves
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import hypergeom

# Apply clean modern styling
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
fig.suptitle('Pokémon TCG AI Research — State of Experiments & Tournament Dynamics', fontsize=18, fontweight='bold', y=0.98, color='#38bdf8')

# -------------------------------------------------------------
# Panel 1: Autoresearch Policy Evolution (AR-015 to AR-027)
# -------------------------------------------------------------
ax1 = axes[0, 0]
ar_ids = ['AR-015', 'AR-017', 'AR-019', 'AR-020', 'AR-021', 'AR-022', 'AR-023', 'AR-024', 'AR-025', 'AR-026', 'AR-027r']
cand_vs_root = [80.0, 60.0, 60.0, 36.7, 73.3, 43.3, 66.7, 63.3, 43.3, 70.0, 43.3] # % WR vs root
cand_vs_panel = [42.5, 30.0, 30.0, 30.0, 26.7, 23.3, 23.3, 13.3, 16.7, 25.0, 15.0] # % WR vs panel

x = np.arange(len(ar_ids))
ax1.plot(x, cand_vs_root, marker='o', color='#22c55e', linewidth=2.5, label='Candidate vs Stage 4 Root (Same Deck)')
ax1.plot(x, cand_vs_panel, marker='s', color='#f59e0b', linewidth=2.5, label='Candidate vs External Panel (6 Opponents)')
ax1.axhline(y=20.0, color='#94a3b8', linestyle='--', label='Frozen Stage 4 Root Panel (20.0%)')

ax1.set_title('1. Policy Evolution & Generalization Gap (GRPO Iterations)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(ar_ids, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Win Rate (%)', fontsize=11)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)

# -------------------------------------------------------------
# Panel 2: Deck-Conditioned Win Rate Shift (Same Weights)
# -------------------------------------------------------------
ax2 = axes[0, 1]
decks = ['Starter #251', 'Stage 4 FP32', 'Yan #633', 'Supreme v0', 'v2 Control\n(vs Crustle)', 'v2 Control\n(vs Alakazam)']
wr_values = [12.9, 17.14, 27.9, 21.67, 60.0, 20.0]
colors = ['#64748b', '#38bdf8', '#10b981', '#06b6d4', '#a855f7', '#f43f5e']

bars = ax2.bar(decks, wr_values, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, wr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_title('2. Deck Saliency on Frozen Weights (Win Rate Multiplier)', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Win Rate (%)', fontsize=11)
ax2.set_ylim(0, 75)
ax2.grid(True, axis='y', alpha=0.2)

# -------------------------------------------------------------
# Panel 3: AR-028 Per-Opponent Matchup Performance
# -------------------------------------------------------------
ax3 = axes[1, 0]
opponents = ['random', 'first_sub', 'lb1009\n(Lucario)', 'lb945\n(Lucario)', 'lb826\n(Alakazam)', 'lb814\n(Crustle)']
root_opp = [70.0, 20.0, 0.0, 0.0, 10.0, 20.0]
v0_opp = [70.0, 20.0, 0.0, 0.0, 0.0, 40.0]
v1_opp = [40.0, 0.0, 0.0, 20.0, 0.0, 0.0]
v2_opp = [0.0, 40.0, 0.0, 0.0, 20.0, 60.0]

w = 0.2
idx = np.arange(len(opponents))
ax3.bar(idx - 1.5*w, root_opp, width=w, label='Stage 4 Root', color='#64748b')
ax3.bar(idx - 0.5*w, v0_opp, width=w, label='Deck v0 (Supreme)', color='#38bdf8')
ax3.bar(idx + 0.5*w, v1_opp, width=w, label='Deck v1 (Tempo)', color='#f59e0b')
ax3.bar(idx + 1.5*w, v2_opp, width=w, label='Deck v2 (Control)', color='#a855f7')

ax3.set_title('3. AR-028 Matchup Matrix by Deck Variant', fontsize=13, fontweight='bold', pad=10)
ax3.set_xticks(idx)
ax3.set_xticklabels(opponents, fontsize=9)
ax3.set_ylabel('Win Rate (%)', fontsize=11)
ax3.set_ylim(0, 85)
ax3.grid(True, axis='y', alpha=0.2)
ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.8)

# -------------------------------------------------------------
# Panel 4: Hypergeometric Resource Curves
# -------------------------------------------------------------
ax4 = axes[1, 1]
k_basics = np.arange(4, 17)
# P(at least 1 basic in 7 cards from 60)
p_single = [1.0 - hypergeom.pmf(0, 60, k, 7) for k in k_basics]
# P(setup within 1 mulligan) = 1 - (1 - P_single)^2
p_mulligan_1 = [1.0 - (1.0 - p)**2 for p in p_single]

ax4.plot(k_basics, [p*100 for p in p_single], marker='o', color='#38bdf8', linewidth=2, label='Initial Hand (n=7)')
ax4.plot(k_basics, [p*100 for p in p_mulligan_1], marker='s', color='#10b981', linewidth=2.5, label='Setup within 1 Mulligan')

# Highlight points
ax4.scatter([5], [p_single[1]*100], color='#ef4444', s=120, zorder=5)
ax4.annotate('Deck #633 (5 Basics)\nInitial: 47.5% / 1-Mull: 72.4%', (5, p_single[1]*100), textcoords="offset points", xytext=(-20, -35), ha='right', fontsize=8.5, color='#f87171', arrowprops=dict(arrowstyle="->", color='#ef4444'))

ax4.scatter([11], [p_mulligan_1[7]*100], color='#10b981', s=140, zorder=5)
ax4.annotate('Deck Supreme (11 Basics)\nInitial: 77.8% / 1-Mull: 95.1%', (11, p_mulligan_1[7]*100), textcoords="offset points", xytext=(-30, -30), ha='center', fontsize=8.5, color='#4ade80', arrowprops=dict(arrowstyle="->", color='#10b981'))

ax4.axhline(y=92.0, color='#f59e0b', linestyle='--', label='Target Setup Threshold (92.0%)')
ax4.set_title('4. Hypergeometric Opening Probability Function P(Setup)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel('Number of Basic Pokémon in 60-Card Deck ($K_b$)', fontsize=11)
ax4.set_ylabel('Probability (%)', fontsize=11)
ax4.set_ylim(40, 102)
ax4.grid(True, alpha=0.2)
ax4.legend(loc='lower right', fontsize=9, framealpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = Path('/Users/alefita/.gemini/antigravity/brain/efbc1117-8bdc-4429-8550-161f3ac70c9e/dashboard_experiments_visual.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Plot saved successfully to: {out_path}")
