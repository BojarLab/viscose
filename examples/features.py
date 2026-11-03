"""One sheet per palette of what glycowork's plots need: hue, style, size, markers, color bars, arrows, and forests."""
import sys
from pathlib import Path
import numpy as np
import viscose as vc


def sheet(palette):
    rng = np.random.default_rng(3)
    n = 90
    x, y, weight = rng.normal(size = n), rng.normal(size = n), rng.random(n) * 10
    fig, axes = vc.subplots(2, 3, figsize = (15, 9), palette = palette)
    fig.suptitle(palette.capitalize(), 'Stand-in data')
    ax = axes[0][0]
    ax.scatter(x, y, hue = rng.choice(['Healthy', 'Early', 'Late'], n), style = rng.choice(['N-glycan', 'O-glycan'], n),
               size = weight, hue_label = 'Stage', style_label = 'Class', size_label = 'Weight')
    ax.set(xlabel = 'PC1 (31%)', ylabel = 'PC2 (12%)')
    ax.text(0.02, 0.98, 'F = 4.2\np = 0.003', axes_fraction = True, va = 'top')
    ax = axes[0][1]
    for i, marker in enumerate(vc.marks.MARKERS):
        ax.scatter([i], [1], s = 220, marker = marker, color = 'a1' if i % 2 else 'b1')
        ax.scatter([i], [0], s = 60, marker = marker, color = 'a0', edge = ('#888888', 0.8))
    ax.set(title = 'Markers', ylim = (-1, 2))
    ax = axes[0][2]
    m = rng.normal(size = (14, 9))
    m[:5] += 1.5
    m[:, :3] -= 1.0
    ax.heatmap(m, [f'S{i}' for i in range(9)], [f'Glycan {i}' for i in range(14)], colorbar = True,
               colorbar_label = 'z-score', show_all = True)
    ax.set_ylabel('Glycans')
    ax = axes[1][0]
    ax.scatter(x, y, c = weight, s = 50)
    ax.colorbar('Weight')
    ax.set_xlabel('PC1 (31%)')
    ax = axes[1][1]
    nodes = rng.random((7, 2))
    for i in range(6):
        ax.arrow(*nodes[i], *nodes[i + 1], gid = f'gwedge{i}')
    ax.scatter(nodes[:, 0], nodes[:, 1], s = 160 + 60 * np.arange(7), color = 'a1', edge = ('#888888', 0.8))
    for i in range(7):
        ax.annotate(f'Node {i}', nodes[i], offset = (10, 0))
    ax.set_title('Network')
    ax.margins(0.12)
    ax.off()
    ax = axes[1][2]
    ax.hlines(np.arange(6), rng.random(6) - 1, rng.random(6) + 0.3)
    ax.scatter(rng.random(6) - 0.2, np.arange(6), s = 40, marker = 'D', color = 'b0')
    ax.axvline(0)
    ax.set(xlabel = 'Effect size', title = 'Forest')
    ax.set_yticks(np.arange(6), [f'Study {i}' for i in range(6)])
    ax.invert_yaxis()
    return fig


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'examples')
    for name in vc.palettes:
        sheet(name).save(out / f'features_{name}.png', dpi = 100)
