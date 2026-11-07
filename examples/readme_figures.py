"""Draws every image in the README into examples/readme/: python examples/readme_figures.py
Needs matplotlib and seaborn for the comparison, pillow for arranging the panels, and glycowork for the glycan
drawings."""
import tempfile
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import viscose as vc
import gallery

OUT = Path(__file__).parent / 'readme'
EFFECTS = {'lumen': 'Spheres, gradients, glow, soft shadows', 'couture': 'Rings, needles, hairlines, spaced capitals',
           'poster': 'White-cut marks, heavy rounded axes, bold type',
           'aura': 'Marks floating on shadows of their own color'}
STRIPS = {'lumen': ('embedding', 'trial', 'ridges'), 'couture': ('spectrum', 'stacked', 'volcano'),
          'poster': ('volcano', 'stacked', 'ridges'), 'aura': ('heatmap', 'trial', 'embedding')}


def tile(fig, side, dpi = 200):
    """A figure as a square white image of the given side, scaled to fit."""
    img = Image.open(BytesIO(fig.to_png(dpi))).convert('RGB')
    return fit(img, side, side)


def fit(img, w, h):
    scale = min(w / img.width, h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    out = Image.new('RGB', (w, h), 'white')
    out.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return out


def arrange(rows, name, gap = 40):
    """Rows of equal images on white, saved under the given name."""
    w, h = rows[0][0].size
    out = Image.new('RGB', (len(rows[0]) * w + (len(rows[0]) - 1) * gap, len(rows) * h + (len(rows) - 1) * gap),
                    'white')
    for r, row in enumerate(rows):
        for j, img in enumerate(row):
            out.paste(img, (j * (w + gap), r * (h + gap)))
    out.save(OUT / name, optimize = True)


def card(palette):
    """A palette's effects on one small scene, above its eight shades."""
    fig, (ax, shades) = vc.subplots(2, 1, figsize = (6, 6), palette = palette, height_ratios = [5, 1])
    fig.suptitle(palette.capitalize(), EFFECTS[palette])
    rng = np.random.default_rng(5)
    x = np.linspace(0, 10, 60)
    ax.bar([1.5, 3, 4.5, 6, 7.5, 9], [2.2, 3.4, 2.9, 4.4, 3.8, 5.1], color = 'a2', z = 1)
    ax.plot(x, 4.6 + 1.6 * np.sin(x / 1.6), color = 'a0')
    ax.scatter(rng.uniform(0.5, 9.5, 12), rng.uniform(1, 7.5, 12), s = 190, color = 'b1', z = 6)
    ax.scatter(rng.uniform(0.5, 9.5, 10), rng.uniform(1, 7.5, 10), s = 110, color = 'a1', z = 6)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 8)
    ax.off()
    for i, shade in enumerate(f'{hue}{i}' for hue in 'ab' for i in range(4)):
        shades.fill_between([i, i + 0.86], 0, 1, color = shade, alpha = 1.0)
    shades.set_ylim(0, 1)
    shades.off()
    return fig


def seaborn_volcano():
    """gallery.volcano's data and labels as matplotlib and seaborn draw them by default."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    rng = np.random.default_rng(1729)
    x = np.clip(rng.normal(0, 1.3, 412), -4.6, 4.6)
    y = 0.72 * np.abs(x) ** 1.75 * rng.uniform(.8, 1.05, 412) + rng.exponential(0.42, 412)
    hits = [(3.4, 12.6, 'Neu5Ac(a2-6)Gal(b1-4)GlcNAc'), (-3.1, 11.6, 'Gal(a1-3)Gal(b1-4)GlcNAc'),
            (2.7, 10.4, 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'), (-2.4, 9.4, 'GlcNAc(b1-2)Man(a1-3)Man'),
            (3.7, 8.4, 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'), (-3.6, 7.4, 'Man(a1-2)Man(a1-3)Man'),
            (3.1, 6.5, 'GalNAc(b1-4)Gal(b1-4)Glc'), (-2.9, 5.6, 'Fuc(a1-2)Gal(b1-3)GalNAc')]
    group = np.where((np.abs(x) > 1) & (y > 2), np.where(x < 0, 'Down', 'Up'), 'Unchanged')
    fig, ax = plt.subplots(figsize = (8, 8))
    sns.scatterplot(x = x, y = y, hue = group, ax = ax)
    for hx, hy, lab in hits:
        ax.scatter([hx], [hy], s = 120, color = 'C1' if hx < 0 else 'C2')
        ax.annotate(lab, (hx, hy), textcoords = 'offset points', xytext = (11 if hx < 0 else -11, 0),
                    ha = 'left' if hx < 0 else 'right', va = 'center')
    for v in (-1, 1):
        ax.axvline(v, color = 'grey', linestyle = ':')
    ax.axhline(2, color = 'grey', linestyle = ':')
    ax.set_xlabel('log2 fold change')
    ax.set_ylabel('-log10 adjusted p')
    ax.set_title('Differential abundance')
    buf = BytesIO()
    fig.savefig(buf, dpi = 200, format = 'png')
    plt.close(fig)
    return Image.open(buf).convert('RGB')


def glycan_volcano():
    rng = np.random.default_rng(1729)
    x = np.clip(rng.normal(0, 1.3, 412), -4.6, 4.6)
    y = 0.72 * np.abs(x) ** 1.75 * rng.uniform(.8, 1.05, 412) + rng.exponential(0.42, 412)
    hits = [(0.6, 12.8, 'Neu5Ac(a2-6)Gal(b1-4)GlcNAc'), (-3.6, 11.2, 'Gal(a1-3)Gal(b1-4)GlcNAc'),
            (2.2, 9.9, 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'), (-3.2, 9.0, 'GlcNAc(b1-2)Man(a1-3)Man'),
            (3.0, 8.2, 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'), (-3.8, 5.6, 'Fuc(a1-2)Gal(b1-3)GalNAc')]
    fig, ax = vc.subplots(figsize = (7, 7), palette = 'aura')
    fig.suptitle('Glycans as labels', 'Real SNFG drawings on a volcano plot')
    sig = (np.abs(x) > 1) & (y > 2)
    ax.scatter(x[~sig], y[~sig], s = 22, color = 'a3', z = 2)
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s = 30, color = 'b1')
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s = 30, color = 'a1')
    for hx, hy, lab in hits:
        ax.scatter([hx], [hy], s = 120, color = 'b1' if hx < 0 else 'a1', z = 5)
        ax.annotate(lab, (hx, hy), offset = (12, 0), svg_label = True)
    ax.axvline(-1)
    ax.axvline(1)
    ax.axhline(2)
    ax.set_xlim(-4.9, 5.4)
    ax.set_ylim(0, 14.2)
    ax.set_xlabel('log_{2} fold change')
    ax.set_ylabel('-log_{10} adjusted p')
    return fig


def glycan_network():
    nodes = {'GlcNAc': (0.08, 0.1), 'Gal(b1-4)GlcNAc': (0.3, 0.38), 'Fuc(a1-3)GlcNAc': (0.38, 0.06),
             'Neu5Ac(a2-6)Gal(b1-4)GlcNAc': (0.4, 0.78), 'Fuc(a1-3)[Gal(b1-4)]GlcNAc': (0.68, 0.3),
             'Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc': (0.85, 0.86)}
    edges = [('GlcNAc', 'Gal(b1-4)GlcNAc'), ('GlcNAc', 'Fuc(a1-3)GlcNAc'),
             ('Gal(b1-4)GlcNAc', 'Neu5Ac(a2-6)Gal(b1-4)GlcNAc'),
             ('Gal(b1-4)GlcNAc', 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'), ('Fuc(a1-3)GlcNAc', 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'),
             ('Fuc(a1-3)[Gal(b1-4)]GlcNAc', 'Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc')]
    fig, ax = vc.subplots(figsize = (7, 7), palette = 'aura')
    fig.suptitle('Glycans as nodes', 'A biosynthetic network, with a drawing at every step')
    for i, (a, b) in enumerate(edges):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        n = np.hypot(x1 - x0, y1 - y0)
        ax.arrow(x0, y0, x1 - 0.04 * (x1 - x0) / n, y1 - 0.04 * (y1 - y0) / n, color = 'b1', lw = 0.9,
                 gid = f'gwedge{i}')
    for i, (name, (x, y)) in enumerate(nodes.items()):
        ax.scatter([x], [y], s = 150, color = 'a1', gid = f'gwnode{i}', z = 5)
        ax.annotate(name, (x, y), offset = (12, 0), svg_label = True)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.04, 1.04)
    ax.off()
    return fig


def with_glycans(fig):
    from glycowork.motif.draw import annotate_figure
    with tempfile.TemporaryDirectory() as tmp:
        fig.save(Path(tmp) / 'plot.svg')
        annotate_figure(str(Path(tmp) / 'plot.svg'), filepath = str(Path(tmp) / 'plot.png'), compact = True)
        return Image.open(Path(tmp) / 'plot.png').convert('RGB').copy()


def markers():
    rng = np.random.default_rng(3)
    n = 150
    stage = np.repeat(['Healthy', 'Early', 'Late'], n // 3)
    shift = np.select([stage == 'Healthy', stage == 'Early'], [-1.6, 0.2], 1.8)
    x, y = rng.normal(shift, 0.8), rng.normal(shift ** 2 / 2 - 0.8, 0.7)
    fig, ax = vc.subplots(figsize = (8, 6), palette = 'aura')
    fig.suptitle('Glycomes by disease stage', '150 patients')
    ax.scatter(x, y, hue = stage, style = rng.choice(['Serum', 'Tissue'], n), size = rng.random(n) * 10,
               hue_label = 'Stage', style_label = 'Source', size_label = 'Abundance')
    ax.text(0.02, 0.98, 'F = 4.2\np = 0.003', axes_fraction = True, va = 'top')
    ax.set(xlabel = 'PC1 (31%)', ylabel = 'PC2 (12%)')
    return fig


if __name__ == '__main__':
    OUT.mkdir(exist_ok = True)
    names = list(vc.palettes)
    figures = {f.__name__: f for f in gallery.FIGURES.values()}
    arrange([[tile(gallery.ridges(p), 640) for p in names],
             [tile(figures[k](p), 640) for k, p in zip(('embedding', 'heatmap', 'volcano', 'trial'), names)]],
            'hero.png')
    arrange([[tile(gallery.volcano(p), 900) for p in names[:2]], [tile(gallery.volcano(p), 900) for p in names[2:]]],
            'one_word.png')
    arrange([[tile(card(p), 900) for p in names[:2]], [tile(card(p), 900) for p in names[2:]]], 'palettes.png')
    for p in names:
        arrange([[tile(figures[k](p), 640) for k in STRIPS[p]]], f'{p}.png')
    arrange([[fit(seaborn_volcano(), 900, 900), tile(gallery.volcano('aura'), 900)]], 'versus.png')
    arrange([[fit(with_glycans(glycan_volcano()), 900, 900), fit(with_glycans(glycan_network()), 900, 900)]],
            'glycans.png')
    arrange([[fit(Image.open(BytesIO(markers().to_png(200))).convert('RGB'), 1400, 1000)]], 'hue_style_size.png')
    arrange([[fit(Image.open(BytesIO(gallery.groups('lumen').to_png(200))).convert('RGB'), 1400, 840)]], 'groups.png')
    arrange([[fit(Image.open(BytesIO(gallery.subgroups('couture').to_png(200))).convert('RGB'), 1400, 840)]],
            'subgroups.png')
    arrange([[tile(gallery.heatmap('aura'), 1100)]], 'clustermap.png')
    for path in sorted(OUT.glob('*.png')):
        print(path.name, path.stat().st_size // 1024, 'kB')
