"""The eight standard figures in every palette, from seeded demonstration data: python examples/gallery.py [outdir]"""
import sys
from pathlib import Path
import numpy as np
import viscose as vc


def volcano(palette):
    rng = np.random.default_rng(1729)
    x = np.clip(rng.normal(0, 1.3, 412), -4.6, 4.6)
    y = 0.72 * np.abs(x) ** 1.75 * rng.uniform(.8, 1.05, 412) + rng.exponential(0.42, 412)
    hits = [(3.4, 12.6, 'Neu5Ac(a2-6)Gal(b1-4)GlcNAc'), (-3.1, 11.6, 'Gal(a1-3)Gal(b1-4)GlcNAc'),
            (2.7, 10.4, 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'), (-2.4, 9.4, 'GlcNAc(b1-2)Man(a1-3)Man'),
            (3.7, 8.4, 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'), (-3.6, 7.4, 'Man(a1-2)Man(a1-3)Man'),
            (3.1, 6.5, 'GalNAc(b1-4)Gal(b1-4)Glc'), (-2.9, 5.6, 'Fuc(a1-2)Gal(b1-3)GalNAc')]
    fig, ax = vc.subplots(palette = palette)
    fig.suptitle('Differential abundance', 'Tumor vs. healthy tissue, 412 glycans')
    sig = (np.abs(x) > 1) & (y > 2)
    ax.scatter(x[~sig], y[~sig], s = 22, color = 'a3', z = 2)
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s = 30, color = 'b1')
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s = 30, color = 'a1')
    for hx, hy, lab in hits:
        ax.scatter([hx], [hy], s = 120, color = 'b1' if hx < 0 else 'a1', z = 5)
        ax.annotate(lab, (hx, hy), offset = (11 if hx < 0 else -11, 0), ha = 'left' if hx < 0 else 'right')
    ax.axvline(-1)
    ax.axvline(1)
    ax.axhline(2)
    ax.set_xlim(-4.9, 4.9)
    ax.set_ylim(0, 14.2)
    ax.text(-4.75, 13.5, 'Down in tumor', size = 11, color = 'b0', bold = True)
    ax.text(4.75, 13.5, 'Up in tumor', size = 11, color = 'a0', bold = True, ha = 'right')
    ax.set_xlabel('log_{2} fold change')
    ax.set_ylabel('-log_{10} adjusted p')
    return fig


def heatmap(palette):
    rng = np.random.default_rng(42)
    samples = [f'{t} {i + 1}' for t in ['Liver', 'Colon', 'Brain', 'Serum'] for i in range(4)]
    glycans = ['Man(a1-2)Man(a1-3)Man', 'Man(a1-3)[Man(a1-6)]Man', 'GlcNAc(b1-2)Man', 'Gal(b1-4)GlcNAc',
               'Gal(b1-3)GalNAc', 'Neu5Ac(a2-3)Gal', 'Neu5Ac(a2-6)Gal', 'Neu5Gc(a2-3)Gal', 'Fuc(a1-2)Gal',
               'Fuc(a1-3)GlcNAc', 'Fuc(a1-4)GlcNAc', 'Fuc(a1-6)GlcNAc', 'GalNAc(a1-3)Gal', 'Gal(a1-3)Gal',
               'GlcNAc(b1-3)Gal', 'GlcNAc(b1-6)GalNAc', 'GalNAc(b1-4)Gal', 'Neu5Ac(a2-8)Neu5Ac', 'Sulf-Gal(b1-3)GalNAc',
               'Gal(b1-4)Glc', 'GlcA(b1-3)Gal', 'Xyl(b1-2)Man', 'Man(a1-6)Man', 'Glc(a1-3)Man', 'GlcNAc(b1-4)GlcNAc',
               'Neu5Ac(a2-3)Gal(b1-3)GalNAc']
    mod = rng.integers(0, 4, 26)
    m = rng.normal(0, 0.55, (26, 16))
    for r, g in enumerate(mod):
        m[r, g * 4:g * 4 + 4] += rng.uniform(1.4, 2.5)
        m[r, ((g + 2) % 4) * 4:((g + 2) % 4) * 4 + 4] -= rng.uniform(0.6, 1.5)
    m = (m - m.mean(1, keepdims = True)) / m.std(1, keepdims = True)
    fig, _ = vc.clustermap(m, samples, glycans, palette = palette, title = 'Glycan abundance by tissue',
                           subtitle = 'z-scores, 26 glycans across 16 samples')
    return fig


def ridges(palette):
    x = np.linspace(0, 100, 500)
    names = ['Serum', 'Liver', 'Kidney', 'Lung', 'Colon', 'Spleen', 'Skin', 'Brain']
    params = [(18, 7, 44, 12, .7), (24, 8, 52, 9, .55), (30, 9, 58, 8, .6), (35, 8, 66, 10, .55),
              (42, 10, 72, 7, .6), (48, 8, 68, 11, .5), (55, 9, 80, 6, .6), (64, 7, 88, 5, .55)]
    ys = [w * np.exp(-.5 * ((x - a) / s) ** 2) / s + (1 - w) * np.exp(-.5 * ((x - b) / t) ** 2) / t for a, s, b, t,
          w in params]
    fig, ax = vc.subplots(palette = palette)
    fig.suptitle('Sialylation across tissues', 'Per-glycan distribution')
    ax.ridgeline(x, ys, names)
    ax.set_xlabel('Sialylated glycans (%)')
    return fig


def spectrum(palette):
    rng = np.random.default_rng(7)
    ann = [(222.10, 34, 'Y1'), (290.09, 100, 'B1'), (308.10, 46, 'C1'), (384.15, 28, 'Y2'), (452.14, 22, 'B2'),
           (470.15, 38, 'C2'), (513.19, 54, 'Y2β'), (675.24, 82, 'Y3'), (966.34, 63, '[M-H]-')]
    amz, ait = np.array([a[0] for a in ann]), np.array([a[1] for a in ann], float)
    lib = ait * rng.uniform(.8, 1.15, len(ann))
    fig, ax = vc.subplots(palette = palette)
    fig.suptitle('MS/MS of disialyl core 1', 'Negative mode, precursor m/z 966.34')
    ax.stem(rng.uniform(190, 1000, 150), rng.exponential(3.2, 150) + .6, color = 'a3', lw = 1.0, z = 2)
    ax.stem(rng.uniform(190, 1000, 28), -(rng.exponential(2.5, 28) + .5), color = 'b3', lw = 1.0, z = 2)
    ax.stem(amz, ait, color = 'a1', lw = 3.6)
    ax.stem(amz, -lib, color = 'b1', lw = 3.6)
    ax.axhline(0, color = 'a0', lw = 1.2, dash = None, z = 6)
    for (m, iv, name) in ann:
        ax.annotate(f'{name}  {m:.2f}', (m, iv), offset = (0, 5), size = 8.8, ha = 'left', rotation = 90.0, halo = 2.2)
    ax.set_xlim(190, 1005)
    ax.set_ylim(-118, 140)
    ax.set_yticks([-100, -50, 0, 50, 100], ['100', '50', '0', '50', '100'])
    ax.text(835, 124, 'Observed', size = 11, color = 'a0', bold = True, ha = 'center')
    ax.text(835, -112, 'Library', size = 11, color = 'b0', bold = True, ha = 'center')
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative intensity (%)')
    return fig


def stacked(palette):
    rng = np.random.default_rng(11)
    names = ['Colon', 'Liver', 'Lung', 'Serum', 'Kidney', 'Skin', 'Spleen', 'Brain']
    vals = rng.dirichlet([2.2, 3.5, 1.1, 2.4, 1.4], 8) * 100
    fig, ax = vc.subplots(palette = palette)
    fig.suptitle('Glycan classes across tissues', 'Share of each class, by tissue')
    ax.stacked_bar(names, vals, ['High-mannose', 'Complex', 'Hybrid', 'O-linked', 'Lipid-linked'],
                   colors = fig.pal.groups(3, 2))
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Share of glycans (%)')
    ax.legend('below')
    return fig


def embedding(palette):
    rng = np.random.default_rng(2024)
    specs = [(-4.3, .7, 2.4, .5, 2.4, 1500), (4.9, 2.6, 1.8, 2.8, 1.7, 1000), (-3.3, -4.0, 2.0, 1.2, 2.0, 1100),
             (1.2, -1.3, 1.9, 4.0, 2.4, 1200), (5.0, -3.5, 1.5, 5.4, 1.6, 700), (.5, 5.9, 1.5, 3.6, 1.5, 500)]
    names = ['Complex N', 'High-mannose', 'Core 1 O', 'Core 2 O', 'Ganglio', 'Lewis']
    fig, ax = vc.subplots(palette = palette)
    fig.suptitle('Structural embedding', '6,000 glycans by class')
    pick = np.random.default_rng(8)
    for (cx, cy, R, a0, span, n), name, c in zip(specs, names, fig.pal.groups(3, 3)):
        t = rng.uniform(a0, a0 + span, n)
        r = R + rng.normal(0, .23 * R ** .5, n)
        x, y = cx + r * np.cos(t) + rng.normal(0, .17, n), cy + r * np.sin(t) + rng.normal(0, .17, n)
        idx = pick.choice(n, min(650, n), replace = False)
        ax.scatter(x[idx], y[idx], s = 8, color = c)
        ax.text(np.median(x), np.percentile(y, 99.5) + .6, name, size = 11.5, bold = True, ha = 'center', halo = 3.0)
    ax.set_xlim(-7.8, 7.8)
    ax.set_ylim(-6.6, 7.6)
    ax.off()
    return fig


def trial(palette):
    rng = np.random.default_rng(99)
    t = np.linspace(0, 48, 160)
    curves = {'Control': .08 + .05 * t / 48, 'Low dose': .1 + .42 * (1 - np.exp(-t / 22)), 'High dose': .1
              + .82 * (1 - np.exp(-t / 11))}
    times = np.array([0, 4, 8, 12, 24, 36, 48.])
    fig, (a, b) = vc.subplots(1, 2, palette = palette, width_ratios = [2.9, 1.25])
    fig.suptitle('Response and benchmark', 'a, response over 48 h. b, score per method')
    a.set_title('a')
    b.set_title('b')
    for (name, y), c in zip(curves.items(), ['b1', 'a1', 'a0']):
        err = .025 + .05 * y
        a.fill_between(t, y - err, y + err, color = c)
        a.plot(t, y, color = c)
        a.scatter(times, np.interp(times, t, y), s = 36, color = c, z = 6)
        a.text(49.8, y[-1], name, size = 10.5)
    a.set_xlim(-1, 69)
    a.set_ylim(0, 1.08)
    a.set_xticks([0, 12, 24, 36, 48])
    a.set_xlabel('Time (h)')
    a.set_ylabel('Response')
    bars = np.array([.46, .61, .74, .86, .95])
    b.bar(list('ABCDE'), bars, color = 'a1')
    for i, v in enumerate(bars):
        b.scatter(np.full(6, i) + rng.uniform(-.18, .18, 6), v + rng.normal(0, .022, 6), s = 14, color = 'ink', z = 6)
    b.set_ylim(0, 1.06)
    b.set_xlabel('Method')
    b.set_ylabel('Score')
    return fig


def groups(palette):
    rng = np.random.default_rng(31)
    stages = ['Healthy', 'Adenoma', 'Carcinoma', 'Metastasis']
    sialyl = {s: rng.normal(m, sd, n) for s, m, sd, n in zip(stages, (2.1, 2.7, 3.6, 4.4), (.5, .7, .8, .9),
                                                                (28, 21, 34, 14))}
    fucose = {s: rng.gamma(k, .55, n) + 1 for s, k, n in zip(stages, (5.2, 4.4, 3.1, 2.6), (28, 21, 34, 14))}
    fig, (a, b) = vc.subplots(1, 2, figsize = (10, 6), palette = palette, sharey = True)
    fig.suptitle('Glycan motifs by stage', '97 colorectal samples')
    a.boxplot(sialyl)
    a.set_title('Sialyl Lewis x')
    a.set_ylabel('Relative abundance (%)')
    b.violinplot(fucose)
    b.set_title('Core fucose')
    return fig


FIGURES = {1: volcano, 2: heatmap, 3: ridges, 4: spectrum, 5: stacked, 6: embedding, 7: trial, 8: groups}


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'gallery')
    out.mkdir(exist_ok = True)
    for name in vc.palettes:
        for no, make in FIGURES.items():
            make(name).save(out / f'{name}{no}.png', dpi = 170)
