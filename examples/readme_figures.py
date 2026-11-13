"""Draws every image in the README into examples/readme/: python examples/readme_figures.py
Needs matplotlib and seaborn for the comparison, pillow for arranging the panels, glycowork for the glycan drawings and
the talk's data, and PowerPoint with pywin32 and OpenCV for the animation of the talk.
`python examples/readme_figures.py trailer` draws the 41-second trailer instead, built from viscose figures,
which needs matplotlib, seaborn, glycowork, and OpenCV, and Windows with PowerPoint and pywin32 for Morph and H.264.
`python examples/readme_figures.py linkedin` draws the 4:5 announcement video `linkedin.mp4` into Downloads,
which needs matplotlib, seaborn, glycowork, and OpenCV, and Windows for H.264."""
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import viscose as vc
from viscose.palettes import CVD
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


def seaborn_volcano(figsize = (8, 8), dpi = 200):
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
    fig, ax = plt.subplots(figsize = figsize)
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
    fig.savefig(buf, dpi = dpi, format = 'png')
    plt.close(fig)
    return Image.open(buf).convert('RGB')


def glycan_volcano(figsize = (7, 7), rise = 1.0, glyphs = True):
    """Below a `rise` of 1, each point is that far on its way up from the axis, each at its own time, for the trailer,
    which also leaves the glycans out with glyphs = False."""
    rng = np.random.default_rng(1729)
    x = np.clip(rng.normal(0, 1.3, 412), -4.6, 4.6)
    y = 0.72 * np.abs(x) ** 1.75 * rng.uniform(.8, 1.05, 412) + rng.exponential(0.42, 412)
    hits = [(0.6, 12.8, 'Neu5Ac(a2-6)Gal(b1-4)GlcNAc'), (-3.6, 11.2, 'Gal(a1-3)Gal(b1-4)GlcNAc'),
            (2.2, 9.9, 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'), (-3.2, 9.0, 'GlcNAc(b1-2)Man(a1-3)Man'),
            (3.0, 8.2, 'Fuc(a1-3)[Gal(b1-4)]GlcNAc'), (-3.8, 5.6, 'Fuc(a1-2)Gal(b1-3)GalNAc')]
    if rise < 1:
        late = np.random.default_rng(5).uniform(0, .4, 412 + len(hits))
        up = ease((rise - late) / .6)
        y = y * up[:412]
        hits = [(hx, hy * u, lab) for (hx, hy, lab), u in zip(hits, up[412:])]
    fig, ax = vc.subplots(figsize = figsize, palette = 'aura')
    fig.suptitle('Glycans as labels', 'SNFG drawings beside their points')
    sig = (np.abs(x) > 1) & (y > 2)
    ax.scatter(x[~sig], y[~sig], s = 22, color = 'a3', z = 2)
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s = 30, color = 'b1')
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s = 30, color = 'a1')
    for hx, hy, lab in hits:
        ax.scatter([hx], [hy], s = 120, color = 'b1' if hx < 0 else 'a1', z = 5)
        if glyphs:
            ax.annotate(lab, (hx, hy), glyph = 'glycan', spread = True)
    ax.axvline(-1)
    ax.axvline(1)
    ax.axhline(2)
    ax.set_xlim(-4.9, 5.4)
    ax.set_ylim(0, 14.2)
    ax.set_xlabel('log_{2} fold change')
    ax.set_ylabel('-log_{10} adjusted p')
    return fig


def glycan_bars(figsize = (7, 7)):
    from glycowork.glycan_data.loader import glycomics_data_loader
    table = glycomics_data_loader.human_colorectal_N_PMID26085185
    shares = table.iloc[:, 1:].astype(float)
    shares = (shares / shares.sum() * 100).mean(axis = 1)
    top = shares.sort_values().index[-8:]
    fig, ax = vc.subplots(figsize = figsize, palette = 'aura')
    fig.suptitle('Glycans as ticks', 'The most abundant N-glycans of colorectal tissue')
    ax.barh(list(table.iloc[top, 0]), shares[top].values)
    ax.tick_params('y', glyphs = 'glycan')
    ax.set_xlabel('Relative abundance (%)')
    return fig


def linked(highlight, palette = 'aura', figsize = (13, 10), mosaic = 'ab;cd', width_ratios = (1, 1.4)):
    """Colorectal O-glycomes by sample in four panels, with samples C3 and T3 and the largest changes in focus, or
    with what a list names in every panel: samples, groups, glycans, or, as numbers, the glycans of the largest changes
    from 0 on."""
    from glycowork.glycan_data.loader import glycomics_data_loader
    table = glycomics_data_loader.human_colorectal_O_PMC9254241.groupby('glycan', sort = False).sum()
    samples = [c for c in table.columns if c[0] in 'CT']
    shares = table[samples].to_numpy(float)
    shares = shares / shares.sum(axis = 0) * 100
    clr = np.log(shares + 0.1)
    clr -= clr.mean(axis = 0)
    tumor = np.array([s[0] == 'T' for s in samples])
    u, sv, _ = np.linalg.svd(clr.T - clr.T.mean(axis = 0), full_matrices = False)
    share = sv ** 2 / (sv ** 2).sum()
    top = np.argsort(-clr.var(axis = 1))[:12]
    glycans = list(table.index[top])
    z = (clr[top] - clr[top].mean(axis = 1, keepdims = True)) / clr[top].std(axis = 1, keepdims = True)
    change = clr[top][:, tumor].mean(axis = 1) - clr[top][:, ~tumor].mean(axis = 1)
    sialylated = shares[np.array(['Neu5Ac' in g for g in table.index])].sum(axis = 0)
    ranked = np.argsort(-np.abs(change))
    pick = ['C3', 'T3'] if highlight is True else None if not highlight else [
        glycans[ranked[k]] if isinstance(k, int) else k for k in highlight]
    fig, axes = vc.subplot_mosaic(mosaic, figsize = figsize, palette = palette, width_ratios = list(width_ratios))
    fig.suptitle('Colorectal O-glycome', '12 tumor and 12 control tissues')
    ax = axes['a']
    ax.scatter(*(u[:, :2] * sv[:2]).T, hue = np.where(tumor, 'Tumor', 'Control'), names = samples, s = 60,
               highlight = pick)
    ax.set(xlabel = 'PC1 (%d%%)' % round(100 * share[0]), ylabel = 'PC2 (%d%%)' % round(100 * share[1]))
    axes['b'].heatmap(z, xticklabels = samples, yticklabels = glycans, colorbar = True,
                      colorbar_label = 'z-score (CLR)', highlight = pick)
    axes['b'].tick_params('y', glyphs = 'glycan')
    ax = axes['c']
    ax.boxplot({'Control': sialylated[~tumor], 'Tumor': sialylated[tumor]},
               names = [list(np.array(samples)[~tumor]), list(np.array(samples)[tumor])], highlight = pick)
    ax.set_ylabel('Sialylated (%)')
    ax = axes['d']
    ax.barh(glycans, change, color = ['a1' if v > 0 else 'b1' for v in change],
            highlight = [glycans[i] for i in ranked[:3]] if highlight is True else pick)
    ax.tick_params('y', glyphs = 'glycan')
    ax.axvline(0)
    ax.set_xlabel('Tumor - control (CLR)')
    return fig


def markers(figsize = (8, 6)):
    rng = np.random.default_rng(3)
    n = 150
    stage = np.repeat(['Healthy', 'Early', 'Late'], n // 3)
    shift = np.select([stage == 'Healthy', stage == 'Early'], [-1.6, 0.2], 1.8)
    x, y = rng.normal(shift, 0.8), rng.normal(shift ** 2 / 2 - 0.8, 0.7)
    fig, ax = vc.subplots(figsize = figsize, palette = 'aura')
    fig.suptitle('Glycomes by disease stage', '150 patients')
    ax.scatter(x, y, hue = stage, style = rng.choice(['Serum', 'Tissue'], n), size = rng.random(n) * 10,
               hue_label = 'Stage', style_label = 'Source', size_label = 'Abundance')
    ax.text(0.02, 0.98, 'F = 4.2\np = 0.003', axes_fraction = True, va = 'top')
    ax.set(xlabel = 'PC1 (31%)', ylabel = 'PC2 (12%)')
    return fig


def labels(palette, figsize = (15, 4.6)):
    """Every series named where it is: at the ends of lines, beside groups of points, and right of stacked bars."""
    rng = np.random.default_rng(11)
    fig, axes = vc.subplots(1, 3, figsize = figsize, palette = palette)
    t = np.linspace(0, 48, 49)
    for name, rate, top in [('Wild type', 0.11, 1.0), ('ΔFUT8', 0.09, 0.94), ('ST6GAL1 KO', 0.06, 0.72),
                            ('MGAT5 KO', 0.035, 0.55)]:
        axes[0].plot(t, top * (1 - np.exp(-rate * t)) + rng.normal(0, 0.006, t.size), label = name)
    axes[0].axhline(0.5, label = 'Half-maximal')
    axes[0].set(xlabel = 'Time (h)', ylabel = 'Sialylation', title = 'Cell lines')
    axes[0].legend()
    centers = {'Plasma': (-3, 1), 'Liver': (2, 4), 'Colon': (3, -2), 'Brain': (-1, -4)}
    xy = np.concatenate([rng.normal(c, 0.9, (90, 2)) for c in centers.values()])
    axes[1].scatter(*xy.T, hue = np.repeat(list(centers), 90), s = 16)
    axes[1].set(xlabel = 'UMAP 1', ylabel = 'UMAP 2', title = 'Tissues')
    axes[2].stacked_bar(['Healthy', 'Adenoma', 'Carcinoma'], [[38, 30, 22, 10], [30, 31, 25, 14], [21, 30, 31, 18]],
                        labels = ['High-mannose', 'Complex', 'Hybrid', 'Paucimannose'], colors = fig.pal.groups(3, 1))
    axes[2].set(ylabel = 'Share (%)', title = 'N-glycan classes')
    axes[2].legend()
    return fig


def crowd(density, figsize = (6, 5), share = None, palette = 'poster'):
    """100,000 cells from two populations, as points alone or with the field that takes over where they pile up, or
    the first `share` of them in a random order on fixed axes, for the trailer."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (50000, 2)) @ [[1, .6], [0, .8]]
    b = rng.normal(0, .5, (50000, 2)) + [2.5, -1.5]
    fig, ax = vc.subplots(figsize = figsize, palette = palette)
    fig.suptitle('100,000 cells', 'Where points pile up, their density' if density else 'Every point')
    cells = np.vstack([a, b])
    if share is not None:
        cells = cells[np.random.default_rng(4).permutation(len(cells))[:max(1, round(share * len(cells)))]]
        ax.set_xlim(-4.74, 4.77)
        ax.set_ylim(-4.56, 5.16)
    ax.scatter(*cells.T, s = 12, density = density)
    ax.set(xlabel = 'UMAP 1', ylabel = 'UMAP 2')
    return fig


def milk():
    """Human milk oligosaccharides by day of lactation, as shares of all measured, with the seven most abundant and
    how many mothers each day has."""
    from glycowork.glycan_data.loader import glycomics_data_loader
    table = glycomics_data_loader.time_series_HMO_PMID22649065
    day = table['ID'].str.extract(r'_d(\d+)_')[0].astype(int).values
    shares = table.iloc[:, 1:].astype(float)
    by_day = (shares.div(shares.sum(axis = 1), axis = 0) * 100).groupby(day).mean()
    return by_day, list(by_day.mean().sort_values().index[-7:]), dict(zip(*np.unique(day, return_counts = True)))


def talk():
    """Real data as slides: human milk oligosaccharides on days 1, 4, and 14 of lactation, then the most abundant
    colorectal N-glycans as a heatmap in sample order and clustered."""
    from glycowork.glycan_data.loader import glycomics_data_loader
    by_day, top, mothers = milk()
    figs = []
    for d in (1, 4, 14):
        fig, ax = vc.subplots(figsize = (13.333, 7.5), palette = 'aura')
        fig.suptitle('Milk oligosaccharides over lactation', f'Day {d}, mean of {mothers[d]} mothers')
        ax.barh(top, by_day.loc[d, top].values)
        ax.tick_params('y', glyphs = 'glycan')
        ax.set_xlim(0, 40)
        ax.set_xlabel('Share of measured HMOs (%)')
        figs.append(fig)
    table = glycomics_data_loader.human_colorectal_N_PMID26085185
    m = table.iloc[:, 1:].astype(float)
    m = m / m.sum() * 100
    z = m.iloc[np.argsort(m.mean(axis = 1).values)[-24:]]
    z = z.sub(z.mean(axis = 1), axis = 0).div(z.std(axis = 1), axis = 0)
    z = z.set_axis([f'N{i + 1}' for i in range(len(z))]).set_axis([c.replace('Avg_RA_', '') for c in z.columns],
                                                                     axis = 1)
    heading = ('Colorectal N-glycans', 'z-scores of the 24 most abundant, in tumor (T) and normal (N) tissue')
    fig, ax = vc.subplots(figsize = (13.333, 7.5), palette = 'aura')
    fig.suptitle(*heading)
    ax.heatmap(z.values, xticklabels = list(z.columns), yticklabels = list(z.index), colorbar = True)
    figs.append(fig)
    fig, _ = vc.clustermap(z, figsize = (13.333, 7.5), palette = 'aura')
    figs.append(fig.suptitle(*heading))
    return figs


def powerpoint_frames(deck, height, fps):
    """A deck as PowerPoint plays it, exported by PowerPoint itself as video, as RGB frames, each slide held for a
    second after its transition, or None without Windows, PowerPoint, pywin32, and OpenCV."""
    import time
    try:
        import cv2
        import win32com.client
    except ImportError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        video = str(Path(tmp) / 'deck.mp4')
        app = win32com.client.DispatchEx('PowerPoint.Application')
        pres = app.Presentations.Open(str(Path(deck).resolve()), True, False, False)
        try:
            pres.CreateVideo(video, False, 1, height, fps, 90)
            while pres.CreateVideoStatus in (1, 2):
                time.sleep(1)
        finally:
            pres.Close()
            app.Quit()
        frames, cap = [], cv2.VideoCapture(video)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
    return frames


def morph_video(deck, name, width = 960, fps = 15):
    """A deck as PowerPoint plays it, turned into an animated WebP, so that what the README shows is Morph's own
    work. Skipped without Windows, PowerPoint, pywin32, and OpenCV."""
    frames = powerpoint_frames(deck, 720, fps)
    if frames is None:
        print('Skipping', name, 'which needs PowerPoint, pywin32, and OpenCV')
        return
    frames = [Image.fromarray(f).resize((width, round(width * f.shape[0] / f.shape[1])), Image.LANCZOS) for f in frames]
    # A slide held still is one frame shown for as long, which the video repeats at every tick
    kept, times = [], []
    for f in frames:
        if kept and np.abs(np.asarray(f, np.int16) - np.asarray(kept[-1], np.int16)).mean() < 0.4:
            times[-1] += round(1000 / fps)
        else:
            kept.append(f)
            times.append(round(1000 / fps))
    # Animated WebP keeps full color, which a GIF's 256 would not: SNFG's small yellow circles lost theirs
    kept[0].save(OUT / name, save_all = True, append_images = kept[1:], duration = times, loop = 0, quality = 82,
                 method = 6)


# The trailer: 40.5 seconds at 30 frames per second, each frame viscose figures of 12.8 by 7.2 inches drawn at 150 dpi,
# so 1920 by 1080 pixels, blended by fades, diagonal wipes, and an iris where a dark page meets a light one; only the
# matplotlib figure it starts from and PowerPoint's Morph of viscose's own slides come from elsewhere
FPS, PAGE, SIZE = 30, (12.8, 7.2), (1920, 1080)
TAGLINE = 'Scientific plots that are pieces of art'
# Settled points keep shimmering, by this much in inches
SHIMMER = 0.016
# Captions sit in a band of this many pixels at the top, as a figure's title does; the wall's six figures fill the rest
TOP = 175
WALL, TILE = ['volcano', 'heatmap', 'ridges', 'spectrum', 'embedding', 'groups'], (610, 430)
# Each feature card shows for this many seconds
CARD = 2.0


def ease(t):
    """Smootherstep from 0 to 1, flat at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


@lru_cache(None)
def letters(n = 24000):
    """Points spread over the ink of the word viscose set large, in page inches, and the letter each lies on."""
    # Dark ink on white, also while dark mode is on
    previous = vc.dark_mode(False)
    fig = vc.Figure(*PAGE, palette = 'aura')
    fig.text(0.5, 0.42, 'viscose', size = 190, bold = True, ha = 'center')
    ink = fig.to_rgba(40, background = 'white')[..., :3].mean(axis = 2) < 128
    vc.dark_mode(*previous)
    rows, cols = np.nonzero(ink)
    rng = np.random.default_rng(1)
    k = rng.integers(0, rows.size, n)
    x = (cols[k] + rng.random(n)) / ink.shape[1] * PAGE[0]
    inked = np.unique(cols)
    letter = np.searchsorted(inked[1:][np.diff(inked) > 1], cols[k], side = 'right')
    return x, PAGE[1] - (rows[k] + rng.random(n)) / ink.shape[0] * PAGE[1], letter


def gathering(palette, t, caption, span, colors):
    """Points that swirl in from beyond the page over `span` seconds and settle into the word, where the field of their
    density takes over from them as they pile up. `colors` take turns letter by letter."""
    tx, ty, letter = letters()
    n = tx.size
    rng = np.random.default_rng(7)
    angle, r = rng.uniform(0, 2 * np.pi, n), 2.5 + 9 * rng.random(n) ** .6
    cx, cy = PAGE[0] / 2, PAGE[1] / 2
    e = ease((t / span - rng.uniform(0, .35, n)) / .65)
    px, py = r * np.cos(angle) * (1 - e) + (tx - cx) * e, r * np.sin(angle) * (1 - e) + (ty - cy) * e
    spin, phase = 2.4 * (1 - e) ** 2, rng.uniform(0, 2 * np.pi, n)
    x = cx + px * np.cos(spin) - py * np.sin(spin) + SHIMMER * np.sin(3.1 * t + phase)
    y = cy + px * np.sin(spin) + py * np.cos(spin) + SHIMMER * np.cos(2.3 * t + 1.7 * phase)
    fig = vc.Figure(*PAGE, palette = palette)
    ax = fig.add_axes((0, 0, 1, 1))
    for k, c in enumerate(colors):
        ax.scatter(x[letter % len(colors) == k], y[letter % len(colors) == k], s = 6, color = c)
    ax.set_xlim(0, PAGE[0])
    ax.set_ylim(0, PAGE[1])
    ax.off()
    if caption:
        fig.text(0.5, 0.2, caption, size = 24, ha = 'center', color = 'muted')
    return fig


def render(fig, size = SIZE, top = False):
    """A figure's pixels at the width given, and a page that grew for its labels fitted back in, on its own paper,
    centered, or at the top with `top`, so that titles in a row line up."""
    img = fig.to_rgba(size[0] * 72 / fig.width)[..., :3]
    if img.shape[:2] != size[::-1]:
        small = Image.fromarray(img)
        small.thumbnail(size, Image.LANCZOS)
        frame = Image.new('RGB', size, tuple(int(v) for v in img[0, 0]))
        frame.paste(small, ((size[0] - small.width) // 2, 0 if top else (size[1] - small.height) // 2))
        img = np.asarray(frame)
    return np.array(img)


def header(palette, title, subtitle):
    """An empty page with a caption as its title, for frames put together from several figures."""
    return render(vc.Figure(*PAGE, palette = palette).suptitle(title, subtitle))


@lru_cache(64)
def wall_tile(kind, palette, t):
    """One of the wall's figures, the ridges flowing and the embedding turning `t` seconds in."""
    size = (7.2, 7.2 * TILE[1] / TILE[0])
    fig = gallery.ridges(palette, t, size) if kind == 'ridges' else gallery.embedding(
        palette, .35 * np.sin(.9 * t), size) if kind == 'embedding' else getattr(gallery, kind)(palette, size)
    return render(fig, TILE)


def wall(palette, t):
    """Six kinds of figure in one palette, under the word that sets it."""
    dark = palette.endswith('-dark')
    img = header(palette, 'Every palette, also by night' if dark else 'One word changes everything',
                 f"palette = '{palette}'")
    for i, kind in enumerate(WALL):
        x, y = 20 + i % 3 * (TILE[0] + 25), TOP + i // 3 * (TILE[1] + 25)
        img[y:y + TILE[1], x:x + TILE[0]] = wall_tile(kind, palette, t if kind in ('ridges', 'embedding') else 0.0)
    return img


def feature_card(i):
    """One of the rapid-fire features, by its place in CARDS."""
    make, title, subtitle = CARDS[i]
    if make == 'themes':
        # One SVG holds the light page and its dark twin; here they meet on a diagonal
        return diagonal(*(render(gallery.trial(p, PAGE).suptitle(title, subtitle)) for p in ('poster', 'poster-dark')))
    if make == 'legible':
        # matplotlib's blue, orange, green, and red, as a protanope sees them: drawn as given, and as viscose draws them
        rng = np.random.default_rng(11)
        centers = {'Plasma': (-3, 1), 'Liver': (2, 4), 'Colon': (3, -2), 'Brain': (-1, -4)}
        xy = np.concatenate([rng.normal(c, 0.9, (90, 2)) for c in centers.values()])
        img = header('poster', title, subtitle)
        for k, legible in enumerate((False, True)):
            fig, ax = vc.subplots(figsize = (7.2, 7.2 * 880 / 930), palette = 'poster')
            fig.suptitle('As viscose draws them' if legible else 'Colors as given', 'Seen with protanopia')
            fig.legible = legible
            ax.scatter(*xy.T, hue = np.repeat(list(centers), 90), colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'],
                       s = 30)
            px = render(fig, (930, 880)) / 255.0
            lin = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4) @ CVD['protan'].T
            lin = np.clip(lin, 0.0, 1.0)
            img[TOP:TOP + 880, 20 + 950 * k:950 + 950 * k] = np.round(255 * np.where(
                lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055))
        return img
    return render(make().suptitle(title, subtitle))


def diagonal(left, right):
    """Two frames of one size meeting on a diagonal, the first above and to the left."""
    h, w = left.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / h
    return np.where(((xx + .45 * yy) / (w / h + .45) > .5)[..., None], right, left)


def recolored(tmp):
    """One of viscose's slides as PowerPoint renders it, in the palette's own theme and then with two of Office's pairs
    of accents 1 and 2, as its Design tab sets them, keyed for the PowerPoint scene."""
    import win32com.client
    fig = gallery.volcano('aura', PAGE).suptitle('Yours to edit in PowerPoint',
                                                 "Every color a theme color: Office's Design tab recolors it all")
    vc.deck([fig], Path(tmp) / 'theme.pptx', morph = False)
    app = win32com.client.DispatchEx('PowerPoint.Application')
    pres = app.Presentations.Open(str((Path(tmp) / 'theme.pptx').resolve()), True, False, False)
    shots = {}
    try:
        # Theme colors are 0xBBGGRR, and accents 1 and 2 are the scheme's fifth and sixth: as drawn, then Office's
        # blue and orange, then its green and gold
        for i, accents in enumerate(((), (0xC47244, 0x317DED), (0x47AD70, 0x00C0FF))):
            for k, bgr in zip((5, 6), accents):
                pres.SlideMaster.Theme.ThemeColorScheme(k).RGB = bgr
            shot = str(Path(tmp) / f'theme{i}.png')
            pres.Slides(1).Export(shot, 'PNG', *SIZE)
            shots[('theme', i)] = np.asarray(Image.open(shot).convert('RGB').resize(SIZE, Image.LANCZOS))
    finally:
        pres.Close()
        app.Quit()
    return shots


def still(key):
    """One state of the trailer as RGB pixels, from its key: the kind of scene and what sets it."""
    kind, *args = key
    if kind == 'gather':
        return render(gathering(*args))
    if kind == 'box' and args[0] == 'seaborn':
        img = header('aura', 'Out of the box', 'matplotlib and seaborn, by default')
        img[180:] = np.asarray(seaborn_volcano((12.8, 6.0), 150))
        return img
    if kind == 'box':
        return render(gallery.volcano('aura', PAGE).suptitle('Out of the box', 'viscose, by default'))
    if kind == 'wall':
        return wall(*args)
    if kind == 'erupt':
        return render(glycan_volcano(PAGE, *args).suptitle('Glycans as glyphs',
                                                           'Real SNFG drawings find open room beside their points'))
    if kind == 'focus':
        return render(linked(list(args[0]) if args[0] else None, 'aura', (16, 9), 'abd;cbd', (1, 1.5, 1.15)).suptitle(
            'Focus in shades', 'Highlight anything, and every panel follows, live in interactive SVG'))
    if kind == 'remember':
        # Saved in Lumen, and brought back from the file alone in Aura's dark version
        if not args[0]:
            return render(gallery.volcano('lumen', PAGE).suptitle('Figures that remember', "fig.save('volcano.pdf')"))
        with tempfile.TemporaryDirectory() as tmp:
            gallery.volcano('lumen', PAGE).save(Path(tmp) / 'volcano.pdf')
            back = vc.open(Path(tmp) / 'volcano.pdf', palette = 'aura-dark')
        return render(back.suptitle('Figures that remember', "vc.open('volcano.pdf', palette = 'aura-dark')"))
    if kind == 'crowd':
        return render(crowd(True, PAGE, args[0]).suptitle(
            f'{round(args[0] * 100000):,} points', "Their density takes over where they pile up. As PDF in 0.13 s, "
            "where matplotlib takes 1.64 s"))
    return feature_card(args[0])


def word_scene(palette, colors, span, caption, at):
    """The word gathering, with its caption fading in `at` seconds in."""
    def scene(u, s):
        plain = ('gather', palette, round(u, 4), '', span, colors)
        captioned, share = (*plain[:3], caption, span, colors), ease((u - at) / .6)
        return [(plain, None)] if share <= 0 else [(captioned, None)] if share >= 1 else [
            (plain, None), (captioned, ('fade', float(share)))]
    return scene


def wall_scene(u, s):
    """The wall through every palette, each wiping in over the last, and then into the night."""
    seq = [*vc.palettes, 'aura-dark']
    k = int(np.clip(u // 1.1, 0, 4))
    now, share = ('wall', seq[k], round(s, 4)), (u - 1.1 * k) / (.6 if k == 4 else .35)
    return [(('wall', seq[k - 1], round(s, 4)), None), (now, ('iris' if k == 4 else 'wipe', float(ease(share))))] if (
        k and share < 1) else [(now, None)]


def erupt_scene(u, s):
    """A volcano plot whose points rise from the axis, and then the glycans of its hits."""
    share = ease((u - 2.1) / .5)
    plain, drawn = ('erupt', round(min(u / 2.0, 1.0), 4), False), ('erupt', 1.0, True)
    return [(plain, None)] if share <= 0 else [(drawn, None)] if share >= 1 else [
        (plain, None), (drawn, ('fade', float(share)))]


def crowd_scene(u, s):
    """Points pouring in, a hundred to a hundred thousand, until their density forms bands."""
    return [(('crowd', round(float(.001 + .999 * ease(u / 2.1)), 5)), None)]


def morph_scene(u, s):
    """PowerPoint's own video of the deck viscose wrote, frame by frame."""
    return [(('ppt', round(u * FPS)), None)]


def steps(keys, at, how = 'fade', span = .25):
    """A scene that shows `keys` in turn, each coming in over the last at its time in `at`, by `how` over `span`
    seconds."""
    def scene(u, s):
        i = sum(u >= t for t in at)
        share = ease((u - at[i - 1]) / span) if i else 1.0
        return [(keys[i], None)] if share >= 1 else [(keys[i - 1], None), (keys[i], (how, float(share)))]
    return scene


def cards_scene(u, s):
    """Feature after feature, each fading in over the last."""
    i = min(int(u // CARD), len(CARDS) - 1)
    share = ease((u - i * CARD) / .25)
    return [(('card', i), None)] if not i or share >= 1 else [(('card', i - 1), None),
                                                              (('card', i), ('fade', float(share)))]


CARDS = [(lambda: labels('lumen', PAGE), 'Labels where the data are',
          'Every series named in its own color, with no legend to decode'),
         ('legible', 'Legible for every reader',
          'Colors that would merge for color-blind readers move apart by themselves'),
         ('themes', 'One SVG, both themes', "It follows the reader's light or dark mode")]
# (start, end, scene, entry) in seconds, where each scene comes in over the last for FADE seconds, by a fade, or by an
# iris where a dark page meets a light one, which a fade would pass through gray; so each scene ends FADE seconds after
# the next one starts
FADE = 0.4
SCRIPT = [(0.0, 5.0, word_scene('aura-dark', ('a1', ), 3.0, TAGLINE, 3.2), None),
          (4.6, 8.1, steps([('box', 'seaborn'), ('box', 'viscose')], [1.5], 'wipe', .7), 'iris'),
          (7.7, 13.7, wall_scene, 'fade'), (13.3, 17.3, erupt_scene, 'iris'), (16.9, 19.9, crowd_scene, 'fade'),
          (19.5, 22.5, morph_scene, 'fade'),
          (22.1, 24.9, steps([('theme', 0), ('theme', 1), ('theme', 2)], [.8, 1.7], span = .35), 'fade'),
          (24.5, 30.9, cards_scene, 'fade'),
          (30.5, 33.5, steps([('focus', None), ('focus', ('T3', )), ('focus', ('C3', )), ('focus', (0, )),
                              ('focus', ('Tumor', ))], [.5, 1.05, 1.6, 2.15], span = .2), 'fade'),
          (33.1, 36.1, steps([('remember', False), ('remember', True)], [1.0], 'wipe', .6), 'fade'),
          (35.7, 40.5, word_scene('poster', ('a1', 'b1'), 2.0, 'pip install viscose', 2.3), 'fade')]
# Morph starts this many seconds into its scene
MORPH_LEAD = 0.5


def trailer_frames():
    """Each frame of the trailer as layers [(key, how)] blended in order, `how` None for the first and otherwise
    ('fade', share), ('wipe', share), or ('iris', share)."""
    frames = []
    for i in range(round(SCRIPT[-1][1] * FPS)):
        s, layers = i / FPS, []
        for start, end, scene, entry in SCRIPT:
            if start <= s < end:
                own = scene(s - start, s)
                if layers:
                    own[0] = (own[0][0], (entry, float(ease((s - start) / FADE))))
                layers += own
        frames.append(layers)
    return frames


def compose(layers, images):
    """One frame from its layers' pixels."""
    h, w = SIZE[1], SIZE[0]
    yy, xx = np.mgrid[0:h, 0:w] / h
    out = None
    for key, how in layers:
        img = images[key].astype(np.float32)
        if how is None:
            out = img
            continue
        kind, share = how
        if kind == 'fade':
            m = np.float32(share)
        else:
            # A soft edge, a fifth of the frame's height, sweeping diagonally or opening as a circle from the middle
            d = (xx + .45 * yy) / (w / h + .45) if kind == 'wipe' else np.hypot(xx - w / h / 2, yy - .5) / 1.02
            m = np.clip((share * 1.2 - d) / .2, 0, 1).astype(np.float32)[..., None]
        out = out * (1 - m) + img * m
    return np.round(out).astype(np.uint8)


def morphed(tmp):
    """PowerPoint's video of the README's colorectal N-glycans, a heatmap in sample order and then clustered, as
    viscose writes them for Morph, keyed by frame from MORPH_LEAD seconds before the cells start to fly."""
    figs = [f.suptitle('Clustered, then morphed', 'viscose writes PowerPoint, and Morph flies every cell to its place')
            for f in talk()[-2:]]
    vc.deck(figs, Path(tmp) / 'morph.pptx')
    video = powerpoint_frames(Path(tmp) / 'morph.pptx', SIZE[1], FPS)
    video = [f if f.shape[:2] == SIZE[::-1] else np.asarray(Image.fromarray(f).resize(SIZE, Image.LANCZOS))
             for f in video]
    moving = [k for k in range(5, len(video)) if np.abs(video[k].astype(np.int16) - video[0]).mean() > 0.5]
    first = moving[0] - round(MORPH_LEAD * FPS)
    return {('ppt', k): video[min(max(first + k, 0), len(video) - 1)] for k in range(len(video))}


def trailer(name = 'trailer.mp4', bitrate = 8_000_000):
    """The trailer, drawn on all but two cores, written by Media Foundation as H.264, and then encoded again at
    `bitrate` by Windows' own transcoder, as OpenCV's writer has a fixed and very high one."""
    import cv2
    frames = trailer_frames()
    keys, last = {}, {}
    for i, layers in enumerate(frames):
        for key, _ in layers:
            keys.setdefault(key, i)
            last[key] = i
    print(len(frames), 'frames from', len(keys), 'stills')
    workers = max(1, (os.cpu_count() or 4) - 2)
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(workers) as pool:
        # PowerPoint's own pictures are made here, before the workers draw the rest
        images = morphed(tmp) | recolored(tmp)
        raw = str(Path(tmp) / 'raw.mp4')
        writer = cv2.VideoWriter(raw, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), FPS, SIZE)
        order, futures = iter(k for k in keys if k not in images), {}
        for i, layers in enumerate(frames):
            # Stills are drawn ahead in the order frames first need them, and dropped once no frame needs them
            while len(futures) < 3 * workers and (key := next(order, None)) is not None:
                futures[key] = pool.submit(still, key)
            for key, _ in layers:
                if key not in images:
                    images[key] = futures.pop(key).result()
            writer.write(np.ascontiguousarray(compose(layers, images)[..., ::-1]))
            for key, _ in layers:
                if last[key] == i:
                    images.pop(key, None)
            if i % 60 == 0:
                print('frame', i, flush = True)
        writer.release()
        subprocess.run(['powershell', '-NoProfile', '-Command', TRANSCODE.format(src = raw, out = OUT.resolve() / name,
                                                                                  bitrate = bitrate, width = SIZE[0],
                                                                                  height = SIZE[1])], check = True)
    print(name, (OUT / name).stat().st_size // 1024, 'kB')


# The announcement for LinkedIn: a 4:5 video of 9 by 11.25 inches at 120 dpi, so 1080 by 1350 pixels, put together from
# viscose figures on Aura's dark page. Each feature is a tile: a caption of FEATURE_HEAD inches over a figure
# FEATURE_HIGH inches high, a third or half of the page wide, all drawn at half size, as are the palettes' figures of
# LOOK inches. A light page in a tile sits INSET pixels inside it, as a page on the dark one
POSTER, POSTER_DPI, FEATURE_HEAD, FEATURE_HIGH, LOOK, INSET = (9.0, 11.25), 120, 0.95, 3.9, (4.5, 4.0), 20
POSTER_EXTRAS = ('PowerPoint with Morph  ·  interactive SVG  ·  one SVG for light and dark',
                 'PDF/X-4 for print  ·  figures that remember their data')
# The ridges flow with a period of 2 pi / 2.2 seconds, so the loop is four of them and they never jump
LOOP = 8 * np.pi / 2.2
# When each tile starts to move, in seconds: the glycans' points rise, the cells pour in, matplotlib's default wipes
# into viscose's, the Lumen file reopens in Aura by night, and the SVG turns dark, light, and dark again; each wipe
# takes WIPE seconds
RISE, POUR, DEFAULT, REOPEN, THEMES, WIPE = 0.6, 1.2, 3.6, 4.6, (5.6, 7.0, 8.2), 0.8


def inset(img):
    """A tile's figure INSET pixels inside Aura's dark page."""
    out = np.empty((img.shape[0] + 2 * INSET, img.shape[1] + 2 * INSET, 3), np.uint8)
    out[:] = np.asarray(Image.new('RGB', (1, 1), vc.get_palette('aura-dark').paper))[0, 0]
    out[INSET:-INSET, INSET:-INSET] = img
    return out


@lru_cache(None)
def poster_piece(key):
    """One piece of the announcement as pixels, from its key, drawn once per process: light pages with dark mode off,
    everything else on Aura's dark page."""
    kind, *args = key
    half = lambda inches: tuple(round(v * POSTER_DPI / 2) for v in inches)
    room = lambda inches: tuple(v - 2 * INSET for v in half(inches))
    third, wide = (2 * POSTER[0] / 3, FEATURE_HIGH), (POSTER[0], FEATURE_HIGH)
    paper = vc.get_palette('aura-dark').paper
    previous = vc.dark_mode(*((False, None) if kind in ('default', 'saved', 'light') else (True, paper)))
    try:
        if kind == 'word':
            return render(gathering('aura', args[0], '', 3.0, ('a1', 'b1')),
                          (round(POSTER[0] * POSTER_DPI), round(POSTER[0] * POSTER_DPI * PAGE[1] / PAGE[0])))
        if kind == 'tagline':
            return render(vc.Figure(POSTER[0], 0.5, palette = 'aura').text(0.5, 0.3, TAGLINE, size = 24, ha = 'center',
                                                                           color = 'muted'),
                          (round(POSTER[0] * POSTER_DPI), round(0.45 * POSTER_DPI)))
        if kind == 'caption':
            return render(vc.Figure(args[0], FEATURE_HEAD, palette = 'aura').suptitle(*args[1:]),
                          half((args[0], FEATURE_HEAD)))
        if kind == 'glycans':
            return render(glycan_volcano(third, *args).suptitle(''), half(third), True)
        if kind == 'crowd':
            return render(crowd(True, third, args[0], 'aura').suptitle(''), half(third), True)
        if kind == 'default':
            return inset(np.asarray(seaborn_volcano(tuple(v - 4 * INSET / POSTER_DPI for v in third),
                                                    POSTER_DPI / 2)))
        if kind == 'box':
            fig = gallery.volcano('aura', third).suptitle('')
            return inset(render(fig.text(0.97, 0.04, 'viscose', size = 22, bold = True, ha = 'right', color = 'a1'),
                                room(third), True))
        if kind == 'saved':
            return inset(render(gallery.trial('lumen', wide).suptitle(''), room(wide), True))
        if kind == 'back':
            # Saved in Lumen on a light page, and brought back from the file alone
            vc.dark_mode(False)
            with tempfile.TemporaryDirectory() as tmp:
                gallery.trial('lumen', wide).suptitle('').save(Path(tmp) / 'trial.pdf')
                vc.dark_mode(True, paper)
                back = vc.open(Path(tmp) / 'trial.pdf', palette = 'aura-dark')
            return inset(render(back, room(wide), True))
        if kind in ('light', 'dark'):
            return inset(render(labels('poster', wide), room(wide), True))
        if kind == 'look':
            return render(gallery.ridges(args[0], args[1], LOOK).suptitle(args[0].capitalize()), half(LOOK))
        footer = vc.Figure(POSTER[0], 1.0, palette = 'aura')
        footer.text(0.5, 0.6, 'pip install viscose', size = 26, bold = True, ha = 'center', color = 'a1')
        for k, extras in enumerate(POSTER_EXTRAS):
            footer.text(0.5, 0.3 - 0.24 * k, extras, size = 12.5, ha = 'center', color = 'muted')
        return render(footer, (round(POSTER[0] * POSTER_DPI), POSTER_DPI))
    finally:
        vc.dark_mode(*previous)


def swept(a, b, share):
    """`b` wiping in over `a` along a soft diagonal edge, `share` of the way, as the trailer wipes."""
    if share <= 0 or share >= 1:
        return b if share >= 1 else a
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / h
    m = np.clip((share * 1.2 - (xx + .45 * yy) / (w / h + .45)) / .2, 0, 1)[..., None]
    return np.round(a * (1 - m) + b * m).astype(np.uint8)


def poster_frame(s):
    """The announcement `s` seconds in: the word gathering, its tagline fading in, and every tile on its own time,
    from top to bottom with the room left spread evenly between them."""
    piece = poster_piece
    settled = piece(('word', 3.0))
    paper = settled[0, 0]
    inked = np.nonzero(np.abs(settled.astype(int) - paper).max(axis = (1, 2)) > 24)[0]
    word = piece(('word', round(s, 4)))[max(inked[0] - 10, 0):inked[-1] + 10]
    tagline = swept(np.full_like(piece(('tagline', )), paper), piece(('tagline', )), ease((s - 2.4) / .9))
    u = s - RISE
    erupted = piece(('glycans', round(float(np.clip(u / 2.0, 0, 1)), 4), False))
    glycans = swept(erupted, piece(('glycans', 1.0, True)), ease((u - 2.1) / .5))
    dark = ease((s - THEMES[0]) / WIPE) - ease((s - THEMES[1]) / WIPE) + ease((s - THEMES[2]) / WIPE)
    third, wide = 2 * POSTER[0] / 3, POSTER[0]
    features = [('Glycans as glyphs', 'Real SNFG drawings find their own room', glycans, third),
                ('100,000 points', 'As PDF in 0.13 s, where matplotlib takes 1.64 s',
                 piece(('crowd', round(float(.001 + .999 * ease((s - POUR) / 2.1)), 4))), third),
                ('Beautiful by default', "matplotlib's and seaborn's default, then viscose's",
                 swept(piece(('default', )), piece(('box', )), ease((s - DEFAULT) / WIPE)), third),
                ('Figures that remember',
                 "fig.save('trial.pdf') in Lumen, then vc.open('trial.pdf', palette = 'aura-dark')",
                 swept(piece(('saved', )), piece(('back', )), ease((s - REOPEN) / WIPE)), wide),
                ('One SVG, both themes', "It follows the reader's light or dark mode",
                 swept(piece(('light', )), piece(('dark', )), dark), wide)]
    tiles = [np.vstack([piece(('caption', width, title, subtitle)), shown]) for title, subtitle, shown, width in
             features]
    looks = [piece(('look', p, round(s % LOOP, 4))) for p in vc.palettes]
    rows = [word, tagline, np.hstack(tiles[:3]), np.hstack(tiles[3:]),
            piece(('caption', POSTER[0] * 2, 'One word, four palettes',
                   "palette = 'lumen', 'couture', 'poster', or 'aura'")), np.hstack(looks), piece(('footer', ))]
    w, h = round(POSTER[0] * POSTER_DPI), round(POSTER[1] * POSTER_DPI)
    page = np.empty((h, w, 3), np.uint8)
    page[:] = paper
    gap, y = (h - sum(r.shape[0] for r in rows)) / (len(rows) + 1), 0.0
    for r in rows:
        y += gap
        page[round(y):round(y) + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]
    return page


def poster(name = 'linkedin.mp4', bitrate = 8_000_000):
    """The announcement as a loop of LOOP seconds, drawn on all but two cores, each taking a run of frames so that what
    holds still is drawn once per process, written by Media Foundation as H.264 and encoded again at `bitrate` by
    Windows' own transcoder, into Downloads."""
    import cv2
    out, size = Path.home() / 'Downloads' / name, (round(POSTER[0] * POSTER_DPI), round(POSTER[1] * POSTER_DPI))
    times = [i / FPS for i in range(round(LOOP * FPS))]
    workers = max(1, (os.cpu_count() or 4) - 2)
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(workers) as pool:
        raw = str(Path(tmp) / 'raw.mp4')
        writer = cv2.VideoWriter(raw, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), FPS, size)
        for i, frame in enumerate(pool.map(poster_frame, times, chunksize = -(-len(times) // workers))):
            writer.write(np.ascontiguousarray(frame[..., ::-1]))
        writer.release()
        subprocess.run(['powershell', '-NoProfile', '-Command', TRANSCODE.format(
            src = raw, out = out, bitrate = bitrate, width = size[0], height = size[1])], check = True)
    print(out, out.stat().st_size // 1024, 'kB')


# Windows' MediaTranscoder through PowerShell, which reaches WinRT without any package
TRANSCODE = """
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and
    $_.GetParameters().Count -eq 1 }}
$op = ($m | Where-Object {{ $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
$act = ($m | Where-Object {{ $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncActionWithProgress`1' }})[0]
function Await($o, [Type]$t) {{
    $k = $op.MakeGenericMethod($t).Invoke($null, @($o)); $k.Wait(-1) | Out-Null; $k.Result }}
function WinRT($name, $space) {{ [Type]::GetType("Windows.$space.$name, Windows.$space, ContentType=WindowsRuntime") }}
$file, $folder = (WinRT 'StorageFile' 'Storage'), (WinRT 'StorageFolder' 'Storage')
$coder, $ready = (WinRT 'MediaTranscoder' 'Media.Transcoding'), (WinRT 'PrepareTranscodeResult' 'Media.Transcoding')
$encoding = WinRT 'MediaEncodingProfile' 'Media.MediaProperties'
$src = Await ($file::GetFileFromPathAsync('{src}')) $file
$dir = Await ($folder::GetFolderFromPathAsync((Split-Path '{out}'))) $folder
$dst = Await ($dir.CreateFileAsync((Split-Path '{out}' -Leaf), 'ReplaceExisting')) $file
$profile = $encoding::CreateMp4('HD1080p')
$profile.Video.Bitrate = {bitrate}
$profile.Video.Width = {width}
$profile.Video.Height = {height}
$profile.Audio = $null
$transcoder = New-Object $coder
$prep = Await ($transcoder.PrepareFileTranscodeAsync($src, $dst, $profile)) $ready
if (-not $prep.CanTranscode) {{ throw "Cannot transcode: $($prep.FailureReason)" }}
$act.MakeGenericMethod([double]).Invoke($null, @($prep.TranscodeAsync())).Wait(-1) | Out-Null
"""


if __name__ == '__main__':
    OUT.mkdir(exist_ok = True)
    if sys.argv[1:] == ['trailer']:
        trailer()
        sys.exit()
    if sys.argv[1:] == ['linkedin']:
        poster()
        sys.exit()
    # Real data as a talk, which the README links to as a deck and shows as PowerPoint plays it
    vc.deck(talk(), OUT / 'talk.pptx')
    morph_video(OUT / 'talk.pptx', 'talk.webp')
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
    arrange([[tile(glycan_volcano(), 900), tile(glycan_bars(), 900)]], 'glycans.png')
    arrange([[fit(Image.open(BytesIO(markers().to_png(200))).convert('RGB'), 1400, 1000)]], 'hue_style_size.png')
    arrange([[fit(Image.open(BytesIO(gallery.groups('lumen').to_png(200))).convert('RGB'), 1400, 840)]], 'groups.png')
    arrange([[fit(Image.open(BytesIO(gallery.subgroups('couture').to_png(200))).convert('RGB'), 1400, 840)]],
            'subgroups.png')
    arrange([[fit(Image.open(BytesIO(labels('lumen').to_png(200))).convert('RGB'), 2400, 740)]], 'labels.png')
    # One file that follows the reader's light or dark mode, as GitHub and PyPI show it
    labels('poster').save(OUT / 'themes.svg', themes = True)
    # Focus in shades, and the same figure as an SVG whose samples light up across its panels under the mouse
    arrange([[fit(Image.open(BytesIO(linked(True).to_png(200))).convert('RGB'), 1400, 1080)]], 'focus.png')
    linked(False, 'poster').save(OUT / 'linked.svg', interactive = True)
    arrange([[tile(gallery.heatmap('aura'), 1100)]], 'clustermap.png')
    arrange([[tile(crowd(False), 900), tile(crowd(True), 900)]], 'crowd.png')
    arrange([[tile(figures[k](p + '-dark'), 640) for k, p in zip(('trial', 'volcano', 'stacked', 'embedding'), names)]],
            'dark.png')
    for path in sorted([*OUT.glob('*.png'), *OUT.glob('*.webp'), *OUT.glob('*.pptx')]):
        print(path.name, path.stat().st_size // 1024, 'kB')
