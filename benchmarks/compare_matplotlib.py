"""viscose against matplotlib and seaborn at 300 dpi. Run it alone on the machine. Median of 3 where cheap, one run
for the slowest cases. Needs matplotlib and seaborn."""
import shutil
import statistics
import tempfile
import time
import warnings
from pathlib import Path
import numpy as np
import viscose as vc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter('ignore')
plt.rcParams['font.family'] = ['Century Gothic', 'DejaVu Sans']
rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())
x400, y400 = rng.normal(size = 412), rng.normal(size = 412)
x20k, y20k = rng.normal(size = 20000), rng.normal(size = 20000)
x100k, y100k = rng.normal(size = 100000), rng.normal(size = 100000)
heat = rng.normal(size = (26, 16))
big = rng.normal(size = (300, 100))
t = np.linspace(0, 48, 300)


def m_scatter(x, y, marker = 'o'):
    f, a = plt.subplots(figsize = (8, 8))
    sns.scatterplot(x = x, y = y, ax = a, s = 20, marker = marker)
    a.set_xlabel('log2 fold change')
    return f


def c_scatter(p, x, y, marker = 'o'):
    f, a = vc.subplots(palette = p)
    a.scatter(x, y, s = 20, marker = marker)
    a.set_xlabel('log2 fold change')
    return f


def m_bar():
    f, a = plt.subplots(figsize = (8, 8))
    sns.barplot(x = list('ABCDE'), y = [.46, .61, .74, .86, .95], ax = a)
    return f


def c_bar(p):
    f, a = vc.subplots(palette = p)
    a.bar(list('ABCDE'), [.46, .61, .74, .86, .95])
    return f


def m_line():
    f, a = plt.subplots(figsize = (8, 8))
    for k in (1, 2, 3):
        a.plot(t, np.sin(t / k), label = str(k))
    a.legend()
    return f


def c_line(p):
    f, a = vc.subplots(palette = p)
    for k in (1, 2, 3):
        a.plot(t, np.sin(t / k), label = str(k))
    a.legend('below')
    return f


def m_cluster(d):
    return sns.clustermap(d, figsize = (8, 8)).figure


def c_cluster(p, d):
    return vc.clustermap(d, palette = p, xticklabels = [str(i) for i in range(d.shape[1])],
                         yticklabels = [str(i) for i in range(d.shape[0])])[0]


def run(make_fig, ext, mpl, reps):
    ts, size = [], 0
    for _ in range(reps):
        f = make_fig()
        path = tmp / f'x.{ext}'
        t0 = time.perf_counter()
        f.savefig(path, dpi = 300) if mpl else f.save(path, dpi = 300)
        ts.append(time.perf_counter() - t0)
        size = path.stat().st_size
        if mpl:
            plt.close(f)
    return statistics.median(ts), size


PALETTES = ('poster', 'lumen', 'aura')
print(f'{"plot":22s}{"fmt":5s}{"matplotlib":>18s}' + ''.join(f'{p:>18s}' for p in PALETTES), flush = True)
for name, m, c, reps, formats in [('bars', m_bar, c_bar, 3, 'png pdf svg'),
                                  ('3 lines, legend', m_line, c_line, 3, 'png pdf svg'),
                                  ('scatter 412', lambda: m_scatter(x400, y400), lambda p: c_scatter(p, x400, y400), 3,
                                   'png pdf svg'),
                                  ('clustermap 26x16', lambda: m_cluster(heat), lambda p: c_cluster(p, heat), 3,
                                   'png pdf svg'),
                                  ('scatter 20,000', lambda: m_scatter(x20k, y20k), lambda p: c_scatter(p, x20k, y20k),
                                   1, 'png pdf svg'),
                                  ('triangles 20,000', lambda: m_scatter(x20k, y20k, '^'),
                                   lambda p: c_scatter(p, x20k, y20k, '^'), 1, 'png pdf svg'),
                                  ('clustermap 300x100', lambda: m_cluster(big), lambda p: c_cluster(p, big), 1,
                                   'png pdf svg'),
                                  ('scatter 100,000', lambda: m_scatter(x100k, y100k),
                                   lambda p: c_scatter(p, x100k, y100k), 1, 'pdf svg')]:
    for ext in formats.split():
        row = [run(m, ext, True, reps)] + [run(lambda: c(p), ext, False, reps) for p in PALETTES]
        print(f'{name:22s}{ext:5s}' + ''.join(f'{v[0]:8.2f}s {v[1] / 1024:7.0f}kB' for v in row), flush = True)
shutil.rmtree(tmp)
print('done', flush = True)
