"""Axes: records what to draw, then lays it out and draws it once the figure is rendered."""
import functools
import math
import numbers
import numpy as np
from . import marks, text
from .palettes import mix


# Past this many cells a heatmap is one image, as matplotlib's rasterized heatmaps are, not thousands of shapes
HEAT_IMAGE_CELLS = 2500
LEGEND_ROW = 16.0
COLORBAR_WIDTH = 11.0
LEGEND_SPOTS = ('below', 'right', 'upper left', 'upper right', 'lower left', 'lower right')
# Axis names shrink to fit their cell down to this size, and past it the page grows
LABEL_MIN = 6.5


def nice_ticks(lo, hi, n = 6, steps = (1, 2, 5, 10)):
    """Round tick positions covering [lo, hi], about n of them, at a step of one of `steps` times a power of ten, such
    as (1, 2, 2.5, 5, 10) for matplotlib's quarters too."""
    if not math.isfinite(hi - lo) or hi <= lo:
        return [lo]
    mag = 10 ** math.floor(math.log10((hi - lo) / max(n - 1, 1)))
    # The round step whose tick count comes closest to n, counting the ticks that actually fall inside the range
    step = min((s * mag for s in steps),
               key = lambda v: abs(math.floor(hi / v + 1e-9) - math.ceil(lo / v - 1e-9) + 1 - n))
    first = math.ceil(lo / step - 1e-9)
    # Rounded to the step's own scale, which keeps 0.6 from reading 0.6000000000000001 and 3e-12 from reading 0
    digits = 10 - math.floor(math.log10(step))
    return [round(i * step, digits) for i in range(first, int(math.floor(hi / step + 1e-9)) + 1)]


def fmt(v, powers = False, digits = 3):
    """A tick value without trailing zeros or a stray minus zero, with `digits` significant digits beside a power of ten
    and three more without one. `powers` writes it with a power of ten however small it is, for the ticks beside a large
    one."""
    if v and (powers or abs(v) >= 1e5 or abs(v) < 1e-3):
        e = int(math.floor(math.log10(abs(v))))
        m = '%.*g' % (digits, v / 10.0 ** e)
        # Rounding can carry over into the next power of ten
        if m.lstrip('-') == '10':
            m, e = m.replace('10', '1'), e + 1
        return f'{m[:-1]}10^{{{e}}}' if m.lstrip('-') == '1' else f'{m}×10^{{{e}}}'
    s = '%.*g' % (digits + 3, v)
    return '0' if s == '-0' else s


def tick_labels(ticks):
    """Ticks in one notation, as 80000 next to 10^5 or 0.001 next to 8×10^-4 reads badly."""
    powers = any(v and (abs(v) >= 1e5 or abs(v) < 1e-3) for v in ticks)
    # As many digits as it takes to tell the ticks apart, as on an axis from 1234567 to 1234570
    for digits in range(3, 16):
        labels = [fmt(v, powers, digits) for v in ticks]
        if len(set(labels)) == len(set(ticks)):
            break
    return labels


def _minus(pal, labels):
    """Numbers written with the palette's minus sign, such as a true minus where its font has one."""
    return labels if pal.minus == '-' else [s.replace('-', pal.minus) for s in labels]


def _numeric(values):
    """Whether values are numbers rather than category labels."""
    return np.asarray(values).dtype.kind in 'fiu'


def _runs(ok):
    """Slices of the stretches where a mask holds, which a NaN breaks a line or a band into."""
    edges = np.flatnonzero(np.diff(np.r_[False, ok, False]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


def _gap(positions):
    """The smallest distance between two positions that differ, as ticks set by hand can repeat."""
    gaps = np.abs(np.diff(positions))
    gaps = gaps[gaps > 0]
    return float(gaps.min()) if gaps.size else math.inf


def _extra_lines(s):
    """How many lines a text has past its first."""
    return str(s).count('\n')


def _check_cmap(cmap):
    """A palette brings its color scales; any other name would fail, or worse succeed, as an attribute lookup."""
    if cmap not in ('diverging', 'sequential', 'sequential_r'):
        raise ValueError(f"cmap has to be 'diverging', 'sequential', or 'sequential_r', got {cmap!r}: a palette brings "
                         "these scales")


# Each scale's map from data to even spacing along the axis, and back
WARPS = {'linear': (lambda v: v, lambda u: u),
         'log': (lambda v: np.log10(np.maximum(v, 1e-300)), lambda u: 10.0 ** u),
         'sqrt': (lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda u: np.sign(u) * u * u)}


def _warp(scale, functions):
    """A scale's (forward, inverse) maps; any other name would quietly draw a linear axis."""
    if scale not in (*WARPS, 'function'):
        raise ValueError(f"An axis scale has to be 'linear', 'log', 'sqrt', or 'function', got {scale!r}")
    if (scale == 'function') != (functions is not None):
        raise ValueError("A 'function' scale takes functions = (forward, inverse), and no other scale takes them")
    return WARPS[scale] if functions is None else tuple(functions)


def _check_limits(lo, hi):
    """Limits as floats; a NaN or an infinity would otherwise only fail once the figure is saved."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f'Axis limits have to be finite, got {lo} and {hi}')
    return float(lo), float(hi)


def _check_marker(marker):
    """Any other name would otherwise only fail once the figure is saved."""
    if marker not in marks.MARKERS:
        raise ValueError(f"Unknown marker {marker!r}; available: {', '.join(marks.MARKERS)}")


def _check_tick_labels(ticks, labels):
    """Too few labels would otherwise only fail once the figure is saved, and too many would pass unnoticed."""
    if labels is not None and ticks is None:
        raise ValueError('Tick labels need the ticks they belong to')
    if labels is not None and len(labels) != len(ticks):
        raise ValueError(f'Ticks need one label each, got {len(ticks)} ticks and {len(labels)} labels')


def _check_lengths(x, y):
    """Different lengths would otherwise only fail once the figure is saved."""
    if x.shape != y.shape:
        raise ValueError(f'x and y have to be of the same length, got {x.size} and {y.size} values')


class Axes:
    def __init__(self, figure, pal):
        self.figure, self.pal = figure, pal
        self.artists, self.legend_items, self._legend_opts = [], [], None
        self._legend_size = None
        self._xlim, self._ylim = None, None
        self.xlabel, self.ylabel, self.title_text = '', '', ''
        self._title_fixed, self._title_pad = None, None
        self._xticks, self._yticks, self._xtl, self._ytl = None, None, None, None
        self._tick_size = None
        self._xfmt, self._yfmt = None, None
        self._label_size, self._label_pad = {'x': None, 'y': None}, {'x': None, 'y': None}
        self._xemph, self._yemph = {}, {}
        self._tick_opts, self._spine_set = {'x': {}, 'y': {}}, {}
        self._xrot, self._yright = 0.0, False
        self._xscale, self._yscale = 'linear', 'linear'
        self._xwarp, self._ywarp = WARPS['linear'], WARPS['linear']
        self._xinv, self._yinv = False, False
        self._cycle = 0
        self._cats = {'x': {}, 'y': {}}
        self._hidden = set()
        self._xmargin, self._ymargin = 0.05, 0.05
        self._scale, self._cbar = None, None
        self._over, self._over_plain = [0.0, 0.0], -math.inf
        self._ridge_labels = None
        self._fixed, self._cell, self._span, self._legend_y, self._align = None, None, (1, 1), None, {}
        self.rect = self._room = (0.0, 0.0, 1.0, 1.0)
        # Axes whose limits these share, and whether their tick labels and name stay hidden as an inner panel's
        self._share, self._outer = {'x': [self], 'y': [self]}, {'x': False, 'y': False}
        self.grid_on = pal.grid
        self.show_all = (False, False)

    # --- data helpers ---
    def _next_color(self, color):
        if color is not None:
            return self.pal.color(color)
        c = self.pal.colors(16)[self._cycle % 16]
        self._cycle += 1
        return c

    def _pos(self, vals, axis):
        """Numeric positions; strings become categories, in order of first appearance, with their own tick labels."""
        arr = np.asarray(vals)
        # A DataFrame's column of numbers can arrive as objects
        if arr.dtype == object and all(isinstance(v, numbers.Real) for v in arr.ravel()):
            return arr.astype(float)
        if arr.dtype.kind in 'US' or arr.dtype == object:
            cats = self._cats[axis]
            for v in arr.ravel():
                cats.setdefault(str(v), len(cats))
            return np.array([cats[str(v)] for v in arr.ravel()], dtype = float).reshape(arr.shape)
        return arr.astype(float)

    def _add(self, kind, z, **kw):
        self.artists.append(dict(kind = kind, z = z, order = len(self.artists), **kw))

    def _labeled(self, label, rgb, kind = 'patch', marker = 'o', area = 36.0):
        """A legend entry; `marker` holds a point's shape, or a line's dash."""
        if label is not None and rgb is not None:
            self.legend_items.append((str(label), rgb, kind, marker, area))

    # --- plotting ---
    def plot(self, x, y = None, color = None, lw = None, label = None, marker = False, alpha = 1.0, dash = None,
             ms = 42, z = 4, clip = True, y_fraction = False, offset = None):
        """A line, with markers at its points if `marker` is True or a marker name; a NaN breaks it. `y_fraction` reads
        y as fractions of the axes' height and `offset` moves each point by (dx, dy) points, which together tie a line
        to a position beyond the data. With clip = False it may leave the axes, and shows while all its x are within the
        x limits."""
        x, y = (np.arange(len(x)), x) if y is None else (x, y)
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        rgb = self._next_color(color)
        self._add('line', z, x = x, y = y, rgb = rgb, lw = lw, alpha = alpha, dash = dash, clip = clip,
                  frac = y_fraction, offset = None if offset is None else np.asarray(offset, dtype = float))
        if marker:
            marker = 'o' if marker is True else marker
            _check_marker(marker)
            self._add('points', z + 1, x = x, y = y, s = ms, rgb = rgb, alpha = alpha, marker = marker, edge = None,
                      gid = None)
        self._labeled(label, rgb, 'line', dash)
        return self

    def step(self, x, y, where = 'pre', color = None, lw = None, label = None, alpha = 1.0, dash = None, z = 4,
             clip = True):
        """A line of steps, as in matplotlib, for counts, cumulative distributions, and survival curves: with
        where = 'pre' each y holds from the x before it up to its own, with 'post' from its own x up to the next, and
        with 'mid' between the midpoints around it."""
        if where not in ('pre', 'post', 'mid'):
            raise ValueError(f"A step's where has to be 'pre', 'post', or 'mid', got {where!r}")
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        if where == 'mid' and x.size > 1:
            edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
            xs, ys = np.repeat(edges, 2)[1:-1], np.repeat(y, 2)
        elif where == 'pre':
            # Each corner repeats a point, taking its x from one neighbor and its y from the other
            xs, ys = np.repeat(x, 2)[:-1], np.repeat(y, 2)[1:]
        else:
            xs, ys = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
        return self.plot(xs, ys, color = color, lw = lw, label = label, alpha = alpha, dash = dash, z = z, clip = clip)

    def scatter(self, x, y, s = 30, color = None, c = None, label = None, alpha = 1.0, cmap = 'sequential', vmin = None,
                vmax = None, z = 3, marker = 'o', edge = None, gid = None, hue = None, style = None, size = None,
                sizes = (20.0, 160.0), colors = None, markers = None, hue_label = None, style_label = None,
                size_label = None, legend = 'right', names = None):
        """Points, as in seaborn's scatterplot. `hue`, `style`, and `size` hold one value per point and map to color,
        marker shape, and area, with a legend. Numeric `hue` with many values, or `c`, maps through a palette scale;
        call `colorbar` to show it. `colors` and `markers` are lists or {value: name} dicts. `edge` is (color, width),
        which with color = 'none' draws hollow rings. `names` holds one label per point, which draws nothing but tells a
        viewer such as slate what the point under the mouse is.
        """
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        if names is not None:
            names = np.atleast_1d(np.asarray(names, dtype = object))
            if names.shape != x.shape:
                raise ValueError(f'names has to hold one label per point, got {names.size} for {x.size} points')
        _check_marker(marker)
        edge = None if edge is None else (self.pal.color(edge[0]), edge[1])
        if hue is not None and _numeric(hue) and len(np.unique(hue)) > 8:
            c, hue = hue, None
        if hue is None and style is None and size is None and c is None:
            rgb = self._next_color(color)
            self._add('points', z, x = x, y = y, s = s, rgb = rgb, alpha = alpha, marker = marker, edge = edge,
                      gid = gid, names = names)
            self._labeled(label, rgb, 'point', marker)
            return self
        n = len(x)
        if c is not None:
            _check_cmap(cmap)
            c = np.broadcast_to(np.asarray(c, dtype = float), x.shape)
            known = c[np.isfinite(c)]
            lo = vmin if vmin is not None else known.min() if known.size else 0.0
            hi = vmax if vmax is not None else known.max() if known.size else 1.0
            levels = np.round((c - lo) / (hi - lo if hi > lo else 1.0) * 24)
            self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
        hues, styles = (np.array([''] * n if v is None else [str(u) for u in v], dtype = object) for v in (hue, style))
        hue_levels, style_levels = list(dict.fromkeys(hues)), list(dict.fromkeys(styles))
        if isinstance(colors, dict):
            hue_rgb = {k: self.pal.color(colors[k]) for k in hue_levels}
        elif colors is not None:
            hue_rgb = dict(zip(hue_levels, [self.pal.color(k) for k in colors]))
        elif hue is not None:
            hue_rgb = dict(zip(hue_levels, self.pal.contrast(len(hue_levels))))
        else:
            hue_rgb = {'': self.pal.color(self.pal.muted) if c is not None else self._next_color(color)}
        if style is None:
            style_marker = {'': marker}
        elif isinstance(markers, dict):
            style_marker = {k: markers[k] for k in style_levels}
        else:
            order = list(marks.MARKERS) if markers is None else list(markers)
            style_marker = {k: order[i % len(order)] for i, k in enumerate(style_levels)}
        for v in style_marker.values():
            _check_marker(v)
        area = np.broadcast_to(np.asarray(s, dtype = float), x.shape)
        size_key = []
        if size is not None:
            if _numeric(size):
                v = np.asarray(size, dtype = float)
                low, high = np.nanmin(v), np.nanmax(v)
                span = (sizes[1] - sizes[0]) / (high - low if high > low else 1.0)
                scale = lambda u: sizes[0] + (np.asarray(u) - low) * span
                area = scale(v)
                ticks = nice_ticks(float(low), float(high), 4)
                size_key = [(name, float(scale(t))) for name, t in zip(tick_labels(ticks), ticks)]
            else:
                names = list(dict.fromkeys(str(u) for u in size))
                steps = np.linspace(sizes[0], sizes[1], max(len(names), 2))
                area = np.array([steps[names.index(str(u))] for u in size], dtype = float)
                size_key = [(k, float(steps[i])) for i, k in enumerate(names)]
        for h in hue_levels:
            for st in style_levels:
                m = (hues == h) & (styles == st)
                # A mapped color makes one group per step of its scale
                groups = [(m, hue_rgb[h])] if c is None else [
                    (m & (levels == level), tuple(getattr(self.pal, cmap)(level / 24)))
                    for level in np.unique(levels[m & np.isfinite(levels)])]
                for chosen, rgb in groups:
                    if chosen.any():
                        self._add('points', z, x = x[chosen], y = y[chosen], s = area[chosen], rgb = rgb, alpha = alpha,
                                  marker = style_marker[st], edge = edge, gid = gid,
                                  names = None if names is None else names[chosen])
        sections = (hue is not None) + (style is not None) + (size is not None)
        plain = self.pal.color(self.pal.muted) if hue is not None else hue_rgb['']
        if hue is not None:
            if sections > 1:
                self.legend_items.append((hue_label or 'Color', None, 'header', 'o', 0.0))
            for h in hue_levels:
                self._labeled(h, hue_rgb[h], 'point', marker)
        if style is not None:
            if sections > 1:
                self.legend_items.append((style_label or 'Shape', None, 'header', 'o', 0.0))
            for st in style_levels:
                self._labeled(st, plain, 'point', style_marker[st])
        if size is not None:
            if sections > 1:
                self.legend_items.append((size_label or 'Size', None, 'header', 'o', 0.0))
            for name, a in size_key:
                self._labeled(name, plain, 'point', marker, a)
        if legend and sections:
            self.legend('right' if legend is True else legend)
        return self

    def bar(self, x, height, width = None, bottom = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
            z = 3):
        """Vertical bars; several colors give one bar each, and string positions become category labels. `width` can
        hold one value per bar."""
        x = np.atleast_1d(self._pos(x, 'x'))
        h = np.broadcast_to(np.asarray(height, dtype = float), x.shape)
        b = np.broadcast_to(np.asarray(bottom, dtype = float), x.shape)
        width = np.broadcast_to(np.asarray(self.pal.bar_width if width is None else width, dtype = float), x.shape)
        cols = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(x)
        self._add('bars', z, x = x, h = h, b = b, w = width, rgb = cols, alpha = alpha, horizontal = False,
                  stacked = stacked)
        self._labeled(label, cols[0] if cols else None)
        return self

    def barh(self, y, width, height = None, left = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
             z = 3):
        """Horizontal bars; `height` can hold one value per bar."""
        y = np.atleast_1d(self._pos(y, 'y'))
        w = np.broadcast_to(np.asarray(width, dtype = float), y.shape)
        b = np.broadcast_to(np.asarray(left, dtype = float), y.shape)
        height = np.broadcast_to(np.asarray(self.pal.bar_width if height is None else height, dtype = float), y.shape)
        cols = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(y)
        self._add('bars', z, x = y, h = w, b = b, w = height, rgb = cols, alpha = alpha, horizontal = True,
                  stacked = stacked)
        self._labeled(label, cols[0] if cols else None)
        return self

    def stacked_bar(self, x, values, labels = None, colors = None, width = None):
        """Bars stacked from a (bars, segments) array. Pass colors = pal.groups(3, 2) to group segments by hue."""
        values = np.asarray(values, dtype = float)
        colors = self.pal.colors(values.shape[1]) if colors is None else [self.pal.color(k) for k in colors]
        # A segment needs more room than a needle to show its color
        width = (0.42 if self.pal.bar == 'needle' else self.pal.bar_width) if width is None else width
        base = np.zeros(values.shape[0])
        for j in range(values.shape[1]):
            self.bar(x, values[:, j], bottom = base, color = colors[j], width = width,
                     label = labels[j] if labels is not None else None, stacked = True)
            base = base + values[:, j]
        return self

    def hist(self, x, bins = 20, color = None, label = None, density = False, alpha = 1.0, frequency = False):
        """A histogram, with its bars touching except for the palette's gap. `density` scales it to an area of one, and
        `frequency` gives counts per unit of x."""
        x = np.asarray(x, dtype = float)
        counts, edges = np.histogram(x[np.isfinite(x)], bins = bins, density = density)
        self.bar(edges[:-1] + np.diff(edges) / 2, counts / np.diff(edges) if frequency else counts,
                 width = np.diff(edges) * 0.94, color = color, label = label, alpha = alpha)
        return self

    def fill_between(self, x, y1, y2 = 0.0, color = None, alpha = 0.22, label = None, z = 2):
        """A band between two curves; the band behind a line takes the line's color."""
        x = self._pos(x, 'x')
        y1 = np.broadcast_to(np.asarray(y1, dtype = float), x.shape)
        y2 = np.broadcast_to(np.asarray(y2, dtype = float), x.shape)
        rgb = self._next_color(color)
        self._add('band', z, x = x, y1 = y1, y2 = y2, rgb = rgb, alpha = alpha)
        self._labeled(label, rgb)
        return self

    def contour(self, x, y, values, levels = 5, color = 'muted', lw = 0.7, alpha = 1.0, dash = None, label = None,
                z = 2):
        """Lines of equal value over a grid, where `values` has one row per y and one column per x. `levels` is how many
        evenly spaced levels to draw between the extremes, or a list of levels."""
        x, y, m = np.asarray(x, dtype = float), np.asarray(y, dtype = float), np.asarray(values, dtype = float)
        if m.shape != (len(y), len(x)):
            raise ValueError(f'values have to have one row per y and one column per x, got {m.shape} for {len(y)} y '
                             f'and {len(x)} x')
        known = np.isfinite(m)
        if np.ndim(levels) == 0:
            levels = np.linspace(m[known].min(), m[known].max(), int(levels) + 2)[1:-1] if known.any() else []
        lines = []
        for level in levels:
            above = m >= level
            across = (above[:, :-1] != above[:, 1:]) & known[:, :-1] & known[:, 1:]
            up = (above[:-1] != above[1:]) & known[:-1] & known[1:]
            # The level crosses each of these grid edges once, where a straight line between its two ends says
            spots = {}
            for r, j in np.argwhere(across):
                spots['h', r, j] = x[j] + (level - m[r, j]) / (m[r, j + 1] - m[r, j]) * (x[j + 1] - x[j]), y[r]
            for r, j in np.argwhere(up):
                spots['v', r, j] = x[j], y[r] + (level - m[r, j]) / (m[r + 1, j] - m[r, j]) * (y[r + 1] - y[r])
            links = {k: [] for k in spots}
            for r, j in np.argwhere(across[:-1] | across[1:] | up[:, :-1] | up[:, 1:]):
                low, high, left, right = ('h', r, j), ('h', r + 1, j), ('v', r, j), ('v', r, j + 1)
                ends = [k for k in (low, high, left, right) if k in spots]
                if len(ends) == 2:
                    pairs = [ends]
                elif len(ends) == 4:
                    # A saddle: the cell's center says which two corners stay joined
                    center = m[r:r + 2, j:j + 2].mean() >= level
                    pairs = [(low, right), (left, high)] if above[r, j] == center else [(left, low), (high, right)]
                else:
                    continue
                for a, b in pairs:
                    links[a].append(b)
                    links[b].append(a)
            # One line per stretch, so that transparency and dashes run along it: open ones from an end, then the loops
            for start in [k for k, v in links.items() if len(v) == 1] + list(links):
                if not links[start]:
                    continue
                chain, at = [start], start
                while links[at]:
                    step = links[at].pop()
                    links[step].remove(at)
                    chain.append(step)
                    at = step
                lines.append(tuple(np.array(v) for v in zip(*[spots[k] for k in chain])))
        rgb = self.pal.color(color)
        self._add('contour', z, x = x, y = y, lines = lines, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        self._labeled(label, rgb, 'line', dash)
        return self

    def errorbar(self, x, y, yerr, color = 'ink', lw = 1.2, capsize = 3.0, z = 5):
        """Error bars around y; `yerr` is one value, one per point, or two rows of those, for below and above."""
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(np.asarray(y, dtype = float))
        _check_lengths(x, y)
        below, above = yerr if np.ndim(yerr) == 2 else (yerr, yerr)
        self._add('errors', z, x = x, lo = y - np.broadcast_to(np.asarray(below, dtype = float), y.shape),
                  hi = y + np.broadcast_to(np.asarray(above, dtype = float), y.shape), rgb = self.pal.color(color),
                  lw = lw, cap = capsize)
        return self

    def _groups(self, data, positions, labels, colors, whis):
        """Groups of values for box, strip, and violin plots, as (positions, labels, colors, values, stats): stats per
        group are its lower whisker, quartile, median, quartile, upper whisker, and the values beyond the whiskers, or
        None for a group without values."""
        if isinstance(data, dict):
            labels, data = list(data) if labels is None else labels, list(data.values())
        positions = np.arange(len(data), dtype = float) if positions is None else np.asarray(positions, dtype = float)
        colors = self.pal.contrast(len(data)) if colors is None else [self.pal.color(k) for k in colors]
        values, stats = [], []
        for group in data:
            v = np.asarray(group, dtype = float).ravel()
            v = v[np.isfinite(v)]
            values.append(v)
            if not v.size:
                stats.append(None)
                continue
            q1, median, q3 = np.percentile(v, (25, 50, 75))
            # With short whiskers a few values can leave none within reach, and then they end at the box
            within = v[(v >= q1 - whis * (q3 - q1)) & (v <= q3 + whis * (q3 - q1))]
            low, high = (within.min(), within.max()) if within.size else (q1, q3)
            stats.append((low, q1, median, q3, high, v[(v < low) | (v > high)]))
        return positions, labels, colors, values, stats

    def _strip(self, values, spread, size):
        """Each group's points spread across a share of its width, `spread` per value, by a golden-ratio sequence over
        the values in order, so that values close in size lie apart, and the same on every draw."""
        offsets = []
        for v, share in zip(values, spread):
            u = np.empty(len(v))
            u[np.argsort(v, kind = 'stable')] = (np.arange(len(v)) * 0.6180339887498949 + 0.5) % 1.0 - 0.5
            offsets.append(u * share)
        # Many points get smaller, so that a crowd still reads as points
        sizes = [float(size) if size is not None else 16.0 * min(1.0, math.sqrt(150.0 / max(len(v), 1))) + 2.0
                 for v in values]
        return offsets, sizes

    def boxplot(self, data, positions = None, labels = None, width = 0.5, colors = None, whis = 1.5, fliers = True,
                alpha = 1.0, z = 3, points = True, jitter = 0.7, point_size = None):
        """Box plots of several groups, in the palette's manner: `data` is a list of value lists or a {label: values}
        dict. A box spans the quartiles with its median marked, and whiskers reach the last value within `whis` box
        heights. With `points`, every value shows as a point over its box, spread across `jitter` of its width, and the
        box takes a paler shade; without, `fliers` shows the values beyond the whiskers. Groups sit at 0, 1, 2, and so
        on, unless `positions` says otherwise."""
        positions, labels, colors, values, stats = self._groups(data, positions, labels, colors, whis)
        offsets, sizes = self._strip(values, [jitter * width] * len(values), point_size)
        self._add('box', z, x = positions, stats = stats, w = width, rgb = colors, alpha = alpha, fliers = fliers,
                  values = values, offsets = offsets if points else None, sizes = sizes, box = True)
        if labels is not None:
            self.set_xticks(positions, labels)
        return self

    def stripplot(self, data, positions = None, labels = None, width = 0.5, colors = None, alpha = 1.0, z = 3,
                  jitter = 0.7, point_size = None):
        """Every value of several groups as a point, spread across `jitter` of the group's `width` so that equal values
        stay apart, as in seaborn's stripplot. `data` is a list of value lists or a {label: values} dict."""
        positions, labels, colors, values, stats = self._groups(data, positions, labels, colors, 1.5)
        offsets, sizes = self._strip(values, [jitter * width] * len(values), point_size)
        self._add('box', z, x = positions, stats = stats, w = width, rgb = colors, alpha = alpha, fliers = False,
                  values = values, offsets = offsets, sizes = sizes, box = False)
        if labels is not None:
            self.set_xticks(positions, labels)
        return self

    def violinplot(self, data, positions = None, labels = None, width = 0.8, colors = None, alpha = 1.0, z = 3,
                   inner = 'box', bw_adjust = 1.0, cut = 2.0, point_size = None):
        """Violin plots of several groups, in the palette's manner: each group's density, by a Gaussian kernel of
        Scott's bandwidth times `bw_adjust`, mirrored around its position and running `cut` bandwidths past its
        extremes, as in seaborn. All violins share one scale, so that each holds the same area, and the widest is
        `width` across. `inner` is 'box' for the quartiles and median inside, 'points' for every value spread across the
        violin's width at its height, or None. `data` is a list of value lists or a {label: values} dict."""
        if inner not in ('box', 'points', None):
            raise ValueError(f"A violin's inner has to be 'box', 'points', or None, got {inner!r}")
        positions, labels, colors, values, stats = self._groups(data, positions, labels, colors, 1.5)
        shapes = []
        for v in values:
            # A single value or equal ones have no spread to draw
            bw = np.std(v, ddof = 1) * len(v) ** -0.2 * bw_adjust if len(v) > 1 else 0.0
            if not bw > 0:
                shapes.append(None)
                continue
            grid = np.linspace(v.min() - cut * bw, v.max() + cut * bw, 128)
            spots, weights = np.unique(v, return_counts = True)
            if len(spots) > 5000:
                # Many values go through a histogram first, far finer than the kernel
                weights, edges = np.histogram(v, 2000)
                spots = (edges[:-1] + edges[1:]) / 2
            density = np.zeros(len(grid))
            for start in range(0, len(spots), 2000):
                d = (grid[:, None] - spots[None, start:start + 2000]) / bw
                density += (np.exp(-0.5 * d * d) * weights[start:start + 2000]).sum(axis = 1)
            shapes.append((grid, density / (len(v) * bw * math.sqrt(2 * math.pi))))
        peak = max([s[1].max() for s in shapes if s is not None], default = 1.0)
        shapes = [None if s is None else (s[0], s[1] / peak * width / 2) for s in shapes]
        offsets, sizes = self._strip(values, [1.0] * len(values), point_size)
        if inner == 'points':
            # Spread as wide as the violin is at each value, within it
            offsets = [u * 1.6 * np.interp(v, s[0], s[1]) if s is not None else u * 0.0
                       for u, v, s in zip(offsets, values, shapes)]
        self._add('violin', z, x = positions, stats = stats, shapes = shapes, w = width, rgb = colors, alpha = alpha,
                  inner = inner, values = values, offsets = offsets, sizes = sizes)
        if labels is not None:
            self.set_xticks(positions, labels)
        return self

    def stem(self, x, y, color = None, lw = 3.0, label = None, cap = False, z = 4):
        """Vertical sticks from zero, as in a mass spectrum."""
        x, y = np.atleast_1d(np.asarray(x, dtype = float)), np.atleast_1d(np.asarray(y, dtype = float))
        _check_lengths(x, y)
        rgb = self._next_color(color)
        self._add('stems', z, x = x, y = y, rgb = rgb, lw = lw, cap = cap)
        self._labeled(label, rgb)
        return self

    def hlines(self, y, x0, x1, color = None, lw = 1.5, alpha = 1.0, dash = None, label = None, z = 2):
        """Horizontal segments, under the points, such as the intervals of a forest plot."""
        y = np.atleast_1d(np.asarray(y, dtype = float))
        x0 = np.broadcast_to(np.asarray(x0, dtype = float), y.shape)
        x1 = np.broadcast_to(np.asarray(x1, dtype = float), y.shape)
        rgb = self._next_color(color)
        self._add('hsegs', z, y = y, x0 = x0, x1 = x1, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        self._labeled(label, rgb, 'line', dash)
        return self

    def vlines(self, x, y0, y1, color = None, lw = 1.0, alpha = 1.0, dash = None, label = None, z = 3):
        """Vertical segments of exactly this width, as many sticks need; `stem` draws them in the palette's manner. A
        list of colors gives each its own."""
        x = np.atleast_1d(np.asarray(x, dtype = float))
        y0 = np.broadcast_to(np.asarray(y0, dtype = float), x.shape)
        y1 = np.broadcast_to(np.asarray(y1, dtype = float), x.shape)
        rgb = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(x)
        self._add('vsegs', z, x = x, y0 = y0, y1 = y1, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        self._labeled(label, rgb[0] if rgb else None, 'line', dash)
        return self

    def axvspan(self, x0, x1, color = 'a2', alpha = 0.25, label = None, z = 1):
        """A band across the axes' height between two x values."""
        rgb = self.pal.color(color)
        self._add('vspan', z, x0 = float(x0), x1 = float(x1), rgb = rgb, alpha = alpha)
        self._labeled(label, rgb)
        return self

    def axhspan(self, y0, y1, color = 'a2', alpha = 0.25, label = None, z = 1):
        """A band across the axes' width between two y values."""
        rgb = self.pal.color(color)
        self._add('hspan', z, y0 = float(y0), y1 = float(y1), rgb = rgb, alpha = alpha)
        self._labeled(label, rgb)
        return self

    def image(self, pixels, x, y, width, height, offset = (0.0, 0.0), y_fraction = False, z = 6):
        """An array of rows and columns, gray or with RGB or RGBA channels, first row at the top, in integers up to 255
        or floats up to 1, with its center at a data position, moved by `offset` points. `width` and `height` are in
        points. `y_fraction` reads y as a fraction of the axes' height. It may leave the axes, and shows while its
        position is within the limits."""
        px = np.asarray(pixels)
        px = (np.clip(px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if px.dtype.kind == 'f' else px.astype(np.uint8)
        px = np.dstack([px] * 3) if px.ndim == 2 else px
        alpha = px[..., 3].tobytes() if px.shape[2] == 4 else None
        self._add('image', z, x = float(x), y = float(y), w = float(width), h = float(height), dx = offset[0],
                  dy = offset[1], frac = y_fraction, nx = px.shape[1], ny = px.shape[0],
                  rgb = px[..., :3].tobytes(), alpha = alpha)
        return self

    def arrow(self, x0, y0, x1, y1, color = 'muted', lw = 1.0, alpha = 1.0, gid = None, z = 2, head = None,
              shrink = (0.0, 0.0), bend = 0.0):
        """An arrow between two data points; `gid` names it in SVG output, for hover text. `head` is the head's length
        in points, or 0 for a plain leader line, `shrink` how many points it stays short of its start and its end, as
        beside a marker or a label, and `bend` turns it into an arc whose middle stands that share of its length to its
        left."""
        self._add('arrow', z, x0 = float(x0), y0 = float(y0), x1 = float(x1), y1 = float(y1),
                  rgb = self.pal.color(color), lw = lw, alpha = alpha, gid = gid, head = head,
                  shrink = tuple(float(v) for v in shrink), bend = float(bend))
        return self

    def axline(self, xy, slope = None, xy2 = None, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        """A straight line through a data point at a slope, or through a second point, across the whole axes, such as
        the diagonal of observed against predicted values; it leaves the limits to the data."""
        if (slope is None) == (xy2 is None):
            raise ValueError('axline takes a slope or a second point, and only one of them')
        x2, y2 = (xy[0] + 1.0, xy[1] + slope) if xy2 is None else xy2
        self._add('aline', z, x0 = float(xy[0]), y0 = float(xy[1]), x1 = float(x2), y1 = float(y2),
                  rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash)
        return self

    def rectangle(self, x, y, width, height, color = 'faint', edge = None, radius = 0.0, alpha = 1.0, z = 1,
                  axes_fraction = False):
        """A rectangle from its lower left corner, in data units or with axes_fraction = True in fractions of the
        axes, such as a box in a schematic or a card behind text. `edge` is (color, width), `radius` rounds its corners
        by that many points, and color = 'none' leaves only the edge."""
        self._add('rect', z, x = float(x), y = float(y), w = float(width), h = float(height),
                  rgb = self.pal.color(color), edge = None if edge is None else (self.pal.color(edge[0]), edge[1]),
                  radius = float(radius), alpha = alpha, frac = axes_fraction)
        return self

    def axhline(self, y = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        self._add('hline', z, v = float(y), rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash)
        return self

    def axvline(self, x = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        self._add('vline', z, v = float(x), rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash)
        return self

    def heatmap(self, data, xticklabels = None, yticklabels = None, cmap = 'diverging', vmin = None, vmax = None,
                center = 0.0, labels_right = True, show_all = False, colorbar = False, colorbar_label = '',
                annot = False, fmt = '.2f', annot_size = None, names = None):
        """A matrix of cells, first row at the top, where NaN is left empty. `cmap` is diverging, sequential, or
        sequential_r, which puts the darkest color at the lowest value. Too many labels are thinned unless show_all,
        which can also be a pair for the x and y axes; the page grows for those that must all show. `annot` writes each
        cell's value into it, formatted by `fmt`, a format spec or a function of the value, in whichever of ink or white
        reads over the cell. `names` holds the
        rows' and the columns' labels as a pair, by default the tick labels, which draws nothing but tells a viewer such
        as slate what the cell under the mouse is."""
        _check_cmap(cmap)
        m = np.asarray(data, dtype = float)
        if m.ndim != 2 or not m.size:
            raise ValueError(f'A heatmap needs rows and columns of at least one cell, got an array of shape {m.shape}')
        rows, columns = (yticklabels, xticklabels) if names is None else names
        for given, n, what in ((rows, m.shape[0], 'row'), (columns, m.shape[1], 'column')):
            if names is not None and given is not None and len(given) != n:
                raise ValueError(f'names has to hold one label per {what}, got {len(given)} for {n} {what}s')
        self.show_all = show_all if isinstance(show_all, tuple) else (show_all, show_all)
        known = m[np.isfinite(m)]
        if cmap == 'diverging':
            d = np.max(np.abs(known - center), initial = 0.0) if vmin is None and vmax is None else max(
                abs((vmin if vmin is not None else center) - center),
                abs((vmax if vmax is not None else center) - center))
            d = float(d) if d else 1.0
            t = 0.5 + (m - center) / (2 * d)
            lo, hi = center - d, center + d
        else:
            lo = vmin if vmin is not None else known.min() if known.size else 0.0
            hi = vmax if vmax is not None else known.max() if known.size else 1.0
            t = (m - lo) / (hi - lo if hi > lo else 1.0)
        self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
        rgb = getattr(self.pal, cmap)(t)
        self._add('heat', 2, m = m, rgb = rgb, rows = None if rows is None else [str(v) for v in rows],
                  columns = None if columns is None else [str(v) for v in columns])
        if annot:
            for r, j in np.argwhere(np.isfinite(m)):
                self.annotate(fmt(m[r, j]) if callable(fmt) else format(m[r, j], fmt), (j + 0.5, m.shape[0] - r - 0.5),
                              size = annot_size or 8.0 * self.pal.spacing, halo = 0.0, color = self.pal.on(rgb[r, j]),
                              ha = 'center', clip = True)
        self.set_xlim(0, m.shape[1])
        self.set_ylim(0, m.shape[0])
        # An empty list of labels means none, as None does
        named = xticklabels is not None and len(xticklabels) > 0
        turned = 90.0 if named and max(len(str(v)) for v in xticklabels) > 2 else 0.0
        self.set_xticks(list(np.arange(m.shape[1]) + 0.5), xticklabels, turned) if named else self.set_xticks([])
        if yticklabels is not None and len(yticklabels) > 0:
            self.set_yticks(list(np.arange(m.shape[0]) + 0.5), [str(v) for v in yticklabels][::-1])
        else:
            self.set_yticks([])
        self._yright = labels_right
        self._hidden |= {'spines', 'grid', 'ticks'}
        self.grid_on = False
        if colorbar:
            self.colorbar(colorbar_label)
        return self

    def colorbar(self, label = '', loc = 'right', source = None, integer = False):
        """Show the color scale of this axes, or of `source`: to its right, or filling it with loc = 'fill'. `integer`
        keeps whole numbers only on its ticks, for counts."""
        if loc not in ('right', 'fill'):
            raise ValueError(f"A colorbar's place has to be right or fill, got {loc!r}")
        self._cbar = dict(label = label, loc = loc, source = source or self, integer = integer)
        return self

    def ridgeline(self, x, curves, labels = None, overlap = 2.05):
        """Distributions stacked with overlap, from the back (first) to the front (last)."""
        x = np.asarray(x, dtype = float)
        heights = [np.nan_to_num(np.asarray(y, dtype = float), posinf = 0.0, neginf = 0.0) for y in curves]
        for y in heights:
            _check_lengths(x, y)
        # A curve of zeros stays flat
        curves = [y / (y.max(initial = 0.0) or 1.0) for y in heights]
        n = len(curves)
        self._add('ridges', 5, x = x, curves = curves, overlap = overlap, labels = labels)
        self.set_ylim(-0.1, n + overlap - 0.9)
        self.set_xlim(float(x.min()), float(x.max()))
        self._yticks = []
        self._hidden.add('left')
        self.grid_on = False
        if labels is not None:
            self._ridge_labels = [str(v) for v in labels]
        return self

    def dendrogram(self, icoord, dcoord, orientation = 'top', n = None, labels = None, scale = False, colors = None,
                   show_all = False):
        """Link lines as scipy gives them; heights run away from the matrix that sits on the given side. `labels` name
        the leaves, which too many of are thinned unless show_all, `scale` keeps ticks on the height axis, and `colors`
        gives each link its own color, to mark clusters."""
        self._add('dendro', 3, ic = icoord, dc = dcoord, orient = orientation,
                  rgb = None if colors is None else [self.pal.color(k) for k in colors])
        self.show_all = (show_all, show_all)
        depth = max(max(d) for d in dcoord) or 1.0
        span = (0, n if n is not None else max(max(i) for i in icoord) / 10 + 0.5)
        if orientation == 'left':
            self.set_xlim(depth, 0)
            self.set_ylim(*span)
        else:
            self.set_ylim(0, depth)
            self.set_xlim(*span)
        self._hidden |= {'spines', 'grid', 'ticks'}
        self._xticks, self._yticks = [], []
        self.grid_on = False
        if labels is not None:
            spots = list(np.arange(len(labels)) + 0.5)
            self.set_yticks(spots, labels) if orientation == 'left' else self.set_xticks(spots, labels)
            self._yright = orientation == 'left'
        if scale and orientation == 'left':
            self._xticks = None
        elif scale:
            self._yticks = None
        return self

    def text(self, x, y, s, size = None, color = 'ink', bold = None, ha = 'left', va = 'center', rotation = 0.0,
             halo = 0.0, z = 20, axes_fraction = False, linespacing = 1.25):
        """Text at a data position, or at a fraction of the axes with axes_fraction = True; several lines are
        `linespacing` sizes apart."""
        self._add('text', z, x = float(x), y = float(y), s = str(s), size = size, rgb = self.pal.color(color),
                  bold = bold, ha = ha, va = va, rotation = rotation, halo = halo, dx = 0.0, dy = 0.0,
                  frac = axes_fraction, tag = False, clip = False, spread = False, linespacing = linespacing,
                  placed = True)
        return self

    def annotate(self, s, xy, offset = (0.0, 0.0), size = None, color = 'ink', bold = None, ha = 'left', va = 'center',
                 rotation = 0.0, halo = None, z = 20, svg_label = False, clip = False, spread = False,
                 linespacing = 1.25):
        """Text next to a data point, offset in points, with a halo of the page's color that keeps it readable over the
        marks. With svg_label = True, SVG output writes it the way matplotlib writes text, so that glycowork's
        annotate_figure can replace it with a glycan drawing. With clip = True it shows only while its point is within
        the limits. Labels with spread = True on one side of their points, such as names at the ends of lines, move
        apart vertically as little as it takes for none to overlap, within the axes' height, each joined to its point
        by a hairline where it had to move. Without a size it takes 8.6 points, and its halo 2.6, both scaled by the
        palette's spacing."""
        k = self.pal.spacing
        self._add('text', z, x = float(xy[0]), y = float(xy[1]), s = str(s), size = 8.6 * k if size is None else size,
                  rgb = self.pal.color(color), bold = bold, ha = ha, va = va, rotation = rotation,
                  halo = 2.6 * k if halo is None else halo, dx = offset[0], dy = offset[1], frac = False,
                  tag = svg_label, clip = clip, spread = spread, linespacing = linespacing)
        return self

    # --- settings ---
    def set_xlabel(self, s, size = None, pad = None):
        """The x axis's name, at `size` points or the palette's size for axis names, `pad` points below the tick labels
        if given, as matplotlib's labelpad. An inner panel of axes sharing x keeps it hidden, as matplotlib does."""
        self.xlabel, self._label_size['x'], self._label_pad['x'] = '' if self._outer['x'] else s, size, pad
        return self

    def set_ylabel(self, s, size = None, pad = None):
        """The y axis's name, at `size` points or the palette's size for axis names, `pad` points beside the tick labels
        if given, as matplotlib's labelpad. An inner panel of axes sharing y keeps it hidden, as matplotlib does."""
        self.ylabel, self._label_size['y'], self._label_pad['y'] = '' if self._outer['y'] else s, size, pad
        return self

    def set_title(self, s, size = None, pad = None):
        """The axes' title; without a size it takes 15 points, less where it would not fit. `pad` is the points from the
        axes to its baseline, eight scaled by the palette's spacing unless given."""
        self.title_text, self._title_fixed, self._title_pad = s, size, pad
        return self

    def set_xlim(self, lo, hi):
        self._xlim = _check_limits(lo, hi)
        return self

    def set_ylim(self, lo, hi):
        self._ylim = _check_limits(lo, hi)
        return self

    def set_xticks(self, ticks = None, labels = None, rotation = 0.0, fmt = None, emphasis = None):
        """Ticks at these positions, or automatic ones with ticks = None, with these labels or their values, whose
        labels turn by `rotation` degrees. `fmt` writes the values, as a format string such as '{:.0%}' or a
        function of the value. `emphasis` maps labels to a color they are set in, bold, such as the method a figure is
        about."""
        _check_tick_labels(ticks, labels)
        self._xticks = None if ticks is None else list(ticks)
        self._xtl = None if labels is None else [str(v) for v in labels]
        self._xrot, self._xfmt, self._xemph = rotation, fmt, dict(emphasis or {})
        return self

    def set_yticks(self, ticks = None, labels = None, fmt = None, emphasis = None):
        """Ticks at these positions, or automatic ones with ticks = None, with these labels or their values. `fmt`
        writes the values, as a format string such as '{:.0%}' or a function of the value. `emphasis` maps labels to a
        color they are set in, bold."""
        _check_tick_labels(ticks, labels)
        self._yticks = None if ticks is None else list(ticks)
        self._ytl = None if labels is None else [str(v) for v in labels]
        self._yfmt, self._yemph = fmt, dict(emphasis or {})
        return self

    def set_tick_size(self, size):
        """The size of the tick labels in points, which otherwise the palette decides."""
        self._tick_size = size
        return self

    def tick_params(self, axis = 'both', length = None, labelsize = None, pad = None, labels = None, top = None,
                    fit = None):
        """Ticks of the x axis, the y axis, or both, as in matplotlib: their `length` and the `pad` between them and
        their labels in points, the labels' size, and labels = False to leave the labels out, as beside a panel
        that shares them. top = True writes the x labels above the axes, as over a table of scores, where a positive
        `rotation` turns them up and to the right. fit = False keeps every label as set, level or turned as asked,
        where viscose would otherwise turn or thin labels that crowd. What is not given stays as it was."""
        if axis not in ('x', 'y', 'both'):
            raise ValueError(f"tick_params takes axis = 'x', 'y', or 'both', got {axis!r}")
        if top is not None and axis != 'x':
            raise ValueError("Only the x labels can go on top, with axis = 'x'")
        for name in ('x', 'y') if axis == 'both' else (axis, ):
            given = dict(length = length, labelsize = labelsize, pad = pad, labels = labels, top = top, fit = fit)
            self._tick_opts[name].update({k: v for k, v in given.items() if v is not None})
        return self

    def set_spines(self, left = None, bottom = None):
        """Show or hide the left and the bottom axis line, with their ticks; None keeps the palette's choice."""
        self._spine_set.update({k: bool(v) for k, v in (('left', left), ('bottom', bottom)) if v is not None})
        return self

    def set_xscale(self, s, functions = None):
        """'linear', 'log', 'sqrt' (signed, so zero and below keep their place), or 'function' with
        functions = (forward, inverse), two maps over arrays from data to even spacing along the axis and back. Axes
        sharing x share it."""
        warp = _warp(s, functions)
        for ax in self._share['x']:
            ax._xwarp, ax._xscale = warp, s
        return self

    def set_yscale(self, s, functions = None):
        """'linear', 'log', 'sqrt' (signed, so zero and below keep their place), or 'function' with
        functions = (forward, inverse), two maps over arrays from data to even spacing along the axis and back. Axes
        sharing y share it."""
        warp = _warp(s, functions)
        for ax in self._share['y']:
            ax._ywarp, ax._yscale = warp, s
        return self

    def set(self, **settings):
        """Several settings at once, as in matplotlib: xlabel, ylabel, title, xlim, ylim, xscale, yscale, tick_size."""
        for name, value in settings.items():
            method = getattr(self, 'set_' + name)
            method(*value) if name in ('xlim', 'ylim') else method(value)
        return self

    def invert_xaxis(self):
        self._xinv = True
        return self

    def invert_yaxis(self):
        self._yinv = True
        return self

    def margins(self, x = 0.05, y = None):
        """Space around the data, as a share of its range."""
        self._xmargin, self._ymargin = x, x if y is None else y
        return self

    def grid(self, on = True):
        self.grid_on = on
        return self

    def legend(self, loc = 'below'):
        """Show a legend: below or right of the axes, or inside at upper left, upper right, lower left, or lower
        right."""
        # Any other name would quietly land at the lower left
        if loc not in LEGEND_SPOTS:
            raise ValueError(f"A legend's place has to be one of {', '.join(LEGEND_SPOTS)}, got {loc!r}")
        self._legend_opts = loc
        return self

    def off(self):
        """Hide axis lines, ticks and grid, for figures whose position axes carry no meaning."""
        self._hidden |= {'spines', 'ticks', 'grid'}
        self._xticks, self._yticks = [], []
        self.grid_on = False
        return self

    # --- positions ---
    def boxes(self):
        """What the axes draw, as rows of (left, bottom, right, top) in page points once `Figure.layout` has placed
        them: one per point, per piece of a line about a point long, per bar and rectangle, and per line of text at its
        unspread place, for placing labels, keys, and cards clear of it all. Bands and spans are left out, as the ground
        under the data."""
        pal, out = self.pal, []
        x0, y0, w, h = self.rect
        lo_x, hi_x = sorted(self.xlim)
        lo_y, hi_y = sorted(self.ylim)
        pieces = []
        for a in self.artists:
            k = a['kind']
            if k == 'points':
                x, y = np.atleast_1d(a['x']), np.atleast_1d(a['y'])
                keep = np.isfinite(x) & np.isfinite(y) & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
                area = np.broadcast_to(np.asarray(a['s'], dtype = float), x.shape)
                r = np.sqrt(np.maximum(area, 0) * pal.point_scale) / 2
                r = r[keep] + (a['edge'][1] / 2 if a['edge'] else 0.0)
                px, py = self._tx(x[keep]), self._ty(y[keep])
                out += list(np.column_stack([px - r, py - r, px + r, py + r]))
            elif k == 'line':
                px, py = self._tx(a['x']), y0 + a['y'] * h if a['frac'] else self._ty(a['y'])
                if a['offset'] is not None:
                    px, py = px + a['offset'][:, 0], py + a['offset'][:, 1]
                pieces.append((px, py, pal.line_width if a['lw'] is None else a['lw']))
            elif k in ('hline', 'vline'):
                v = float(self._ty(a['v']) if k == 'hline' else self._tx(a['v']))
                ends = ([x0, x0 + w], [v, v]) if k == 'hline' else ([v, v], [y0, y0 + h])
                pieces.append((*map(np.array, ends), a['lw']))
            elif k == 'hsegs':
                for y, xa, xb in zip(self._ty(a['y']), self._tx(a['x0']), self._tx(a['x1'])):
                    pieces.append((np.array([xa, xb]), np.array([y, y]), a['lw']))
            elif k == 'vsegs':
                for x, ya, yb in zip(self._tx(a['x']), self._ty(a['y0']), self._ty(a['y1'])):
                    pieces.append((np.array([x, x]), np.array([ya, yb]), a['lw']))
            elif k == 'bars':
                for x, b, v, bw in zip(a['x'], a['b'], a['h'], a['w']):
                    xs, ys = (self._tx([b, b + v]), self._ty([x - bw / 2, x + bw / 2])) if a['horizontal'] else (
                        self._tx([x - bw / 2, x + bw / 2]), self._ty([self._base(b, self._yscale, self.ylim), b + v]))
                    out.append([min(xs), min(ys), max(xs), max(ys)])
            elif k in ('box', 'violin'):
                # Each group as one box around what it draws
                for x, stats, v, shape in zip(a['x'], a['stats'], a['values'], a.get('shapes', [None] * len(a['x']))):
                    if stats is None:
                        continue
                    half = a['w'] / 2 if shape is None else float(shape[1].max())
                    shown = v if k == 'violin' or a['offsets'] is not None or a['fliers'] else []
                    ys = [*([stats[0], stats[4]] if k == 'box' and a['box'] else []), *shown,
                          *(shape[0][[0, -1]] if shape is not None else [])]
                    xs, ys = self._tx([x - half, x + half]), self._ty([min(ys), max(ys)])
                    out.append([min(xs), min(ys), max(xs), max(ys)])
            elif k == 'rect':
                xs = [x0 + a['x'] * w, x0 + (a['x'] + a['w']) * w] if a['frac'] else self._tx([a['x'], a['x'] + a['w']])
                ys = [y0 + a['y'] * h, y0 + (a['y'] + a['h']) * h] if a['frac'] else self._ty([a['y'], a['y'] + a['h']])
                out.append([min(xs), min(ys), max(xs), max(ys)])
            elif k == 'text' and not a['rotation'] and math.isfinite(a['x'] + a['y']):
                px = x0 + a['x'] * w if a['frac'] else float(self._tx(a['x']))
                py = y0 + a['y'] * h if a['frac'] else float(self._ty(a['y']))
                size, bold = a['size'] or pal.tick_size, pal.text_bold if a['bold'] is None else a['bold']
                lines = str(a['s']).split('\n')
                # As text.draw sets them: the block's lift, then each line's baseline and its glyphs' reach
                lift = {'center': 0.5, 'bottom': 1.0}.get(a['va'], 0.0) * (len(lines) - 1)
                shift = {'baseline': 0.0, 'center': -text.CAP_HEIGHT * size / 2, 'top': -text.CAP_HEIGHT * size,
                         'bottom': 0.22 * size}[a['va']]
                with text.family(pal.font):
                    for i, line in enumerate(lines):
                        tw = text.width(line, size, bold)
                        ha = a.get('_ha', a['ha'])
                        left = px + a.get('_dx', a['dx']) - {'left': 0.0, 'center': tw / 2, 'right': tw}[ha]
                        base = py + a['dy'] - (i - lift) * size * a['linespacing'] + shift
                        out.append([left, base - 0.22 * size, left + tw, base + text.CAP_HEIGHT * size])
        for px, py, lw in pieces:
            for xa, ya, xb, yb in zip(px[:-1], py[:-1], px[1:], py[1:]):
                if np.isfinite([xa, ya, xb, yb]).all():
                    t = np.linspace(0.0, 1.0, max(2, int(math.hypot(xb - xa, yb - ya)) + 1))
                    sx, sy = xa + t * (xb - xa), ya + t * (yb - ya)
                    out += list(np.column_stack([sx - lw / 2, sy - lw / 2, sx + lw / 2, sy + lw / 2]))
        return np.array(out, dtype = float).reshape(-1, 4)

    def get_xlim(self):
        """The x limits as drawn, whether set or automatic."""
        self._range()
        return self.xlim

    def get_ylim(self):
        """The y limits as drawn, whether set or automatic."""
        self._range()
        return self.ylim

    def page(self, x, y):
        """Data positions in page points, from the page's lower left, once `Figure.layout` has placed the axes."""
        return self._tx(x), self._ty(y)

    def data(self, px, py):
        """Page points as data positions, the reverse of `page`."""
        x0, y0, w, h = self.rect
        out = []
        for v, start, size, (lo, hi), (forward, inverse) in ((px, x0, w, self.xlim, self._xwarp),
                                                              (py, y0, h, self.ylim, self._ywarp)):
            t = (np.asarray(v, dtype = float) - start) / size
            lo, hi = forward(np.float64(lo)), forward(np.float64(hi))
            out.append(inverse(lo + t * (hi - lo)))
        return tuple(out)

    # --- limits and ticks ---
    def _extent(self, axis):
        lo, hi = [], []
        sticky = []
        log = (self._xscale if axis == 'x' else self._yscale) == 'log'
        for a in self.artists:
            k = a['kind']
            if axis == 'x' and k in ('line', 'points', 'stems', 'band', 'errors', 'vsegs'):
                vals = a['x']
            elif axis == 'y' and k == 'line' and a['frac']:
                continue
            elif axis == 'y' and k in ('line', 'points', 'hsegs'):
                vals = a['y']
            elif axis == 'y' and k == 'vsegs':
                vals = np.r_[a['y0'], a['y1']]
            elif axis == 'x' and k == 'vspan':
                vals = np.array([a['x0'], a['x1']])
            elif axis == 'y' and k == 'hspan':
                vals = np.array([a['y0'], a['y1']])
            elif axis == 'y' and k == 'band':
                vals = np.r_[a['y1'], a['y2']]
            elif axis == 'y' and k == 'stems':
                vals = np.r_[a['y'], 0.0]
                sticky.append(0.0)
            elif axis == 'y' and k == 'errors':
                vals = np.r_[a['lo'], a['hi']]
            elif axis == 'x' and k == 'hsegs':
                vals = np.r_[a['x0'], a['x1']]
            elif k == 'arrow':
                vals = np.array([a[axis + '0'], a[axis + '1']])
            elif k == 'rect' and not a['frac']:
                vals = np.array([a[axis], a[axis] + a['w' if axis == 'x' else 'h']])
            elif k == 'contour':
                vals = a[axis]
            elif k in ('box', 'violin') and axis == 'x':
                vals = np.r_[a['x'] - a['w'] * 0.66, a['x'] + a['w'] * 0.66]
            elif k == 'box' and a['offsets'] is not None:
                vals = np.concatenate([np.zeros(0)] + a['values'])
            elif k == 'box':
                # Fliers the plot does not show take up no room in it
                vals = np.array([u for v in a['stats'] if v for u in (v[0], v[4], *(v[5] if a['fliers'] else ()))])
            elif k == 'violin':
                vals = np.concatenate([np.zeros(0)] + [v if s is None else s[0][[0, -1]]
                                                       for v, s in zip(a['values'], a['shapes'])])
            elif k == 'bars' and (axis == 'x') != a['horizontal']:
                # A little room beside the outer bars, in their own scale, as a histogram's can be of any
                vals = np.r_[a['x'] - a['w'] * 0.6, a['x'] + a['w'] * 0.6]
            elif k == 'bars':
                vals = np.r_[a['b'], a['b'] + a['h']]
                sticky.extend(np.unique(a['b']).tolist())
            elif k == 'hline' and axis == 'y' or k == 'vline' and axis == 'x':
                vals = np.array([a['v']])
            elif k == 'text' and not a['frac']:
                vals = np.array([a[axis]])
            else:
                continue
            vals = np.asarray(vals, dtype = float)
            # A log axis has no place for zero or below, where the base of a bar usually is
            vals = vals[np.isfinite(vals) & (vals > 0)] if log else vals[np.isfinite(vals)]
            if vals.size:
                lo.append(vals.min())
                hi.append(vals.max())
        return (min(lo), max(hi), sticky) if lo else None

    def _range(self):
        """The limits of both axes, as set or from the data, of all the axes they share an axis with."""
        for axis, scale, margin in (('x', self._xscale, self._xmargin), ('y', self._yscale, self._ymargin)):
            group = self._share[axis]
            found = [e for e in (ax._extent(axis) for ax in group) if e is not None]
            lo, hi, sticky = ((min(e[0] for e in found), max(e[1] for e in found), [v for e in found for v in e[2]])
                              if found else (0.0, 1.0, []))
            # Limits set on one of them, or turned around, hold for all
            fixed = next((v for v in [getattr(ax, '_' + axis + 'lim') for ax in [self] + group] if v is not None),
                         None)
            inverted = any(getattr(ax, '_' + axis + 'inv') for ax in group)
            if scale == 'log':
                # Nothing above zero leaves one decade to show
                lo = lo if lo > 0 else 1.0
                hi = hi if hi > lo else lo * 10
                span = 0.03 * math.log10(hi / lo)
                lo, hi = 10 ** (math.log10(lo) - span), 10 ** (math.log10(hi) + span)
            else:
                # The margin is a share of the axis as drawn, so measured after the scale's map
                forward, inverse = self._xwarp if axis == 'x' else self._ywarp
                ends = forward(np.array([lo, hi, *sticky], dtype = float))
                lo, hi, sticky = float(ends[0]), float(ends[1]), ends[2:]
                pad = (hi - lo) * margin if hi > lo else (abs(lo) * 0.1 or 1.0)
                # Bars of no height stick to both ends, and the axis still needs a range
                lo, hi = (lo if any(abs(lo - s) < 1e-12 for s in sticky) else lo - pad,
                          hi if hi > lo and any(abs(hi - s) < 1e-12 for s in sticky) else hi + pad)
                lo, hi = (float(v) for v in inverse(np.array([lo, hi])))
            if fixed is not None:
                # A log axis has no zero or negative end, so such a limit stays automatic
                lo, hi = [v if v > 0 or scale != 'log' else auto for v, auto in zip(fixed, (lo, hi))]
                pad = abs(lo) * 0.1 or 1.0
                lo, hi = (lo - pad, hi + pad) if lo == hi else (lo, hi)
            setattr(self, axis + 'lim', (hi, lo) if inverted else (lo, hi))

    def _limits(self):
        self._over = [0.0, 0.0]
        self._legend_size = None
        if self._cbar and self._cbar['source']._scale is None:
            raise ValueError('A colorbar shows the color scale of a heatmap or of scatter(c = ...), and these axes '
                             'have neither')
        # What fitting the labels turned or moved last time starts over, as limits or a size may have changed
        self.xrot = self._xrot
        for a in self.artists:
            if a['kind'] == 'text':
                a['_ha'], a['_dx'] = a['ha'], a['dx']
        self._range()
        for axis, ticks, labels, form in (('x', self._xticks, self._xtl, self._xfmt),
                                          ('y', self._yticks, self._ytl, self._yfmt)):
            cats = self._cats[axis]
            lo, hi = sorted(getattr(self, axis + 'lim'))
            if ticks is None and cats:
                ticks, labels = list(cats.values()), list(cats.keys())
            elif ticks is None and (self._xscale if axis == 'x' else self._yscale) == 'log':
                powers = range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
                ticks = [10.0 ** e for e in powers if lo <= 10.0 ** e <= hi]
                # Less than a decade holds too few powers of ten to read the axis by
                if len(ticks) < 2:
                    ticks = [m * 10.0 ** e for e in powers for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
                if len(ticks) < 2:
                    ticks = nice_ticks(lo, hi)
            elif ticks is None:
                ticks = nice_ticks(lo, hi)
            else:
                # Ticks set by hand can lie beyond limits that zooming narrowed
                shown = [i for i, v in enumerate(ticks) if lo - 1e-9 * (hi - lo) <= v <= hi + 1e-9 * (hi - lo)]
                ticks, labels = [ticks[i] for i in shown], None if labels is None else [labels[i] for i in shown]
            # Only numbers take the palette's minus sign, not labels given or categories
            numbers = labels is None
            if labels is None and form is not None:
                labels = [form(v) if callable(form) else form.format(v) for v in ticks]
            setattr(self, axis + 'ticks', ticks)
            labels = labels if labels is not None else tick_labels(ticks)
            # Ticks without labels, as beside a panel that shares them
            if self._tick_opts[axis].get('labels', True) is False or self._outer[axis]:
                labels = []
            setattr(self, axis + 'labels', _minus(self.pal, labels) if numbers else labels)

    def _tx(self, v):
        x0, y0, w, h = self.rect
        forward = self._xwarp[0]
        lo, hi = forward(np.float64(self.xlim[0])), forward(np.float64(self.xlim[1]))
        return x0 + (forward(np.asarray(v, dtype = float)) - lo) / (hi - lo) * w

    def _ty(self, v):
        x0, y0, w, h = self.rect
        forward = self._ywarp[0]
        lo, hi = forward(np.float64(self.ylim[0])), forward(np.float64(self.ylim[1]))
        return y0 + (forward(np.asarray(v, dtype = float)) - lo) / (hi - lo) * h

    # --- layout ---
    def _tick_sizes(self):
        """Tick label sizes for the x and y axes; a heatmap's labels are many, so they are smaller."""
        if self._tick_size is not None:
            sizes = self._tick_size, self._tick_size
        elif any(a['kind'] == 'heat' for a in self.artists):
            sizes = 8.2 * self.pal.spacing, 8.0 * self.pal.spacing
        else:
            sizes = self.pal.tick_size, self.pal.tick_size
        return tuple(self._tick_opts[axis].get('labelsize', v) for axis, v in zip('xy', sizes))

    def _spines(self):
        """The axis lines drawn, by the palette's choice unless set_spines changed it."""
        if 'spines' in self._hidden:
            return ()
        return tuple(s for s in ('left', 'bottom')
                     if self._spine_set.get(s, s in self.pal.spines and s not in self._hidden))

    def _tick_room(self, axis):
        """The spine's offset, the tick's length, and how far from the plot an axis's tick labels start, in points."""
        pal, spines = self.pal, self._spines()
        off = pal.spine_offset if spines else 0.0
        # Ticks ride their axis line, so a hidden line takes its ticks along
        drawn = {'x': 'bottom', 'y': 'left'}[axis] in spines and 'ticks' not in self._hidden
        tl = self._tick_opts[axis].get('length', pal.tick_length) if drawn else 0.0
        return off, tl, off + tl + self._tick_opts[axis].get('pad', 5.0 * pal.spacing)

    def _title_size(self):
        """A long title shrinks to fit the figure's width, down to a readable size; past that the page grows."""
        if self._title_fixed is not None:
            return self._title_fixed
        k = self.pal.spacing
        wide = text.width(self.title_text, 15.0 * k, self.pal.title_bold) or 1.0
        return max(9.0 * k, min(15.0 * k, 15.0 * k * (self.figure._base[0] - self.rect[0] - 3.0 * k) / wide))

    def _axis_name_widths(self, size):
        """How long the x and y axis names run at a size."""
        pal, track = self.pal, 0.9 if self.pal.caps else 0.0
        return [text.width(text.spaced(pal, s), size, pal.label_bold, track) for s in (self.xlabel, self.ylabel)]

    def _xtick_height(self):
        """How far the x tick labels reach below their start."""
        sx = self._tick_sizes()[0]
        if self.xrot:
            sin = abs(math.sin(math.radians(self.xrot)))
            return max((text.width(s, sx, self.pal.tick_bold) for s in self.xlabels), default = 0.0) * sin + sx * 0.4
        return sx * (1.0 + 1.25 * max(map(_extra_lines, self.xlabels), default = 0))

    def _ridge_width(self):
        """Room the ridges' names take left of the axes, or nothing without any."""
        pal = self.pal
        if not self._ridge_labels:
            return 0.0
        k = pal.spacing
        return 8.0 * k + max(text.width(text.spaced(pal, s), 7.8 * k if pal.caps else pal.tick_size + k,
                                        pal.text_bold, 0.9 if pal.caps else 0.0) for s in self._ridge_labels)

    def _ylabel_room(self):
        """Room the y axis name takes beside the tick labels."""
        size, pad = self._label_size['y'] or self.pal.label_size, self._label_pad['y']
        if not self.ylabel:
            return 0.0
        lines = 1.25 * _extra_lines(self.ylabel)
        # The name's baseline sits `pad` beyond the tick labels, and its glyphs a size beyond that
        # `pad` runs to the name's lowest glyph, which faces the tick labels: a descender, or deeper, a subscript
        low = 0.37 if any(rise < 0 for _, rise, _ in text.runs(self.ylabel)) else 0.22
        return size * (1.7 + lines) if pad is None else pad + size * (1.0 + lines + low)

    def _colorbar_width(self):
        """Room a colorbar at the right takes, past the tick labels: bar, its ticks, and its label."""
        source = self._cbar['source']
        ticks = nice_ticks(source._scale['lo'], source._scale['hi'], 5)
        label = self.pal.label_size * 1.6 if self._cbar['label'] else 0.0
        k = self.pal.spacing
        widest = max(text.width(s, 8.4 * k) for s in _minus(self.pal, tick_labels(ticks)))
        return 12.0 * k + COLORBAR_WIDTH * k + 5.0 * k + widest + label

    def _margins(self):
        """Room the decorations need around the plot area, in points: left, right, bottom, top."""
        pal = self.pal
        off, xtl, xpad = self._tick_room('x')
        _, ytl, ypad = self._tick_room('y')
        sy = self._tick_sizes()[1]
        yw = max((text.width(s, sy, pal.tick_bold) for s in self.ylabels), default = 0.0)
        k = pal.spacing
        left, right, bottom, top = 3.0 * k, 3.0 * k, 3.0 * k, 3.0 * k
        # Ticks left without labels still reach out from their axis line
        yside = ypad + yw if self.ylabels else off + ytl
        if self.yticks is not None and len(self.yticks):
            if self._yright:
                right += yside
            else:
                left += yside
        above = self._tick_opts['x'].get('top', False)
        if len(self.xticks) and above:
            bottom += off + xtl
            top += self._tick_opts['x'].get('pad', 5.0 * k) + self._xtick_height() if self.xlabels else 0.0
        elif len(self.xticks):
            bottom += xpad + self._xtick_height() if self.xlabels else off + xtl
        if self.ylabel:
            # A spine held away from the data takes room even without tick labels
            room = self._ylabel_room() + (0.0 if len(self.yticks) else off)
            if self._yright:
                right += room
            else:
                left += room
        if self.xlabel:
            size, gap = self._label_size['x'] or pal.label_size, self._label_pad['x']
            lines = 1.25 * _extra_lines(self.xlabel)
            bottom += size * (1.7 + lines) if gap is None else gap + size * (1.15 + lines)
            bottom += 0.0 if len(self.xticks) else off
        left += self._ridge_width()
        if self.title_text:
            pad = 8.0 * k if self._title_pad is None else self._title_pad
            top += 13.0 * k + pad + 1.25 * self._title_size() * _extra_lines(self.title_text)
        if self.legend_items and self._legend_opts in ('below', 'right'):
            tw, th = self._legend_size or self._legend_natural()
            if self._legend_opts == 'below':
                bottom += th + 8.0 * k
            else:
                right += tw + 18.0 * k
        if self._cbar and self._cbar['loc'] == 'right':
            right += self._colorbar_width()
        return left, right, bottom, top

    def _legend_entries(self):
        """Each item's text size, tracking, shown text, and width, in legend order."""
        pal = self.pal
        size = 7.4 * pal.spacing if pal.caps else pal.tick_size - 0.5 * pal.spacing
        track = 0.7 if pal.caps else 0.0
        out = []
        for s, rgb, kind, marker, area in self.legend_items:
            # Its rows are of one height
            shown = text.spaced(pal, s).replace('\n', ' ')
            header = kind == 'header'
            out.append((shown, rgb, kind, marker, area, text.width(shown, size, pal.text_bold or header, track)
                        + (0.0 if header else 18.0 * pal.spacing)))
        return size, track, out

    def _legend_natural(self):
        """Legend size before the axes are placed: one row or one column."""
        _, _, entries = self._legend_entries()
        widths, row = [e[5] for e in entries], LEGEND_ROW * self.pal.spacing
        return (max(widths), row * len(widths)) if self._legend_opts == 'right' else (sum(widths), row)

    def _legend_grid(self):
        """Offsets of the legend's items from its top left, wrapped to the axes, and the legend's total size."""
        _, _, w, h = self.rect
        _, _, entries = self._legend_entries()
        widths, k = [e[5] for e in entries], self.pal.spacing
        row = LEGEND_ROW * k
        if self._legend_opts == 'below':
            spots, x, y = [], 0.0, 0.0
            for v in widths:
                if x and x + v > w:
                    x, y = 0.0, y + row
                spots.append((x, y))
                x += v + 14.0 * k
            return spots, (w if y else sum(widths) + 14.0 * k * (len(widths) - 1), y + row)
        per = max(1, int(h // row)) if self._legend_opts == 'right' else len(widths)
        spots, offset, total_h = [], 0.0, 0.0
        for start in range(0, len(widths), per):
            for i in range(start, min(start + per, len(widths))):
                spots.append((offset, (i - start) * row))
            total_h = max(total_h, (min(start + per, len(widths)) - start) * row)
            offset += max(widths[start:start + per]) + 12.0 * k
        return spots, (offset - 12.0 * k, total_h)

    def _fit_labels(self):
        """Turn tick labels that would collide, thin those that would still collide, wrap the legend, and move or make
        room for annotations that would leave the page. True if anything changed."""
        pal = self.pal
        changed = False
        sx, sy = self._tick_sizes()
        if len(self.xticks) > 1 and self.xlabels and self._tick_opts['x'].get('fit', True):
            gap = _gap(self._tx(self.xticks))
            if not self.xrot and max(text.width(s, sx, pal.tick_bold) for s in self.xlabels) + 6.0 * pal.spacing > gap:
                self.xrot = 90.0
                changed = True
            if self.xrot and gap < sx * 1.05 and not self.show_all[0]:
                k = int(math.ceil(sx * 1.05 / gap))
                self.xticks, self.xlabels = self.xticks[::k], self.xlabels[::k]
                changed = True
        if len(self.yticks) > 1 and self.ylabels and not self.show_all[1] and self._tick_opts['y'].get('fit', True):
            gap = _gap(self._ty(self.yticks))
            if gap < sy * 1.05:
                k = int(math.ceil(sy * 1.05 / gap))
                self.yticks, self.ylabels = self.yticks[::k], self.ylabels[::k]
                changed = True
        if self.legend_items and self._legend_opts:
            size = self._legend_grid()[1]
            if self._legend_size is None or max(abs(u - v) for u, v in zip(size, self._legend_size)) > 0.5:
                self._legend_size = size
                changed = True
        page = self.figure.width
        over = [0.0, 0.0]
        # What the right side lacks for all but glycan labels, whose drawings replace them, for Figure._width_unlabeled
        self._over_plain = -math.inf
        if self.title_text:
            over[1] = self.rect[0] + text.width(self.title_text, self._title_size(), pal.title_bold) - (page - 3.0)
            self._over_plain = over[1]
        for a in self.artists:
            if a['kind'] != 'text' or a['rotation'] or a['frac'] or a['ha'] == 'center' or not math.isfinite(a['x']):
                continue
            tw = text.width(a['s'], a['size'] or pal.tick_size, pal.text_bold if a['bold'] is None else a['bold'])
            px, gap = float(self._tx(a['x'])), abs(a['dx']) + 3.0
            over_right, over_left = px + gap + tw - (page - 3.0), tw + gap - (px - 3.0)
            natural_right = a['ha'] == 'left'
            # The label's own side if it fits, else the other side if that fits, else whichever side overflows less. A
            # glycan drawing always grows rightward from its label's anchor, so such a label stays and the page grows,
            # as does text placed with `text`, where its caller put it.
            if a['tag'] or a.get('placed') or (over_right if natural_right else over_left) <= 0:
                use_right = natural_right
            elif (over_left if natural_right else over_right) <= 0:
                use_right = not natural_right
            else:
                use_right = over_right <= over_left
            a['_ha'] = 'left' if use_right else 'right'
            a['_dx'] = a['dx'] if use_right == natural_right else (gap if use_right else -gap)
            over[1 if use_right else 0] = max(over[1 if use_right else 0], over_right if use_right else over_left)
            if use_right and not a['tag']:
                self._over_plain = max(self._over_plain, over_right)
        # A page held to a window's size has nowhere to grow
        for side in () if self.figure._held else (0, 1):
            if over[side] > 0.5:
                self._over[side] += over[side]
                changed = True
        return changed

    # --- drawing ---
    def _draw_marks(self, c):
        x0, y0, w, h = self.rect
        c.saveState()
        for a in sorted(self.artists, key = lambda r: (r['z'], r['order'])):
            k = a['kind']
            n = len(c.ops)
            # Color 'none' draws nothing but an edge, while a dendrogram's None means the palette's choice
            if k == 'text' or (k != 'dendro' and a.get('rgb', k) is None and not a.get('edge')):
                continue
            pad = 9.0 if k in ('points', 'box', 'violin') else 0.0 if k == 'heat' else 1.0
            # A named element stays a bare path, for the code that looks for it in the SVG
            gid = a.get('gid')
            free = gid or k == 'image' or not a.get('clip', True)
            c.state['clip'] = None if free else [((x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad), c.state['ctm'])]
            getattr(self, '_d_' + k)(c, a)
            for op in c.ops[n:]:
                op['gid'] = gid
                if k not in ('hline', 'vline', 'vspan', 'hspan', 'image', 'contour', 'aline', 'rect'):
                    op['mark'] = True
        c.restoreState()

    def _in_x(self, v):
        """Whether x values lie within the x limits, as what may leave the axes must once zooming has left it behind."""
        lo, hi = sorted(self.xlim)
        return bool(np.all((np.asarray(v) >= lo) & (np.asarray(v) <= hi)))

    def _d_line(self, c, a):
        mask = np.isfinite(a['x']) & np.isfinite(a['y'])
        # A log axis has no place for zero or below, so those points break the line too
        if self._xscale == 'log':
            mask &= a['x'] > 0
        if self._yscale == 'log' and not a['frac']:
            mask &= a['y'] > 0
        if not mask.any() or (not a['clip'] and not self._in_x(a['x'][mask])):
            return
        _, y0, _, h = self.rect
        xs, ys = self._tx(a['x']), y0 + a['y'] * h if a['frac'] else self._ty(a['y'])
        if a['offset'] is not None:
            xs, ys = xs + a['offset'][:, 0], ys + a['offset'][:, 1]
        for run in _runs(mask):
            if run.stop - run.start > 1:
                marks.line(c, self.pal, xs[run], ys[run], a['rgb'], a['lw'], a['alpha'], a['dash'])

    def _d_vsegs(self, c, a):
        by_color = {}
        for x, lo, hi, rgb in zip(self._tx(a['x']), self._ty(a['y0']), self._ty(a['y1']), a['rgb']):
            if rgb is not None and np.isfinite((x, lo, hi)).all():
                by_color.setdefault(rgb, []).append(((x, x), (lo, hi)))
        for rgb, lines in by_color.items():
            marks.polylines(c, lines, rgb, a['lw'], a['alpha'], cap = 0, dash = a['dash'])

    def _d_vspan(self, c, a):
        _, y0, _, h = self.rect
        x0, x1 = float(self._tx(a['x0'])), float(self._tx(a['x1']))
        if math.isfinite(x0 + x1):
            marks.polygon(c, [x0, x1, x1, x0], [y0, y0, y0 + h, y0 + h], a['rgb'], a['alpha'])

    def _d_hspan(self, c, a):
        x0, _, w, _ = self.rect
        y0, y1 = float(self._ty(a['y0'])), float(self._ty(a['y1']))
        if math.isfinite(y0 + y1):
            marks.polygon(c, [x0, x0 + w, x0 + w, x0], [y0, y0, y1, y1], a['rgb'], a['alpha'])

    def _d_image(self, c, a):
        _, y0, _, h = self.rect
        lo, hi = sorted(self.ylim)
        if not self._in_x(a['x']) or (not a['frac'] and not lo <= a['y'] <= hi):
            return
        cy = y0 + a['y'] * h if a['frac'] else float(self._ty(a['y']))
        op = dict(c.state)
        op.update(kind = 'image', w = a['nx'], h = a['ny'], rgb = a['rgb'], alpha = a['alpha'],
                  ctm = (a['w'], 0.0, 0.0, a['h'], float(self._tx(a['x'])) + a['dx'] - a['w'] / 2,
                         cy + a['dy'] - a['h'] / 2))
        c.ops.append(op)

    def _d_points(self, c, a):
        mask = np.isfinite(a['x']) & np.isfinite(a['y'])
        # A point beyond the limits is not drawn, while one at the edge is drawn whole, which the clip's margin allows
        for v, (lo, hi) in ((a['x'], sorted(self.xlim)), (a['y'], sorted(self.ylim))):
            slack = 1e-9 * (hi - lo)
            mask &= (v >= lo - slack) & (v <= hi + slack)
        s = a['s'] if np.ndim(a['s']) == 0 else np.asarray(a['s'])[mask]
        marks.points(c, self.pal, self._tx(a['x'][mask]), self._ty(a['y'][mask]), s, a['rgb'], a['alpha'], a['marker'],
                     a['edge'])

    def _d_band(self, c, a):
        for run in _runs(np.isfinite(a['x']) & np.isfinite(a['y1']) & np.isfinite(a['y2'])):
            x, y1, y2 = a['x'][run], a['y1'][run], a['y2'][run]
            if len(x) > 1:
                marks.polygon(c, np.r_[self._tx(x), self._tx(x[::-1])], np.r_[self._ty(y1), self._ty(y2[::-1])],
                              a['rgb'], a['alpha'])

    def _base(self, v, scale, limits):
        """Where a bar or a stem starts: a log axis has no place for zero or below, so the start of the axis."""
        return v if scale != 'log' or v > 0 else min(limits)

    def _d_bars(self, c, a):
        room = np.full(a['x'].shape, np.inf)
        if self.pal.bar == 'needle' and a['x'].size > 1:
            # A needle's wide top stays clear of the bars beside it, as those of a histogram touch
            order = np.argsort(a['x'])
            free = (np.diff(a['x'][order]) - (a['w'][order][:-1] + a['w'][order][1:]) / 2) / 2
            room[order[:-1]] = np.maximum(free, 0.0)
            room[order[1:]] = np.fmin(room[order[1:]], np.maximum(free, 0.0))
        for x, b, v, w, rgb, free in zip(a['x'], a['b'], a['h'], a['w'], a['rgb'], room):
            if rgb is None or not np.isfinite((x, b, v, w)).all():
                continue
            if a['horizontal']:
                x1, x2 = self._tx(self._base(b, self._xscale, self.xlim)), self._tx(b + v)
                y1, y2 = self._ty(x - w / 2), self._ty(x + w / 2)
            else:
                x1, x2 = self._tx(x - w / 2), self._tx(x + w / 2)
                y1, y2 = self._ty(self._base(b, self._yscale, self.ylim)), self._ty(b + v)
            lx, ly, bw, bh = float(min(x1, x2)), float(min(y1, y2)), float(abs(x2 - x1)), float(abs(y2 - y1))
            if a['stacked']:
                marks.segment(c, self.pal, lx, ly, bw, bh, rgb, 0.0 if b == 0 else max(self.pal.heat_gap, 0.8),
                              a['alpha'])
            else:
                # Its base lies at the far end if it runs down or left on the page
                marks.bar(c, self.pal, lx, ly, bw, bh, rgb, a['horizontal'], a['alpha'],
                          x2 < x1 if a['horizontal'] else y2 < y1, min(0.7, free / w) if w else 0.0)

    def _d_errors(self, c, a):
        for x, lo, hi in zip(self._tx(a['x']), self._ty(a['lo']), self._ty(a['hi'])):
            if np.isfinite((x, lo, hi)).all():
                marks.polyline(c, [x, x], [lo, hi], a['rgb'], a['lw'], cap = 0)
                for v in (lo, hi):
                    marks.polyline(c, [x - a['cap'], x + a['cap']], [v, v], a['rgb'], a['lw'], cap = 0)

    def _d_contour(self, c, a):
        if a['lines']:
            marks.polylines(c, [(self._tx(xs), self._ty(ys)) for xs, ys in a['lines']], a['rgb'], a['lw'], a['alpha'],
                            dash = a['dash'])

    def _paired(self, rgb):
        """A group's mark in a paler shade under points in its own color, or, where its color is the palest already, in
        that color under points a shade darker."""
        dark, _, light = self.pal.shades(rgb)
        return (light, rgb) if light != rgb else (rgb, dark)

    def _group_points(self, c, x, v, offsets, size, rgb, alpha):
        """A group's values as points at their spread positions, leaving out those beyond the limits."""
        px, (lo, hi) = x + offsets, sorted(self.ylim)
        keep = (v >= lo) & (v <= hi) & (px >= min(self.xlim)) & (px <= max(self.xlim))
        marks.points(c, self.pal, self._tx(px[keep]), self._ty(v[keep]), size, rgb, alpha)

    def _d_box(self, c, a):
        dots = a['offsets'] is not None
        for i, (x, stats, rgb) in enumerate(zip(a['x'], a['stats'], a['rgb'])):
            if stats is None:
                continue
            x0, x1 = float(self._tx(x - a['w'] / 2)), float(self._tx(x + a['w'] / 2))
            fill, ink = self._paired(rgb) if dots and a['box'] else (rgb, rgb)
            if a['box']:
                marks.box(c, self.pal, min(x0, x1), abs(x1 - x0), [float(self._ty(v)) for v in stats[:5]], fill,
                          a['alpha'])
            if dots:
                self._group_points(c, x, a['values'][i], a['offsets'][i], a['sizes'][i], ink, a['alpha'])
            elif a['fliers'] and len(stats[5]):
                marks.points(c, self.pal, np.full(len(stats[5]), (x0 + x1) / 2), self._ty(stats[5]), 12.0, rgb,
                             a['alpha'])

    def _d_violin(self, c, a):
        pal = self.pal
        for i, (x, stats, shape, rgb) in enumerate(zip(a['x'], a['stats'], a['shapes'], a['rgb'])):
            if stats is None:
                continue
            dots = a['inner'] == 'points'
            fill, ink = self._paired(rgb) if dots else (rgb, rgb)
            cx = float(self._tx(x))
            if shape is not None:
                grid, half = shape
                ys = self._ty(grid)
                left, right = self._tx(x - half), self._tx(x + half)
                marks.violin(c, pal, np.r_[left, right[::-1]], np.r_[ys, ys[::-1]], cx,
                             float(np.max(np.abs(right - left))) / 2, fill, a['alpha'])
            if dots or shape is None:
                self._group_points(c, x, a['values'][i], a['offsets'][i], a['sizes'][i], ink, a['alpha'])
            elif a['inner'] == 'box':
                low, q1, median, q3, high = (float(self._ty(v)) for v in stats[:5])
                span = abs(float(self._tx(x + a['w'] / 2) - self._tx(x - a['w'] / 2)))
                marks.inner_box(c, pal, cx, min(max(span * 0.08, 2.5), 7.0), (low, q1, median, q3, high), rgb,
                                a['alpha'])

    def _d_stems(self, c, a):
        pal = self.pal
        keep = np.isfinite(a['x']) & np.isfinite(a['y'])
        xs, ys = self._tx(a['x'][keep]), self._ty(a['y'][keep])
        zero = float(self._ty(self._base(0.0, self._yscale, self.ylim)))
        scale = {'needle': 0.35, 'flat': 1.6}.get(pal.bar, 1.0)
        for x, y in zip(xs, ys):
            if pal.bar in ('deep', 'soft'):
                dark, base, light = pal.shades(a['rgb'])
                hw = a['lw'] / 2.0
                grad = ((x, zero, x, y), (dark, light) if pal.bar == 'deep' else (light, base))
                marks.polygon(c, [x - hw, x + hw, x + hw, x - hw], [zero, zero, y, y], None, 1.0, grad)
            else:
                marks.polyline(c, [x, x], [zero, y], a['rgb'], a['lw'] * scale, cap = 0)
        if a['cap']:
            marks.points(c, pal, xs, ys, 26.0, a['rgb'])

    def _d_hsegs(self, c, a):
        for y, x0, x1 in zip(self._ty(a['y']), self._tx(a['x0']), self._tx(a['x1'])):
            if np.isfinite((y, x0, x1)).all():
                marks.line(c, self.pal, [x0, x1], [y, y], a['rgb'], a['lw'], a['alpha'], a['dash'])

    def _d_arrow(self, c, a):
        x0, y0 = float(self._tx(a['x0'])), float(self._ty(a['y0']))
        x1, y1 = float(self._tx(a['x1'])), float(self._ty(a['y1']))
        n = math.hypot(x1 - x0, y1 - y0)
        if not math.isfinite(x0 + y0 + x1 + y1) or n <= sum(a['shrink']):
            return
        (s0, s1), ux, uy = a['shrink'], (x1 - x0) / n, (y1 - y0) / n
        marks.arrow(c, self.pal, x0 + ux * s0, y0 + uy * s0, x1 - ux * s1, y1 - uy * s1, a['rgb'], a['lw'], a['alpha'],
                    a['head'], a['bend'])

    def _d_aline(self, c, a):
        x0, y0, w, h = self.rect
        px0, py0 = float(self._tx(a['x0'])), float(self._ty(a['y0']))
        dx, dy = float(self._tx(a['x1'])) - px0, float(self._ty(a['y1'])) - py0
        if not math.isfinite(px0 + py0 + dx + dy) or not (dx or dy):
            return
        # Far enough each way to cross the whole axes, which the clip then trims
        reach = 2.0 * math.hypot(w, h) / math.hypot(dx, dy)
        marks.polyline(c, [px0 - reach * dx, px0 + reach * dx], [py0 - reach * dy, py0 + reach * dy], a['rgb'],
                       a['lw'], cap = 0, dash = a['dash'])

    def _d_rect(self, c, a):
        x0, y0, w, h = self.rect
        if a['frac']:
            xs, ys = [x0 + a['x'] * w, x0 + (a['x'] + a['w']) * w], [y0 + a['y'] * h, y0 + (a['y'] + a['h']) * h]
        else:
            xs = [float(self._tx(a['x'])), float(self._tx(a['x'] + a['w']))]
            ys = [float(self._ty(a['y'])), float(self._ty(a['y'] + a['h']))]
        if not math.isfinite(sum(xs) + sum(ys)):
            return
        (lx, hx), (ly, hy) = sorted(xs), sorted(ys)
        r = max(0.0, min(a['radius'], (hx - lx) / 2, (hy - ly) / 2))
        q = r * (1 - marks.KAPPA)
        p = c.beginPath()
        p.moveTo(lx + r, ly)
        # Each side, then the quarter circle into the next one
        for start, c1, c2, end in (((hx - r, ly), (hx - q, ly), (hx, ly + q), (hx, ly + r)),
                                   ((hx, hy - r), (hx, hy - q), (hx - q, hy), (hx - r, hy)),
                                   ((lx + r, hy), (lx + q, hy), (lx, hy - q), (lx, hy - r)),
                                   ((lx, ly + r), (lx, ly + q), (lx + q, ly), (lx + r, ly))):
            p.lineTo(*start)
            if r:
                p.curveTo(*c1, *c2, *end)
        p.close()
        edge = a['edge'] or (None, None)
        c.saveState()
        marks._style(c, a['rgb'], a['alpha'], edge[0], edge[1], a['alpha'], join = 1)
        c.drawPath(p, fill = a['rgb'] is not None, stroke = a['edge'] is not None)
        c.restoreState()

    def _d_hline(self, c, a):
        x0, _, w, _ = self.rect
        y = float(self._ty(a['v']))
        if math.isfinite(y):
            marks.polyline(c, [x0, x0 + w], [y, y], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_vline(self, c, a):
        _, y0, _, h = self.rect
        x = float(self._tx(a['v']))
        if math.isfinite(x):
            marks.polyline(c, [x, x], [y0, y0 + h], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_heat(self, c, a):
        ny, nx = a['m'].shape
        # Through the limits, as they decide which way up the cells are and which of them show
        xs, ys = self._tx(np.arange(nx + 1)), self._ty(ny - np.arange(ny + 1))
        if nx * ny > HEAT_IMAGE_CELLS:
            op = dict(c.state)
            px = (a['rgb'] * 255.0 + 0.5).astype(np.uint8)
            seen = (np.isfinite(a['m']) * 255).astype(np.uint8)
            rows, columns = -1 if ys[-1] > ys[0] else 1, -1 if xs[-1] < xs[0] else 1
            op.update(kind = 'image', w = nx, h = ny, rgb = px[::rows, ::columns].tobytes(),
                      ctm = (float(abs(xs[-1] - xs[0])), 0.0, 0.0, float(abs(ys[-1] - ys[0])),
                             float(min(xs[0], xs[-1])), float(min(ys[0], ys[-1]))),
                      alpha = seen[::rows, ::columns].tobytes())
            c.ops.append(op)
            return
        cw, ch = float(abs(xs[1] - xs[0])), float(abs(ys[1] - ys[0]))
        for r in range(ny):
            for j in range(nx):
                if np.isfinite(a['m'][r, j]):
                    marks.heat_cell(c, self.pal, float(min(xs[j], xs[j + 1])), float(min(ys[r], ys[r + 1])), cw, ch,
                                    tuple(a['rgb'][r, j]))

    def _d_ridges(self, c, a):
        pal = self.pal
        x = a['x']
        n = len(a['curves'])
        tx = self._tx(x)
        for i, y in enumerate(a['curves']):
            base = n - 1 - i
            t = 0.12 + 0.88 * i / max(n - 1, 1)
            rgb = tuple(pal.sequential(t))
            ys = self._ty(base + y * a['overlap'])
            top = tuple(pal.sequential(max(0.0, t - 0.3)))
            if pal.ridge == 'tint':
                top = mix(pal.color('a2'), 0.14 + 0.62 * i / max(n - 1, 1), pal.color('paper'))
                rgb = pal.color('a0')
            peak = float(self._ty(base + np.nanmax(y) * a['overlap']) - self._ty(base))
            marks.ridge(c, pal, tx, float(self._ty(base)), ys, rgb, top, peak)

    def _d_dendro(self, c, a):
        pal = self.pal
        lw = {'deep': 1.2, 'needle': 0.45, 'flat': 2.6, 'soft': 1.5}[pal.bar]
        rgb = {'deep': pal.color('a1'), 'needle': pal.color('ink'), 'flat': pal.color('a0'),
               'soft': pal.color('a2')}[pal.bar]
        lines = {}
        for k, (ic, dc) in enumerate(zip(a['ic'], a['dc'])):
            pos, dep = np.asarray(ic) / 10.0, np.asarray(dc, dtype = float)
            lines.setdefault(rgb if a['rgb'] is None else a['rgb'][k], []).append(
                (self._tx(dep), self._ty(pos)) if a['orient'] == 'left' else (self._tx(pos), self._ty(dep)))
        for color, group in lines.items():
            marks.polylines(c, group, color, lw)

    def _draw_axes(self, c):
        pal = self.pal
        x0, y0, w, h = self.rect
        ink = pal.color(pal.spine_color or pal.ink)
        if self.grid_on and 'grid' not in self._hidden:
            for v in self.yticks:
                y = float(self._ty(v))
                marks.polyline(c, [x0, x0 + w], [y, y], pal.color(pal.faint), 1.1, cap = 0)
        off = pal.spine_offset
        cap = 1 if pal.spine_round else 0
        spines = self._spines()
        xt, yt = [float(self._tx(v)) for v in self.xticks], [float(self._ty(v)) for v in self.yticks]
        if 'left' in spines:
            tl = self._tick_room('y')[1]
            lo, hi = (min(yt), max(yt)) if off and yt else (y0, y0 + h)
            marks.polyline(c, [x0 - off, x0 - off], [lo, hi], ink, pal.spine_width, cap = cap)
            for y in yt if tl else ():
                marks.polyline(c, [x0 - off - tl, x0 - off], [y, y], ink, pal.spine_width, cap = 0)
        if 'bottom' in spines:
            tl = self._tick_room('x')[1]
            lo, hi = (min(xt), max(xt)) if off and xt else (x0, x0 + w)
            marks.polyline(c, [lo, hi], [y0 - off, y0 - off], ink, pal.spine_width, cap = cap)
            for x in xt if tl else ():
                marks.polyline(c, [x, x], [y0 - off - tl, y0 - off], ink, pal.spine_width, cap = 0)

    def _draw_text(self, c):
        pal = self.pal
        x0, y0, w, h = self.rect
        tick_rgb = pal.color(pal.tick_color or pal.muted)
        off, xtl, pad = self._tick_room('x')
        _, ytl, ypad = self._tick_room('y')
        size = pal.tick_size
        heat = 'heat' in {a['kind'] for a in self.artists}
        small, ytext = self._tick_sizes()
        ink = pal.color(pal.ink)
        above = self._tick_opts['x'].get('top', False)
        rise = y0 + h + self._tick_opts['x'].get('pad', 5.0 * pal.spacing)
        for v, s in zip(self.xticks, self.xlabels):
            x = float(self._tx(v))
            rgb, bold = ((pal.color(self._xemph[s]), True) if s in self._xemph else
                         (ink if heat and self.xrot else tick_rgb, pal.tick_bold))
            if self.xrot and above:
                # The label starts at its tick and runs up and away from the axes
                text.draw(c, x - small * 0.36 * math.sin(math.radians(self.xrot)), rise, s, small, rgb, bold,
                          ha = 'left' if self.xrot > 0 else 'right', rotation = self.xrot, va = 'baseline')
            elif self.xrot:
                # The label's end sits at its tick, and it hangs below the axis either way
                text.draw(c, x + small * 0.36 * math.sin(math.radians(self.xrot)), y0 - pad, s, small, rgb, bold,
                          ha = 'right' if self.xrot > 0 else 'left', rotation = self.xrot, va = 'baseline')
            else:
                text.draw(c, x, rise if above else y0 - pad, s, small, rgb, bold, ha = 'center',
                          va = 'bottom' if above else 'top')
        yw = 0.0
        for v, s in zip(self.yticks, self.ylabels):
            y = float(self._ty(v))
            yw = max(yw, text.width(s, ytext, pal.tick_bold))
            rgb, bold = ((pal.color(self._yemph[s]), True) if s in self._yemph else
                         (ink if heat else tick_rgb, pal.tick_bold))
            if self._yright:
                text.draw(c, x0 + w + ypad - 2.0 * pal.spacing, y, s, ytext, rgb, bold, ha = 'left', va = 'center')
            else:
                text.draw(c, x0 - ypad, y, s, ytext, rgb, bold, ha = 'right', va = 'center')
        label_rgb = pal.color(pal.label_color or pal.ink)
        track = 0.9 if pal.caps else 0.0
        # Long axis names shrink to stay within their cell, centered on the axes
        rx, ry, rw, rh = self._room
        fits = (2 * min(x0 + w / 2 - rx, rx + rw - x0 - w / 2), 2 * min(y0 + h / 2 - ry, ry + rh - y0 - h / 2))
        base = [self._label_size[axis] or pal.label_size for axis in 'xy']
        size_x, size_y = [max(min(LABEL_MIN, given), min(given, given * room / wide)) if wide else given
                          for room, wide, given in zip(fits, [self._axis_name_widths(v)[i] for i, v in enumerate(base)],
                                                       base)]
        if self.xlabel:
            drop = (pad + self._xtick_height() if self.xlabels and not above else off + xtl if len(self.xticks) else
                    off)
            gap = base[0] * 0.55 if self._label_pad['x'] is None else self._label_pad['x']
            text.draw(c, x0 + w / 2, y0 - drop - gap, text.spaced(pal, self.xlabel), size_x,
                      label_rgb, pal.label_bold, ha = 'center', va = 'top', tracking = track)
        if self.ylabel:
            beside = ypad + yw if self.ylabels else off + ytl if len(self.yticks) else off
            # Further lines run away from the axes on the right, and towards them on the left
            ylx = (x0 + w + beside + base[1] * 1.3 if self._yright else
                   x0 - beside - self._ridge_width() - self._ylabel_room() + base[1])
            text.draw(c, ylx, y0 + h / 2, text.spaced(pal, self.ylabel), size_y, label_rgb, pal.label_bold,
                      ha = 'center', va = 'baseline', rotation = 90.0, tracking = track)
        if self.title_text:
            title_size = self._title_size()
            lift = rise - y0 - h + self._xtick_height() if above and self.xlabels and len(self.xticks) else 0.0
            pad = 8.0 * pal.spacing if self._title_pad is None else self._title_pad
            text.draw(c, x0, y0 + h + lift + pad + 1.25 * title_size * _extra_lines(self.title_text),
                      self.title_text, title_size, pal.color(pal.title_color or pal.ink), pal.title_bold, ha = 'left')
        for a in self.artists:
            if a['kind'] == 'ridges' and a['labels'] is not None:
                n = len(a['curves'])
                for i, s in enumerate(self._ridge_labels):
                    t = text.spaced(pal, s)
                    tsize = 7.8 * pal.spacing if pal.caps else size + pal.spacing
                    text.draw(c, x0 - 8.0 * pal.spacing, float(self._ty(n - 1 - i)) + pal.spacing, t, tsize, ink,
                              pal.text_bold,
                              ha = 'right', tracking = track)
        lo, hi = sorted(self.ylim)
        texts = [a for a in sorted((r for r in self.artists if r['kind'] == 'text'), key = lambda r: r['order'])
                 if math.isfinite(a['x'] + a['y'])
                 and not (a['clip'] and not (self._in_x(a['x']) and lo <= a['y'] <= hi))]
        placed, groups = {}, []
        for side in ('left', 'right'):
            # Only labels whose spans across the page overlap can collide, so each run of those moves on its own
            spans = []
            for a in (a for a in texts if a['spread'] and a['_ha'] == side):
                start = float(self._tx(a['x'])) + a['_dx']
                tw = text.width(a['s'], a['size'] or size, pal.text_bold if a['bold'] is None else a['bold'])
                spans.append((start, start + tw, a) if side == 'left' else (start - tw, start, a))
            for start, end, a in sorted(spans, key = lambda v: v[0]):
                if groups and groups[-1][0] == side and start < groups[-1][1]:
                    groups[-1][1] = max(groups[-1][1], end)
                    groups[-1][2].append(a)
                else:
                    groups.append([side, end, [a]])
        for _, _, group in groups:
            want = np.array([float(self._ty(a['y'])) + a['dy'] for a in group])
            gap = max((1.25 + a['linespacing'] * _extra_lines(a['s'])) * (a['size'] or size) for a in group)
            order = np.argsort(want, kind = 'stable')
            # Closest to where they belong under a least gap: shifted down by their rank's gaps, the positions only have
            # to rise, which pooling adjacent violators solves exactly, and then the axes' height bounds all alike
            blocks = []
            for v in want[order] - gap * np.arange(len(group)):
                blocks.append([v, 1])
                while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                    total, count = blocks.pop()
                    blocks[-1][0] += total
                    blocks[-1][1] += count
            fit = np.concatenate([np.full(count, total / count) for total, count in blocks])
            floor = y0 + gap / 2
            fit = np.clip(fit, floor, max(floor, y0 + h - gap / 2 - gap * (len(group) - 1)))
            placed.update((id(group[i]), (y, gap)) for i, y in zip(order, fit + gap * np.arange(len(group))))
        for a in texts:
            bold = pal.text_bold if a['bold'] is None else a['bold']
            px = x0 + a['x'] * w if a['frac'] else float(self._tx(a['x']))
            py = y0 + a['y'] * h if a['frac'] else float(self._ty(a['y']))
            ty, gap = placed.get(id(a), (py + a['dy'], 0.0))
            # Only a spread label that had to move gets a leader
            if id(a) in placed and abs(ty - py - a['dy']) > 0.3 * gap:
                way = 1.0 if a['_ha'] == 'left' else -1.0
                k = pal.spacing
                marks.polyline(c, [px + way * 2.0 * k, px + a['_dx'] - way * 1.5 * k], [py, ty], a['rgb'], 0.5 * k, 0.6,
                               cap = 0)
            n = len(c.ops)
            text.draw(c, px + a['_dx'], ty, a['s'], a['size'] or size, a['rgb'], bold, ha = a['_ha'],
                      va = a['va'], rotation = a['rotation'], halo = a['halo'], halo_color = pal.color('paper'),
                      linespacing = a['linespacing'])
            if a['tag']:
                for op in c.ops[n:]:
                    op['label'] = (a['order'], id(self), a['s'])
        # Right of the axes, in order: tick labels, the axis name, the legend, and the color bar
        right = 0.0
        if self._yright:
            right = (yw + ypad if self.ylabels else off + ytl if len(self.yticks) else 0.0) + self._ylabel_room()
        self._draw_legend(c, right)
        if self._cbar:
            self._draw_colorbar(c, right)

    def _draw_colorbar(self, c, right):
        pal = self.pal
        k, bar = pal.spacing, COLORBAR_WIDTH * pal.spacing
        x0, y0, w, h = self.rect
        scale = self._cbar['source']._scale
        lo, hi = scale['lo'], scale['hi']
        # One value makes a scale of no length, whose only tick sits at its start
        span = hi - lo or 1.0
        stops = [tuple(v) for v in getattr(pal, scale['cmap'])(np.linspace(0.0, 1.0, 9))]
        ticks = [t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi and (t == int(t) or not self._cbar['integer'])]
        ink, label = pal.color(pal.ink), self._cbar['label']
        if self._cbar['loc'] == 'fill':
            bh = min(9.0 * k, h * 0.3)
            bw, by = w * 0.8, y0 + h * 0.34
            marks.polygon(c, [x0, x0 + bw, x0 + bw, x0], [by, by, by + bh, by + bh], None, 1.0,
                          ((x0, by, x0 + bw, by), stops))
            for t, s in zip(ticks, _minus(pal, tick_labels(ticks))):
                tx = x0 + (t - lo) / span * bw
                text.draw(c, tx, by - 3.0 * k, s, 7.4 * k, ink, ha = 'center', va = 'top')
            if label:
                text.draw(c, x0, by + bh + 4.0 * k, text.spaced(pal, label), (7.8 if pal.caps else 8.6) * k, ink,
                          ha = 'left')
            return
        right += self._legend_grid()[1][0] + 18.0 * k if self.legend_items and self._legend_opts == 'right' else 0.0
        bx = x0 + w + right + 12.0 * k
        bh = min(h * 0.6, 170.0 * k)
        by = y0 + h - bh
        marks.polygon(c, [bx, bx + bar, bx + bar, bx], [by, by, by + bh, by + bh], None, 1.0,
                      ((bx, by, bx, by + bh), stops))
        widest = 0.0
        for t, s in zip(ticks, _minus(pal, tick_labels(ticks))):
            text.draw(c, bx + bar + 5.0 * k, by + (t - lo) / span * bh, s, 8.4 * k, ink, va = 'center')
            widest = max(widest, text.width(s, 8.4 * k))
        if label:
            text.draw(c, bx + bar + 5.0 * k + widest + pal.label_size * 1.3, by + bh / 2,
                      text.spaced(pal, label), pal.label_size, pal.color(pal.label_color or pal.ink), pal.label_bold,
                      ha = 'center', va = 'baseline', rotation = 90.0)

    def _draw_legend(self, c, right):
        pal = self.pal
        if not self.legend_items or self._legend_opts is None:
            return
        u, row = pal.spacing, LEGEND_ROW * pal.spacing
        x0, y0, w, h = self.rect
        size, track, entries = self._legend_entries()
        spots, (tw, th) = self._legend_grid()
        loc = self._legend_opts
        if loc == 'right':
            left, top = x0 + w + right + 16.0 * u, y0 + h - 8.0 * u
        elif loc == 'below':
            rows = round(th / row)
            left = x0 if tw >= w else x0 + (w - tw) / 2
            top = (y0 - 40.0 * u if self._legend_y is None else self._legend_y) + (rows - 1) * row
        else:
            left = x0 + w - tw - 8.0 * u if 'right' in loc else x0 + 10.0 * u
            top = y0 + h - 12.0 * u if 'upper' in loc else y0 + 12.0 * u + th - row
        for (dx, dy), (s, rgb, k, marker, area, _) in zip(spots, entries):
            x, y = left + dx, top - dy
            if k == 'header':
                text.draw(c, x, y, s, size, pal.color(pal.ink), True, va = 'center', tracking = track)
                continue
            if k == 'line':
                marks.polyline(c, [x, x + 12.0 * u], [y, y], rgb, max(pal.line_width, 1.2 * u), cap = 0, dash = marker)
            elif k == 'point':
                # As large as in the plot, up to what a legend row can hold
                marks.points(c, pal, [x + 6.0 * u], [y], min(area, 196.0 * u * u / pal.point_scale), rgb,
                             marker = marker)
            else:
                lo, hi = y - 4.5 * u, y + 4.5 * u
                marks.polygon(c, [x, x + 12.0 * u, x + 12.0 * u, x], [lo, lo, hi, hi], rgb)
            text.draw(c, x + 17.0 * u, y, s, size, pal.color(pal.ink), pal.text_bold, va = 'center', tracking = track,
                      halo = 0.0 if loc in ('right', 'below') else 2.6 * u, halo_color = pal.color('paper'))


def _recorded(method):
    """Notes a call on its figure for Figure.in_palette, as colors, widths, and the like are settled from the palette
    when a plot method runs. Only the outermost call: what it calls in turn would run again by itself in a replay."""
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        fig = self.figure
        fig._inside += 1
        try:
            result = method(self, *args, **kwargs)
        finally:
            fig._inside -= 1
        if not fig._inside:
            fig._calls.append((self, method.__name__, args, kwargs))
        return result
    return call


# All that only asks or converts is left out; a method that fails records nothing, as it changed nothing to replay
for _name, _method in list(vars(Axes).items()):
    if callable(_method) and _name[0] != '_' and _name not in ('get_xlim', 'get_ylim', 'page', 'data', 'boxes'):
        setattr(Axes, _name, _recorded(_method))
