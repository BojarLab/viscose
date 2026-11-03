"""Axes: records what to draw, then lays it out and draws it once the figure is rendered."""
import math
import numpy as np
from . import marks, text
from .palettes import mix


# Past this many cells a heatmap is one image, as matplotlib's rasterized heatmaps are, not thousands of shapes
HEAT_IMAGE_CELLS = 2500
LEGEND_ROW = 16.0
COLORBAR_WIDTH = 11.0


def nice_ticks(lo, hi, n = 6):
    """Round tick positions covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    mag = 10 ** math.floor(math.log10((hi - lo) / max(n - 1, 1)))
    # The round step whose tick count comes closest to n, counting the ticks that actually fall inside the range
    step = min((s * mag for s in (1, 2, 5, 10)),
               key = lambda v: abs(math.floor(hi / v + 1e-9) - math.ceil(lo / v - 1e-9) + 1 - n))
    first = math.ceil(lo / step - 1e-9)
    return [round(i * step, 10) for i in range(first, int(math.floor(hi / step + 1e-9)) + 1)]


def fmt(v):
    """A tick value without trailing zeros or a stray minus zero."""
    if abs(v) >= 1e5 or (v and abs(v) < 1e-3):
        m, e = ('%.3g' % (v / 10 ** math.floor(math.log10(abs(v))))), int(math.floor(math.log10(abs(v))))
        return f'10^{{{e}}}' if m == '1' else f'{m}×10^{{{e}}}'
    s = ('%.6g' % v)
    return '0' if s == '-0' else s


def _numeric(values):
    """Whether values are numbers rather than category labels."""
    return np.asarray(values).dtype.kind in 'fiu'


def _check_cmap(cmap):
    """A palette brings two color scales; any other name would fail, or worse succeed, as an attribute lookup."""
    if cmap not in ('diverging', 'sequential'):
        raise ValueError(f"cmap has to be 'diverging' or 'sequential', got {cmap!r}: a palette brings these two scales")


def _check_scale(scale):
    """Any other name would quietly draw a linear axis."""
    if scale not in ('linear', 'log'):
        raise ValueError(f"An axis scale has to be 'linear' or 'log', got {scale!r}")


def _check_limits(lo, hi):
    """Limits as floats; a NaN or an infinity would otherwise only fail once the figure is saved."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f'Axis limits have to be finite, got {lo} and {hi}')
    return float(lo), float(hi)


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
        self._xticks, self._yticks, self._xtl, self._ytl = None, None, None, None
        self._xrot, self._yright = 0.0, False
        self._xscale, self._yscale = 'linear', 'linear'
        self._xinv, self._yinv = False, False
        self._cycle = 0
        self._cats = {'x': {}, 'y': {}}
        self._hidden = set()
        self._xmargin, self._ymargin = 0.05, 0.05
        self._scale, self._cbar = None, None
        self._over = [0.0, 0.0]
        self.rect = (0.0, 0.0, 1.0, 1.0)
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
        if arr.dtype.kind in 'US' or arr.dtype == object:
            cats = self._cats[axis]
            for v in arr.ravel():
                cats.setdefault(str(v), len(cats))
            return np.array([cats[str(v)] for v in arr.ravel()], dtype = float).reshape(arr.shape)
        return arr.astype(float)

    def _add(self, kind, z, **kw):
        self.artists.append(dict(kind = kind, z = z, order = len(self.artists), **kw))

    def _labeled(self, label, rgb, kind = 'patch', marker = 'o', area = 36.0):
        if label is not None:
            self.legend_items.append((str(label), rgb, kind, marker, area))

    # --- plotting ---
    def plot(self, x, y = None, color = None, lw = None, label = None, marker = False, alpha = 1.0, dash = None,
             ms = 42,
             z = 4):
        """A line, with markers at its points if `marker` is True or a marker name."""
        x, y = (np.arange(len(x)), x) if y is None else (x, y)
        x, y = self._pos(x, 'x'), self._pos(y, 'y')
        _check_lengths(x, y)
        rgb = self._next_color(color)
        self._add('line', z, x = x, y = y, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        if marker:
            self._add('points', z + 1, x = x, y = y, s = ms, rgb = rgb, alpha = alpha,
                      marker = 'o' if marker is True else marker,
                      edge = None, gid = None)
        self._labeled(label, rgb, 'line')
        return self

    def scatter(self, x, y, s = 30, color = None, c = None, label = None, alpha = 1.0, cmap = 'sequential', vmin = None,
                vmax = None, z = 3, marker = 'o', edge = None, gid = None, hue = None, style = None, size = None,
                sizes = (20.0, 160.0), colors = None, markers = None, hue_label = None, style_label = None,
                size_label = None, legend = 'right'):
        """Points, as in seaborn's scatterplot. `hue`, `style`, and `size` hold one value per point and map to color,
        marker shape, and area, with a legend. Numeric `hue` with many values, or `c`, maps through a palette scale;
        call
        `colorbar` to show it. `colors` and `markers` are lists or {value: name} dicts. `edge` is (color, width)."""
        x, y = self._pos(x, 'x'), self._pos(y, 'y')
        _check_lengths(x, y)
        edge = None if edge is None else (self.pal.color(edge[0]), edge[1])
        if hue is not None and _numeric(hue) and len(np.unique(hue)) > 8:
            c, hue = hue, None
        if c is not None:
            _check_cmap(cmap)
            c = np.asarray(c, dtype = float)
            lo, hi = np.nanmin(c) if vmin is None else vmin, np.nanmax(c) if vmax is None else vmax
            t = (c - lo) / (hi - lo if hi > lo else 1.0)
            self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
            for level in np.unique(np.round(t[np.isfinite(t)] * 24)):
                m = np.round(t * 24) == level
                area = s if np.ndim(s) == 0 else np.asarray(s)[m]
                self._add('points', z, x = x[m], y = y[m], s = area, rgb = tuple(getattr(self.pal, cmap)(level / 24)),
                          alpha = alpha, marker = marker, edge = edge, gid = gid)
        if hue is None and style is None and size is None:
            if c is None:
                rgb = self._next_color(color)
                self._add('points', z, x = x, y = y, s = s, rgb = rgb, alpha = alpha, marker = marker, edge = edge,
                          gid = gid)
                self._labeled(label, rgb, 'point', marker)
            return self
        n = len(x)
        hues, styles = (np.array([''] * n if v is None else [str(u) for u in v], dtype = object) for v in (hue, style))
        hue_levels, style_levels = list(dict.fromkeys(hues)), list(dict.fromkeys(styles))
        if isinstance(colors, dict):
            hue_rgb = {k: self.pal.color(colors[k]) for k in hue_levels}
        elif colors is not None:
            hue_rgb = dict(zip(hue_levels, [self.pal.color(k) for k in colors]))
        elif hue is not None:
            hue_rgb = dict(zip(hue_levels, self.pal.contrast(len(hue_levels))))
        else:
            hue_rgb = {'': self._next_color(color)}
        names = list(marks.MARKERS)
        if isinstance(markers, dict):
            style_marker = {k: markers[k] for k in style_levels}
        else:
            order = names if markers is None else list(markers)
            style_marker = {k: order[i % len(order)] for i, k in enumerate(style_levels)}
        area = np.full(n, float(s)) if np.ndim(s) == 0 else np.asarray(s, dtype = float)
        size_key = []
        if size is not None:
            if _numeric(size):
                v = np.asarray(size, dtype = float)
                lo, hi = np.nanmin(v), np.nanmax(v)
                span = (sizes[1] - sizes[0]) / (hi - lo if hi > lo else 1.0)
                scale = lambda u: sizes[0] + (np.asarray(u) - lo) * span
                area = scale(v)
                size_key = [(fmt(t), float(scale(t))) for t in nice_ticks(float(lo), float(hi), 4)]
            else:
                levels = list(dict.fromkeys(str(u) for u in size))
                steps = np.linspace(sizes[0], sizes[1], max(len(levels), 2))
                area = np.array([steps[levels.index(str(u))] for u in size], dtype = float)
                size_key = [(k, float(steps[i])) for i, k in enumerate(levels)]
        for h in hue_levels:
            for st in style_levels:
                m = (hues == h) & (styles == st)
                if m.any():
                    self._add('points', z, x = x[m], y = y[m], s = area[m], rgb = hue_rgb[h], alpha = alpha,
                              marker = style_marker[st], edge = edge, gid = gid)
        sections = (hue is not None) + (style is not None) + (size is not None)
        neutral = self.pal.color(self.pal.muted)
        if hue is not None:
            if sections > 1:
                self.legend_items.append((hue_label or 'Color', None, 'header', 'o', 0.0))
            for h in hue_levels:
                self._labeled(h, hue_rgb[h], 'point', marker)
        if style is not None:
            if sections > 1:
                self.legend_items.append((style_label or 'Shape', None, 'header', 'o', 0.0))
            for st in style_levels:
                self._labeled(st, neutral if hue is not None else hue_rgb[''], 'point', style_marker[st])
        if size is not None:
            if sections > 1:
                self.legend_items.append((size_label or 'Size', None, 'header', 'o', 0.0))
            for name, a in size_key:
                self._labeled(name, neutral if hue is not None else hue_rgb[''], 'point', marker, a)
        if legend:
            self._legend_opts = legend
        return self

    def bar(self, x, height, width = None, bottom = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
            z = 3):
        """Vertical bars; several colors give one bar each, and string positions become category labels."""
        x = self._pos(x, 'x')
        h = np.broadcast_to(np.asarray(height, dtype = float), x.shape)
        b = np.broadcast_to(np.asarray(bottom, dtype = float), x.shape)
        width = self.pal.bar_width if width is None else width
        cols = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(x)
        self._add('bars', z, x = x, h = h, b = b, w = width, rgb = cols, alpha = alpha, horizontal = False,
                  stacked = stacked)
        self._labeled(label, cols[0] if cols else None)
        return self

    def barh(self, y, width, height = None, left = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
             z = 3):
        """Horizontal bars."""
        y = self._pos(y, 'y')
        w = np.broadcast_to(np.asarray(width, dtype = float), y.shape)
        b = np.broadcast_to(np.asarray(left, dtype = float), y.shape)
        height = self.pal.bar_width if height is None else height
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
                 width = float(np.diff(edges).mean()) * 0.94, color = color, label = label, alpha = alpha)
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

    def errorbar(self, x, y, yerr, color = 'ink', lw = 1.2, capsize = 3.0, z = 5):
        x, y = self._pos(x, 'x'), np.asarray(y, dtype = float)
        self._add('errors', z, x = x, y = y, e = np.broadcast_to(np.asarray(yerr, dtype = float), y.shape),
                  rgb = self.pal.color(color), lw = lw, cap = capsize)
        return self

    def stem(self, x, y, color = None, lw = 3.0, label = None, cap = False, z = 4):
        """Vertical sticks from zero, as in a mass spectrum."""
        x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
        rgb = self._next_color(color)
        self._add('stems', z, x = x, y = y, rgb = rgb, lw = lw, cap = cap)
        self._labeled(label, rgb)
        return self

    def hlines(self, y, x0, x1, color = None, lw = 1.5, alpha = 1.0, z = 2):
        """Horizontal segments, under the points, such as the intervals of a forest plot."""
        y = np.atleast_1d(np.asarray(y, dtype = float))
        x0 = np.broadcast_to(np.asarray(x0, dtype = float), y.shape)
        x1 = np.broadcast_to(np.asarray(x1, dtype = float), y.shape)
        self._add('hsegs', z, y = y, x0 = x0, x1 = x1, rgb = self._next_color(color), lw = lw, alpha = alpha)
        return self

    def arrow(self, x0, y0, x1, y1, color = 'muted', lw = 1.0, alpha = 1.0, gid = None, z = 2):
        """An arrow between two data points; `gid` names it in SVG output, for hover text."""
        self._add('arrow', z, x0 = float(x0), y0 = float(y0), x1 = float(x1), y1 = float(y1),
                  rgb = self.pal.color(color),
                  lw = lw, alpha = alpha, gid = gid)
        return self

    def axhline(self, y = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), z = 1):
        self._add('hline', z, v = float(y), rgb = self.pal.color(color), lw = lw, dash = dash)
        return self

    def axvline(self, x = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), z = 1):
        self._add('vline', z, v = float(x), rgb = self.pal.color(color), lw = lw, dash = dash)
        return self

    def heatmap(self, data, xticklabels = None, yticklabels = None, cmap = 'diverging', vmin = None, vmax = None,
                center = 0.0, labels_right = True, show_all = False, colorbar = False, colorbar_label = ''):
        """A matrix of cells, first row at the top, where NaN is left empty. `cmap` is diverging or sequential. Too many
        labels are thinned unless show_all, which can also be a pair for the x and y axes."""
        _check_cmap(cmap)
        m = np.asarray(data, dtype = float)
        if not m.size:
            raise ValueError('A heatmap needs at least one cell')
        self.show_all = show_all if isinstance(show_all, tuple) else (show_all, show_all)
        if cmap == 'diverging':
            d = np.nanmax(np.abs(m - center)) if vmin is None and vmax is None else max(
                abs((vmin if vmin is not None else center) - center),
                abs((vmax if vmax is not None else center) - center))
            d = float(d) if d else 1.0
            t = 0.5 + (m - center) / (2 * d)
            lo, hi = center - d, center + d
        else:
            lo, hi = np.nanmin(m) if vmin is None else vmin, np.nanmax(m) if vmax is None else vmax
            t = (m - lo) / (hi - lo if hi > lo else 1.0)
        self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
        self._add('heat', 2, m = m, rgb = getattr(self.pal, cmap)(t))
        self.set_xlim(0, m.shape[1])
        self.set_ylim(0, m.shape[0])
        self._xticks = list(np.arange(m.shape[1]) + 0.5) if xticklabels is not None else []
        self._xtl = [str(v) for v in xticklabels] if xticklabels is not None else None
        self._yticks = list(np.arange(m.shape[0]) + 0.5) if yticklabels is not None else []
        self._ytl = [str(v) for v in yticklabels][::-1] if yticklabels is not None else None
        self._xrot = 90.0 if xticklabels is not None and max(len(str(v)) for v in xticklabels) > 2 else 0.0
        self._yright = labels_right
        self._hidden |= {'spines', 'grid', 'ticks'}
        self.grid_on = False
        if colorbar:
            self.colorbar(colorbar_label)
        return self

    def colorbar(self, label = '', loc = 'right', source = None):
        """Show the color scale of this axes, or of `source`: to its right, or filling it with loc = 'fill'."""
        self._cbar = dict(label = label, loc = loc, source = source or self)
        return self

    def ridgeline(self, x, curves, labels = None, overlap = 2.05):
        """Distributions stacked with overlap, from the back (first) to the front (last)."""
        x = np.asarray(x, dtype = float)
        curves = [np.asarray(y, dtype = float) / np.nanmax(y) for y in curves]
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

    def dendrogram(self, icoord, dcoord, orientation = 'top', n = None, labels = None, scale = False):
        """Link lines as scipy gives them; heights run away from the matrix that sits on the given side. `labels` name
        the leaves, and `scale` keeps ticks on the height axis."""
        self._add('dendro', 3, ic = icoord, dc = dcoord, orient = orientation)
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
             halo = 0.0,
             z = 20, axes_fraction = False):
        """Text at a data position, or at a fraction of the axes with axes_fraction = True."""
        self._add('text', z, x = float(x), y = float(y), s = str(s), size = size, rgb = self.pal.color(color),
                  bold = bold, ha = ha, va = va, rotation = rotation, halo = halo, dx = 0.0, dy = 0.0,
                  frac = axes_fraction, tag = False)
        return self

    def annotate(self, s, xy, offset = (0.0, 0.0), size = 8.6, color = 'ink', bold = None, ha = 'left', va = 'center',
                 rotation = 0.0, halo = 2.6, z = 20, svg_label = False):
        """Text next to a data point, offset in points, with a white halo that keeps it readable over the marks. With
        svg_label = True, SVG output writes it the way matplotlib writes text, so that glycowork's annotate_figure can
        replace it with a glycan drawing."""
        self._add('text', z, x = float(xy[0]), y = float(xy[1]), s = str(s), size = size, rgb = self.pal.color(color),
                  bold = bold, ha = ha, va = va, rotation = rotation, halo = halo, dx = offset[0], dy = offset[1],
                  frac = False, tag = svg_label)
        return self

    # --- settings ---
    def set_xlabel(self, s):
        self.xlabel = s
        return self

    def set_ylabel(self, s):
        self.ylabel = s
        return self

    def set_title(self, s):
        self.title_text = s
        return self

    def set_xlim(self, lo, hi):
        self._xlim = _check_limits(lo, hi)
        return self

    def set_ylim(self, lo, hi):
        self._ylim = _check_limits(lo, hi)
        return self

    def set_xticks(self, ticks, labels = None, rotation = 0.0):
        self._xticks = list(ticks)
        self._xtl = None if labels is None else [str(v) for v in labels]
        self._xrot = rotation
        return self

    def set_yticks(self, ticks, labels = None):
        self._yticks = list(ticks)
        self._ytl = None if labels is None else [str(v) for v in labels]
        return self

    def set_xscale(self, s):
        _check_scale(s)
        self._xscale = s
        return self

    def set_yscale(self, s):
        _check_scale(s)
        self._yscale = s
        return self

    def set(self, **settings):
        """Several settings at once, as in matplotlib: xlabel, ylabel, title, xlim, ylim, xscale, yscale."""
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
        """Show a legend: below or right of the axes, or inside at upper left, upper right, lower left or lower
        right."""
        self._legend_opts = loc
        return self

    def off(self):
        """Hide axis lines, ticks and grid, for figures whose position axes carry no meaning."""
        self._hidden |= {'spines', 'ticks', 'grid'}
        self._xticks, self._yticks = [], []
        self.grid_on = False
        return self

    # --- limits and ticks ---
    def _extent(self, axis):
        lo, hi = [], []
        sticky = []
        for a in self.artists:
            k = a['kind']
            if axis == 'x' and k in ('line', 'points', 'stems', 'band', 'errors'):
                vals = a['x']
            elif axis == 'y' and k in ('line', 'points', 'hsegs'):
                vals = a['y']
            elif axis == 'y' and k == 'band':
                vals = np.r_[a['y1'], a['y2']]
            elif axis == 'y' and k == 'stems':
                vals = np.r_[a['y'], 0.0]
                sticky.append(0.0)
            elif axis == 'y' and k == 'errors':
                vals = np.r_[a['y'] - a['e'], a['y'] + a['e']]
            elif axis == 'x' and k == 'hsegs':
                vals = np.r_[a['x0'], a['x1']]
            elif k == 'arrow':
                vals = np.array([a[axis + '0'], a[axis + '1']])
            elif k == 'bars' and (axis == 'x') != a['horizontal']:
                vals = np.r_[a['x'] - a['w'] / 2 - 0.08, a['x'] + a['w'] / 2 + 0.08]
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
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo.append(vals.min())
                hi.append(vals.max())
        return (min(lo), max(hi), sticky) if lo else (0.0, 1.0, [])

    def _limits(self):
        self._over = [0.0, 0.0]
        self._legend_size = None
        for a in self.artists:
            if a['kind'] == 'text':
                a['_ha'], a['_dx'] = a['ha'], a['dx']
        for axis, fixed, scale, margin in (('x', self._xlim, self._xscale, self._xmargin),
                                           ('y', self._ylim, self._yscale, self._ymargin)):
            lo, hi, sticky = self._extent(axis)
            if scale == 'log':
                lo, hi = max(lo, 1e-12), max(hi, lo * 10)
                span = 0.03 * math.log10(hi / lo)
                lo, hi = 10 ** (math.log10(lo) - span), 10 ** (math.log10(hi) + span)
            else:
                pad = (hi - lo) * margin if hi > lo else (abs(lo) * 0.1 or 1.0)
                # Bars of no height stick to both ends, and the axis still needs a range
                lo, hi = (lo if any(abs(lo - s) < 1e-12 for s in sticky) else lo - pad,
                          hi if hi > lo and any(abs(hi - s) < 1e-12 for s in sticky) else hi + pad)
            if fixed is not None:
                # A log axis has no zero or negative end, so such a limit stays automatic
                lo, hi = [v if v > 0 or scale != 'log' else auto for v, auto in zip(fixed, (lo, hi))]
                pad = abs(lo) * 0.1 or 1.0
                lo, hi = (lo - pad, hi + pad) if lo == hi else (lo, hi)
            setattr(self, axis + 'lim', (lo, hi))
        for axis, ticks, labels, inverted in (('x', self._xticks, self._xtl, self._xinv),
                                              ('y', self._yticks, self._ytl, self._yinv)):
            cats = self._cats[axis]
            lo, hi = sorted(getattr(self, axis + 'lim'))
            if ticks is None and cats:
                ticks, labels = list(cats.values()), list(cats.keys())
            elif ticks is None and (self._xscale if axis == 'x' else self._yscale) == 'log':
                ticks = [10.0 ** e for e in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
                         if lo <= 10.0 ** e <= hi]
            elif ticks is None:
                ticks = nice_ticks(lo, hi)
            setattr(self, axis + 'ticks', ticks)
            setattr(self, axis + 'labels', labels if labels is not None else [fmt(v) for v in ticks])
            if inverted:
                setattr(self, axis + 'lim', tuple(reversed(getattr(self, axis + 'lim'))))

    def _tx(self, v):
        x0, y0, w, h = self.rect
        lo, hi = self.xlim
        v = np.asarray(v, dtype = float)
        if self._xscale == 'log':
            return x0 + (np.log10(np.maximum(v, 1e-300)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * w
        return x0 + (v - lo) / (hi - lo) * w

    def _ty(self, v):
        x0, y0, w, h = self.rect
        lo, hi = self.ylim
        v = np.asarray(v, dtype = float)
        if self._yscale == 'log':
            return y0 + (np.log10(np.maximum(v, 1e-300)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * h
        return y0 + (v - lo) / (hi - lo) * h

    # --- layout ---
    def _tick_sizes(self):
        """Tick label sizes for the x and y axes; a heatmap's labels are many, so they are smaller."""
        heat = any(a['kind'] == 'heat' for a in self.artists)
        return (8.2, 8.0) if heat else (self.pal.tick_size, self.pal.tick_size)

    def _title_size(self):
        """A long title shrinks to fit the figure's width, down to a readable size; past that the page grows."""
        wide = text.width(self.title_text, 15.0, self.pal.title_bold)
        return max(9.0, min(15.0, 15.0 * (self.figure._base[0] - self.rect[0] - 3.0) / wide))

    def _colorbar_width(self):
        """Room a colorbar at the right takes, past the tick labels: bar, its ticks, and its label."""
        source = self._cbar['source']
        ticks = nice_ticks(source._scale['lo'], source._scale['hi'], 5)
        label = self.pal.label_size * 1.6 if self._cbar['label'] else 0.0
        return 12.0 + COLORBAR_WIDTH + 5.0 + max(text.width(fmt(v), 8.4) for v in ticks) + label

    def _margins(self):
        """Room the decorations need around the plot area, in points: left, right, bottom, top."""
        pal = self.pal
        off = pal.spine_offset if 'spines' not in self._hidden and pal.spines else 0.0
        tick = pal.tick_length if 'ticks' not in self._hidden and pal.spines else 0.0
        pad = off + tick + 5.0
        sx, sy = self._tick_sizes()
        yw = max((text.width(s, sy, pal.tick_bold) for s in self.ylabels), default = 0.0)
        xh = sx * 1.0
        if self._xrot:
            sin = math.sin(math.radians(self._xrot))
            xh = max((text.width(s, sx, pal.tick_bold) for s in self.xlabels), default = 0.0) * sin + sx * 0.4
        left, right, bottom, top = 3.0, 3.0, 3.0, 3.0
        if self.ylabels and self.yticks is not None and len(self.yticks):
            if self._yright:
                right += pad + yw
            else:
                left += pad + yw
        if self.xlabels and len(self.xticks):
            bottom += pad + xh
        if self.ylabel:
            if self._yright:
                right += pal.label_size * 1.7
            else:
                left += pal.label_size * 1.7
        if self.xlabel:
            bottom += pal.label_size * 1.7
        if getattr(self, '_ridge_labels', None):
            left += max(text.width(s, pal.tick_size + 1, pal.text_bold) for s in self._ridge_labels) + 8.0
        if self.title_text:
            top += 21.0
        if self.legend_items and self._legend_opts in ('below', 'right'):
            tw, th = self._legend_size or self._legend_natural()
            if self._legend_opts == 'below':
                bottom += th + 8.0
            else:
                right += tw + 18.0
        if self._cbar and self._cbar['loc'] == 'right':
            right += self._colorbar_width()
        return left, right, bottom, top

    def _legend_entries(self):
        """Each item's text size, tracking, shown text, and width, in legend order."""
        pal = self.pal
        size = 7.4 if pal.caps else pal.tick_size - 0.5
        track = 0.7 if pal.caps else 0.0
        out = []
        for s, rgb, kind, marker, area in self.legend_items:
            shown = text.spaced(pal, s)
            header = kind == 'header'
            out.append((shown, rgb, kind, marker, area, text.width(shown, size, pal.text_bold or header, track)
                        + (0.0 if header else 18.0)))
        return size, track, out

    def _legend_natural(self):
        """Legend size before the axes are placed: one row or one column."""
        _, _, entries = self._legend_entries()
        widths = [e[5] for e in entries]
        return (max(widths), LEGEND_ROW * len(widths)) if self._legend_opts == 'right' else (sum(widths), LEGEND_ROW)

    def _legend_grid(self):
        """Offsets of the legend's items from its top left, wrapped to the axes, and the legend's total size."""
        _, _, w, h = self.rect
        _, _, entries = self._legend_entries()
        widths = [e[5] for e in entries]
        if self._legend_opts == 'below':
            spots, x, y = [], 0.0, 0.0
            for v in widths:
                if x and x + v > w:
                    x, y = 0.0, y + LEGEND_ROW
                spots.append((x, y))
                x += v + 14.0
            return spots, (w if y else sum(widths) + 14.0 * (len(widths) - 1), y + LEGEND_ROW)
        per = max(1, int(h // LEGEND_ROW)) if self._legend_opts == 'right' else len(widths)
        spots, offset, total_h = [], 0.0, 0.0
        for start in range(0, len(widths), per):
            for i in range(start, min(start + per, len(widths))):
                spots.append((offset, (i - start) * LEGEND_ROW))
            total_h = max(total_h, (min(start + per, len(widths)) - start) * LEGEND_ROW)
            offset += max(widths[start:start + per]) + 12.0
        return spots, (offset - 12.0, total_h)

    def _fit_labels(self):
        """Turn tick labels that would collide, thin those that would still collide, wrap the legend, and move or make
        room
        for annotations that would leave the page. True if anything changed."""
        pal = self.pal
        changed = False
        sx, sy = self._tick_sizes()
        if len(self.xticks) > 1:
            gap = np.min(np.abs(np.diff(self._tx(self.xticks))))
            if not self._xrot and max(text.width(s, sx, pal.tick_bold) for s in self.xlabels) + 6.0 > gap:
                self._xrot = 90.0
                changed = True
            if self._xrot and gap < sx * 1.05 and not self.show_all[0]:
                k = int(math.ceil(sx * 1.05 / gap))
                self.xticks, self.xlabels = self.xticks[::k], self.xlabels[::k]
                changed = True
        if len(self.yticks) > 1 and not self.show_all[1]:
            gap = np.min(np.abs(np.diff(self._ty(self.yticks))))
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
        if self.title_text:
            over[1] = self.rect[0] + text.width(self.title_text, self._title_size(), pal.title_bold) - (page - 3.0)
        for a in self.artists:
            if a['kind'] != 'text' or a['rotation'] or a['frac'] or a['ha'] == 'center':
                continue
            tw = text.width(a['s'], a['size'] or pal.tick_size, pal.text_bold if a['bold'] is None else a['bold'])
            px, gap = float(self._tx(a['x'])), abs(a['dx']) + 3.0
            over_right, over_left = px + gap + tw - (page - 3.0), tw + gap - (px - 3.0)
            natural_right = a['ha'] == 'left'
            # The label's own side if it fits, else the other side if that fits, else whichever side overflows less. A
            # glycan drawing always grows rightward from its label's anchor, so such a label stays and the page grows.
            if a['tag'] or (over_right if natural_right else over_left) <= 0:
                use_right = natural_right
            elif (over_left if natural_right else over_right) <= 0:
                use_right = not natural_right
            else:
                use_right = over_right <= over_left
            a['_ha'] = 'left' if use_right else 'right'
            a['_dx'] = a['dx'] if use_right == natural_right else (gap if use_right else -gap)
            over[1 if use_right else 0] = max(over[1 if use_right else 0], over_right if use_right else over_left)
        for side in (0, 1):
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
            if k == 'text':
                continue
            pad = 9.0 if k == 'points' else 1.0
            # A named element stays a bare path, for the code that looks for it in the SVG
            gid = a.get('gid')
            c.state['clip'] = None if gid else [((x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad), c.state['ctm'])]
            getattr(self, '_d_' + k)(c, a)
            for op in c.ops[n:]:
                op['gid'] = gid
                if k not in ('hline', 'vline'):
                    op['mark'] = True
        c.restoreState()

    def _d_line(self, c, a):
        mask = np.isfinite(a['x']) & np.isfinite(a['y'])
        if mask.any():
            marks.line(c, self.pal, self._tx(a['x'][mask]), self._ty(a['y'][mask]), a['rgb'], a['lw'], a['alpha'],
                       a['dash'])

    def _d_points(self, c, a):
        mask = np.isfinite(a['x']) & np.isfinite(a['y'])
        s = a['s'] if np.ndim(a['s']) == 0 else np.asarray(a['s'])[mask]
        marks.points(c, self.pal, self._tx(a['x'][mask]), self._ty(a['y'][mask]), s, a['rgb'], a['alpha'], a['marker'],
                     a['edge'])

    def _d_band(self, c, a):
        px = np.r_[self._tx(a['x']), self._tx(a['x'][::-1])]
        py = np.r_[self._ty(a['y1']), self._ty(a['y2'][::-1])]
        marks.polygon(c, px, py, a['rgb'], a['alpha'])

    def _d_bars(self, c, a):
        for x, b, v, rgb in zip(a['x'], a['b'], a['h'], a['rgb']):
            if a['horizontal']:
                x1, x2, y1, y2 = self._tx(b), self._tx(b + v), self._ty(x - a['w'] / 2), self._ty(x + a['w'] / 2)
            else:
                x1, x2, y1, y2 = self._tx(x - a['w'] / 2), self._tx(x + a['w'] / 2), self._ty(b), self._ty(b + v)
            lx, ly, bw, bh = float(min(x1, x2)), float(min(y1, y2)), float(abs(x2 - x1)), float(abs(y2 - y1))
            if a['stacked']:
                marks.segment(c, self.pal, lx, ly, bw, bh, rgb, 0.0 if b == 0 else max(self.pal.heat_gap, 0.8))
            else:
                marks.bar(c, self.pal, lx, ly, bw, bh, rgb, a['horizontal'], a['alpha'])

    def _d_errors(self, c, a):
        for x, y, e in zip(self._tx(a['x']), a['y'], a['e']):
            lo, hi = self._ty(y - e), self._ty(y + e)
            marks.polyline(c, [x, x], [lo, hi], a['rgb'], a['lw'], cap = 0)
            for v in (lo, hi):
                marks.polyline(c, [x - a['cap'], x + a['cap']], [v, v], a['rgb'], a['lw'], cap = 0)

    def _d_stems(self, c, a):
        pal = self.pal
        xs, zero, ys = self._tx(a['x']), float(self._ty(0.0)), self._ty(a['y'])
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
            marks.line(c, self.pal, [x0, x1], [y, y], a['rgb'], a['lw'], a['alpha'])

    def _d_arrow(self, c, a):
        marks.arrow(c, self.pal, float(self._tx(a['x0'])), float(self._ty(a['y0'])), float(self._tx(a['x1'])),
                    float(self._ty(a['y1'])), a['rgb'], a['lw'], a['alpha'])

    def _d_hline(self, c, a):
        x0, _, w, _ = self.rect
        y = float(self._ty(a['v']))
        marks.polyline(c, [x0, x0 + w], [y, y], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_vline(self, c, a):
        _, y0, _, h = self.rect
        x = float(self._tx(a['v']))
        marks.polyline(c, [x, x], [y0, y0 + h], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_heat(self, c, a):
        x0, y0, w, h = self.rect
        ny, nx = a['m'].shape
        cw, ch = w / nx, h / ny
        if nx * ny > HEAT_IMAGE_CELLS:
            op = dict(c.state)
            px = (a['rgb'] * 255.0 + 0.5).astype(np.uint8)
            op.update(kind = 'image', w = nx, h = ny, rgb = px.tobytes(), ctm = (w, 0.0, 0.0, h, x0, y0),
                      alpha = (np.isfinite(a['m']) * 255).astype(np.uint8).tobytes())
            c.ops.append(op)
            return
        for r in range(ny):
            for j in range(nx):
                if np.isfinite(a['m'][r, j]):
                    marks.heat_cell(c, self.pal, x0 + j * cw, y0 + (ny - 1 - r) * ch, cw, ch, tuple(a['rgb'][r, j]))

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
                top = mix(pal.color('a2'), 0.14 + 0.62 * i / max(n - 1, 1))
                rgb = pal.color('a0')
            peak = float(self._ty(base + np.nanmax(y) * a['overlap']) - self._ty(base))
            marks.ridge(c, pal, tx, float(self._ty(base)), ys, rgb, top, peak)

    def _d_dendro(self, c, a):
        pal = self.pal
        lw = {'deep': 1.2, 'needle': 0.45, 'flat': 2.6, 'soft': 1.5}[pal.bar]
        rgb = {'deep': pal.color('a1'), 'needle': pal.color('ink'), 'flat': pal.color('a0'),
               'soft': pal.color('a2')}[pal.bar]
        lines = []
        for ic, dc in zip(a['ic'], a['dc']):
            pos, dep = np.asarray(ic) / 10.0, np.asarray(dc, dtype = float)
            lines.append((self._tx(dep), self._ty(pos)) if a['orient'] == 'left' else (self._tx(pos), self._ty(dep)))
        marks.polylines(c, lines, rgb, lw)

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
        tl = pal.tick_length
        spines = () if 'spines' in self._hidden else pal.spines
        xt, yt = [float(self._tx(v)) for v in self.xticks], [float(self._ty(v)) for v in self.yticks]
        if 'left' in spines and 'left' not in self._hidden:
            lo, hi = (min(yt), max(yt)) if off and yt else (y0, y0 + h)
            marks.polyline(c, [x0 - off, x0 - off], [lo, hi], ink, pal.spine_width, cap = cap)
            for y in yt if tl and 'ticks' not in self._hidden else ():
                marks.polyline(c, [x0 - off - tl, x0 - off], [y, y], ink, pal.spine_width, cap = 0)
        if 'bottom' in spines:
            lo, hi = (min(xt), max(xt)) if off and xt else (x0, x0 + w)
            marks.polyline(c, [lo, hi], [y0 - off, y0 - off], ink, pal.spine_width, cap = cap)
            for x in xt if tl and 'ticks' not in self._hidden else ():
                marks.polyline(c, [x, x], [y0 - off - tl, y0 - off], ink, pal.spine_width, cap = 0)

    def _draw_text(self, c):
        pal = self.pal
        x0, y0, w, h = self.rect
        tick_rgb = pal.color(pal.tick_color or pal.muted)
        off = pal.spine_offset if pal.spines and 'spines' not in self._hidden else 0.0
        tl = pal.tick_length if pal.spines and 'ticks' not in self._hidden else 0.0
        pad = off + tl + 5.0
        size = pal.tick_size
        heat = 'heat' in {a['kind'] for a in self.artists}
        small, ytext = self._tick_sizes()
        ink = pal.color(pal.ink)
        xh = small
        for v, s in zip(self.xticks, self.xlabels):
            x = float(self._tx(v))
            if self._xrot:
                text.draw(c, x + small * 0.36 * math.sin(math.radians(self._xrot)), y0 - pad, s, small,
                          ink if heat else tick_rgb, pal.tick_bold, ha = 'right', rotation = self._xrot,
                          va = 'baseline')
            else:
                text.draw(c, x, y0 - pad, s, small, tick_rgb, pal.tick_bold, ha = 'center', va = 'top')
        yw = 0.0
        for v, s in zip(self.yticks, self.ylabels):
            y = float(self._ty(v))
            yw = max(yw, text.width(s, ytext, pal.tick_bold))
            if self._yright:
                text.draw(c, x0 + w + pad - 2.0, y, s, ytext, ink if heat else tick_rgb, pal.tick_bold, ha = 'left',
                          va = 'center')
            else:
                text.draw(c, x0 - pad, y, s, ytext, tick_rgb, pal.tick_bold, ha = 'right', va = 'center')
        if self._xrot and self.xlabels:
            xh = max(text.width(s, small,
                                pal.tick_bold) for s in self.xlabels) * math.sin(math.radians(self._xrot)) + small * 0.4
        label_rgb = pal.color(pal.label_color or pal.ink)
        track = 0.9 if pal.caps else 0.0
        if self.xlabel:
            text.draw(c, x0 + w / 2, y0 - pad - xh - pal.label_size * 0.55, text.spaced(pal, self.xlabel),
                      pal.label_size,
                      label_rgb, pal.label_bold, ha = 'center', va = 'top', tracking = track)
        if self.ylabel:
            ylx = x0 + w + pad + yw + pal.label_size * 1.3 if self._yright else x0 - pad - yw - pal.label_size * 0.7
            text.draw(c, ylx, y0 + h / 2, text.spaced(pal, self.ylabel), pal.label_size, label_rgb, pal.label_bold,
                      ha = 'center',
                      va = 'baseline', rotation = 90.0, tracking = track)
        if self.title_text:
            text.draw(c, x0, y0 + h + 8.0, self.title_text, self._title_size(), pal.color(pal.title_color or pal.ink),
                      pal.title_bold, ha = 'left')
        for a in self.artists:
            if a['kind'] == 'ridges' and a['labels'] is not None:
                n = len(a['curves'])
                for i, s in enumerate(self._ridge_labels):
                    t = text.spaced(pal, s)
                    tsize = 7.8 if pal.caps else size + 1
                    text.draw(c, x0 - 8.0, float(self._ty(n - 1 - i)) + 1.0, t, tsize, ink, pal.text_bold,
                              ha = 'right', tracking = track)
        for a in sorted((r for r in self.artists if r['kind'] == 'text'), key = lambda r: r['order']):
            bold = pal.text_bold if a['bold'] is None else a['bold']
            px = x0 + a['x'] * w if a['frac'] else float(self._tx(a['x']))
            py = y0 + a['y'] * h if a['frac'] else float(self._ty(a['y']))
            n = len(c.ops)
            text.draw(c, px + a['_dx'], py + a['dy'], a['s'], a['size'] or size, a['rgb'], bold, ha = a['_ha'],
                      va = a['va'],
                      rotation = a['rotation'], halo = a['halo'])
            if a['tag']:
                for op in c.ops[n:]:
                    op['label'] = (a['order'], id(self), a['s'])
        self._draw_legend(c)
        if self._cbar:
            self._draw_colorbar(c, yw, pad)

    def _draw_colorbar(self, c, yw, pad):
        pal = self.pal
        x0, y0, w, h = self.rect
        scale = self._cbar['source']._scale
        lo, hi = scale['lo'], scale['hi']
        stops = [tuple(v) for v in getattr(pal, scale['cmap'])(np.linspace(0.0, 1.0, 9))]
        ticks = [t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi]
        ink, label = pal.color(pal.ink), self._cbar['label']
        if self._cbar['loc'] == 'fill':
            bh = min(9.0, h * 0.3)
            bw, by = w * 0.8, y0 + h * 0.34
            marks.polygon(c, [x0, x0 + bw, x0 + bw, x0], [by, by, by + bh, by + bh], None, 1.0,
                          ((x0, by, x0 + bw, by), stops))
            for t in ticks:
                tx = x0 + (t - lo) / (hi - lo) * bw
                text.draw(c, tx, by - 3.0, fmt(t), 7.4, ink, ha = 'center', va = 'top')
            if label:
                text.draw(c, x0, by + bh + 4.0, text.spaced(pal, label), 7.8 if pal.caps else 8.6, ink, ha = 'left')
            return
        right = yw + pad if self._yright and len(self.yticks) else 0.0
        right += pal.label_size * 1.7 if self.ylabel and self._yright else 0.0
        bx = x0 + w + right + 12.0
        bh = min(h * 0.6, 170.0)
        by = y0 + h - bh
        marks.polygon(c, [bx, bx + COLORBAR_WIDTH, bx + COLORBAR_WIDTH, bx], [by, by, by + bh, by + bh], None, 1.0,
                      ((bx, by, bx, by + bh), stops))
        widest = 0.0
        for t in ticks:
            text.draw(c, bx + COLORBAR_WIDTH + 5.0, by + (t - lo) / (hi - lo) * bh, fmt(t), 8.4, ink, va = 'center')
            widest = max(widest, text.width(fmt(t), 8.4))
        if label:
            text.draw(c, bx + COLORBAR_WIDTH + 5.0 + widest + pal.label_size * 1.3, by + bh / 2,
                      text.spaced(pal, label), pal.label_size, pal.color(pal.label_color or pal.ink), pal.label_bold,
                      ha = 'center', va = 'baseline', rotation = 90.0)

    def _draw_legend(self, c):
        pal = self.pal
        if not self.legend_items or self._legend_opts is None:
            return
        x0, y0, w, h = self.rect
        size, track, entries = self._legend_entries()
        spots, (tw, th) = self._legend_grid()
        loc = self._legend_opts
        if loc == 'right':
            left, top = x0 + w + 16.0, y0 + h - 8.0
        elif loc == 'below':
            rows = round(th / LEGEND_ROW)
            left = x0 if tw >= w else x0 + (w - tw) / 2
            top = getattr(self, '_legend_y', y0 - 40.0) + (rows - 1) * LEGEND_ROW
        else:
            left = x0 + w - tw - 8.0 if 'right' in loc else x0 + 10.0
            top = y0 + h - 12.0 if 'upper' in loc else y0 + 12.0 + th - LEGEND_ROW
        for (dx, dy), (s, rgb, k, marker, area, _) in zip(spots, entries):
            x, y = left + dx, top - dy
            if k == 'header':
                text.draw(c, x, y, s, size, pal.color(pal.ink), True, va = 'center', tracking = track)
                continue
            if k == 'line':
                marks.polyline(c, [x, x + 12.0], [y, y], rgb, max(pal.line_width, 1.2), cap = 0)
            elif k == 'point':
                # As large as in the plot, up to what a legend row can hold
                marks.points(c, pal, [x + 6.0], [y], min(area, 196.0 / pal.point_scale), rgb, marker = marker)
            else:
                marks.polygon(c, [x, x + 12.0, x + 12.0, x], [y - 4.5, y - 4.5, y + 4.5, y + 4.5], rgb)
            text.draw(c, x + 17.0, y, s, size, pal.color(pal.ink), pal.text_bold, va = 'center', tracking = track,
                      halo = 0.0 if loc in ('right', 'below') else 2.6)
