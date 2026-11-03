"""Figure: lays out axes, draws through glycorender's display list and writes PNG, PDF, or SVG."""
import builtins
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import numpy as np
from glycorender import pdfwrite, raster
from . import text
from .axes import Axes
from .palettes import get_palette


# Shadow offset and blur in points; a blur cannot stay vector, so the PDF and SVG carry it as an image at this
# resolution
SHADOW_DX, SHADOW_DY, SHADOW_BLUR, SHADOW_DPI = 1.2, 5.0, 4.7, 100.0
# Margins may take this share of an axes' cell before the figure grows
PLOT_SHARE_MARGINS = 0.55
# Coding agents and CI can hold a real console, so a terminal alone does not mean someone wants a window per figure
NO_VIEWER = ('CLAUDECODE', 'CI', 'PYTEST_CURRENT_TEST', 'VISCOSE_NO_VIEWER')


def show_png(png, viewer = True):
    """Show PNG bytes: inline in Jupyter or slate, in the system's image viewer from a person's terminal, and nowhere
    otherwise, so that test runs, pipelines, and coding agents stay silent. Setting the environment variable
    VISCOSE_NO_VIEWER or passing viewer = False turns the viewer off. `png` can be a function giving the bytes, which
    then costs nothing where there is nowhere to show them."""
    display = getattr(builtins, 'display', None) or getattr(sys.modules.get('__main__'), 'display', None)
    # IPython in a terminal has display() too, but can only print an image's name there
    ipython = sys.modules.get('IPython')
    terminal = ipython is not None and type(ipython.get_ipython()).__name__ == 'TerminalInteractiveShell'
    if display is not None and not terminal:
        data = png() if callable(png) else png
        # slate knows this for a figure by its type's module, viscose.figure
        display(type('Png', (), {'_repr_png_': lambda self: data})())
        return
    if not (viewer and sys.stdin is not None and sys.stdin.isatty()) or any(os.environ.get(n) for n in NO_VIEWER):
        return
    with tempfile.NamedTemporaryFile(suffix = '.png', prefix = 'viscose_', delete = False) as f:
        f.write(png() if callable(png) else png)
    if sys.platform == 'win32':
        os.startfile(f.name)
    elif sys.platform == 'darwin' or os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', f.name], stdout = subprocess.DEVNULL,
                         stderr = subprocess.DEVNULL)


def _shadow_layer(ops, width, height, dpi, strength):
    """The marks, blurred and moved, as a straight-alpha RGBA array whose color over white multiplies like the
    marks'."""
    sc = dpi / 72.0
    img = raster.render(ops, width, height, sc, sc, None).astype(np.float32) / 255.0
    alpha = img[..., 3]
    radius = max(1, int(round(SHADOW_BLUR * sc)))
    # How far the marks darken white in each channel, blurred; a soft shadow is this darkening at the given strength
    dark = np.stack([raster._box_blur((1.0 - img[..., i]) * alpha, radius) for i in range(3)], axis = -1)
    dx, dy = int(round(SHADOW_DX * sc)), int(round(SHADOW_DY * sc))
    # Moved, with whatever leaves the page gone rather than back in at the opposite edge
    moved = np.zeros_like(dark)
    moved[dy:, dx:] = dark[:dark.shape[0] - dy, :dark.shape[1] - dx]
    dark = moved * strength
    out_a = dark.max(axis = -1)
    safe = np.where(out_a > 1e-4, out_a, 1.0)
    rgb = np.where(out_a[..., None] > 1e-4, 1.0 - dark / safe[..., None], 1.0)
    return np.dstack([np.clip(rgb, 0, 1), np.clip(out_a, 0, 1)])


def _upscale(plane, h, w, box):
    """The (y0, y1, x0, x1) window of a bilinear resize to (h, w) of one channel, for a shadow so soft that its
    full-resolution detail would be wasted. Plane by plane, whose long inner loops run several times faster."""
    for axis, n, lo, hi in ((0, h, box[0], box[1]), (1, w, box[2], box[3])):
        u = np.clip((np.arange(lo, hi) + 0.5) * plane.shape[axis] / n - 0.5, 0.0, plane.shape[axis] - 1.0)
        i = np.minimum(u.astype(int), plane.shape[axis] - 2)
        t = (u - i).astype(np.float32)
        t = t[:, None] if axis == 0 else t
        plane = np.take(plane, i, axis = axis) * (1 - t) + np.take(plane, i + 1, axis = axis) * t
    return plane


def _path_data(segments):
    return ''.join('Z' if s[0] == 'h' else {'m': 'M', 'l': 'L', 'c': 'C'}[s[0]]
                   + ' '.join('%.2f' % v for v in s[1:]) for s in segments)


def _hex(rgb):
    return '#%02X%02X%02X' % tuple(max(0, min(255, int(round(v * 255)))) for v in rgb)


def _label_groups(c, ops, height):
    """Labels as matplotlib writes text: a comment naming them, then one group whose transform starts at their position
    in
    page coordinates. glycowork's annotate_figure swaps such groups for glycan drawings."""
    groups = {}
    for op in ops:
        groups.setdefault(op['label'], []).append(op)
    out = []
    for key, members in groups.items():
        a, b, cc, d, e, f = members[0]['ctm']
        paths = []
        for op in members:
            if op['kind'] == 'path':
                paths.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" stroke-linecap="round" '
                             'stroke-linejoin="round" />'
                             % (_path_data(op['path']), _hex(op['stroke_rgb']), op['line_width']))
            else:
                glyphs = text.outline(c, op['ttf'], op['text'], op['x'], op['y'], op['size'])
                paths.append('<path d="%s" fill="%s" />' % (_path_data(glyphs.ops), _hex(op['fill_rgb'])))
        comment = '<!-- %s -->\n' % key[2] if '--' not in key[2] else ''
        out.append('%s<g transform="translate(%.2f %.2f) matrix(%.6g %.6g %.6g %.6g 0 0)">%s</g>'
                   % (comment, e, height - f, a, -b, cc, -d, ''.join(paths)))
    return '\n'.join(out)


def _with_ids(svg, ops):
    """Wrap each element's run of output in a group with its `gid`. The emitted body has one line per drawn op."""
    named = [o.get('gid') for o in ops if (o['fill'] or o['stroke'] if o['kind'] == 'path' else
                                           o['kind'] == 'image' or o['text'].strip())]
    start = svg.index('\n', svg.index('<g transform="matrix(')) + 1
    end = svg.rindex('\n</g>\n</svg>')
    lines = svg[start:end].split('\n')
    if len(lines) != len(named) or not any(named):
        return svg
    out, current = [], None
    for line, gid in zip(lines, named):
        if gid != current:
            if current:
                out.append('</g>')
            if gid:
                out.append('<g id="%s">' % gid)
            current = gid
        out.append(line)
    if current:
        out.append('</g>')
    return svg[:start] + '\n'.join(out) + svg[end:]


class Figure:
    def __init__(self, width = 8.0, height = 8.0, palette = None):
        """A figure of the given size in inches. Its palette decides the type, axes, and special effects."""
        self.pal = get_palette(palette)
        self.width, self.height = width * 72.0, height * 72.0
        self._base = (self.width, self.height)
        self.axes = []
        self._title, self._subtitle = '', ''
        self._grid = None

    def suptitle(self, title, subtitle = ''):
        """The figure's title in the palette's manner, with an optional line under it."""
        self._title, self._subtitle = title, subtitle
        return self

    def add_axes(self, rect):
        """Axes at a fixed (left, bottom, width, height) rectangle in figure fractions."""
        ax = Axes(self, self.pal)
        ax._fixed = rect
        self.axes.append(ax)
        return ax

    def _head(self):
        return 66.0 if self._title and self._subtitle else (46.0 if self._title else 10.0)

    def _layout(self):
        """Fit each axes into its grid cell, leaving exactly the room its labels need."""
        for ax in self.axes:
            ax._limits()
        # A first pass places the axes so that labels can be fitted to them, and later ones place them around the result
        for _ in range(5):
            self._place()
            if not any([ax._fit_labels() for ax in self.axes]):
                break

    def _place(self):
        """Rectangles for all axes. Labels that would take most of a cell, or run off the page, widen the figure instead
        of
        squeezing the plot, as matplotlib's bbox_inches = 'tight' does."""
        pad, gap, head = 16.0, 10.0, self._head()
        self.width, self.height = self._base
        if self._title:
            pal = self.pal
            self.width = max(self.width, 32.0 + pad + max(text.width(self._title, pal.title_size, pal.title_bold),
                                                          text.width(self._subtitle, pal.subtitle_size)))
        if self._grid is not None:
            nrows, ncols, wr, hr = self._grid
            extra = np.zeros((ncols, 2))
            for ax in self.axes:
                if getattr(ax, '_fixed', None) is None:
                    l, r, b, t = ax._margins()
                    row, col = ax._cell
                    extra[col] = np.maximum(extra[col], ax._over)
                    grow_w = (l + r) / PLOT_SHARE_MARGINS / (wr[col] / sum(wr)) + 2 * pad + gap * (ncols - 1)
                    grow_h = (b + t) / PLOT_SHARE_MARGINS / (hr[row] / sum(hr)) + pad + head + gap * (nrows - 1)
                    self.width = min(max(self.width, grow_w), self._base[0] * 3)
                    self.height = min(max(self.height, grow_h), self._base[1] * 3)
            total_w = self.width - 2 * pad - gap * (ncols - 1)
            total_h = self.height - pad - head - gap * (nrows - 1)
            cw, ch = np.asarray(wr, float) / sum(wr) * total_w, np.asarray(hr, float) / sum(hr) * total_h
            self.width += extra.sum()
        for ax in self.axes:
            l, r, b, t = ax._margins()
            if getattr(ax, '_fixed', None) is not None:
                fx, fy, fw, fh = ax._fixed
                ax.rect = (fx * self.width, fy * self.height, fw * self.width, fh * self.height)
            else:
                row, col = ax._cell
                cx = pad + sum(cw[:col]) + gap * col + extra[:col].sum() + extra[col, 0]
                cy = pad + sum(ch[row + 1:]) + gap * (nrows - 1 - row)
                ax.rect = (cx + l, cy + b, max(cw[col] - l - r, 10.0), max(ch[row] - b - t, 10.0))
                ax._legend_y = cy + 10.0
        # Dendrograms share their matrix's rows or columns, whatever room its labels took
        for ax in self.axes:
            for axis, other in getattr(ax, '_align', {}).items():
                x, y, w, h = ax.rect
                ox, oy, ow, oh = other.rect
                ax.rect = (ox, y, ow, h) if axis == 'x' else (x, oy, w, oh)

    def _canvas(self, stamp_dpi = None, marker_forms = False, layout = True):
        """Everything drawn onto a fresh display list; `stamp_dpi` names a raster output, where many discs become an
        image, and `marker_forms` a PDF, where markers of one size can be one stored shape."""
        if layout:
            self._layout()
        c = pdfwrite.Canvas(BytesIO(), pagesize = (self.width, self.height))
        c.stamp_dpi = stamp_dpi
        c.marker_forms = marker_forms
        pal = self.pal
        for ax in self.axes:
            ax._draw_axes(c)
        for ax in self.axes:
            ax._draw_marks(c)
        if self._title:
            y = self.height - 38.0
            text.draw(c, 30.0, y, self._title, pal.title_size, pal.color(pal.title_color or pal.ink), pal.title_bold)
            if self._subtitle:
                caps = pal.extra.get('subtitle_caps')
                text.draw(c, 32.0, y - 20.0, self._subtitle.upper() if caps else self._subtitle, pal.subtitle_size,
                          pal.color(pal.subtitle_color or pal.muted), pal.subtitle_bold,
                          tracking = 0.9 if caps else 0.0)
        for ax in self.axes:
            ax._draw_text(c)
        return c

    def _shadow_op(self, dpi, layout = True):
        """The palette's soft shadow as an image op to go under everything else, or None for a palette without one."""
        if not self.pal.shadow:
            return None
        res = min(dpi, SHADOW_DPI)
        c = self._canvas(stamp_dpi = res, layout = layout)
        marks = [op for op in c.ops if op.get('mark')]
        if not marks:
            return None
        arr = _shadow_layer(marks, self.width, self.height, res, self.pal.shadow)
        # Only the box that some alpha survives rounding in, on the output's own grid, so that it composites fast
        seen = arr[..., 3] * 255.0 + 0.5 >= 1.0
        ys, xs = np.flatnonzero(seen.any(axis = 1)), np.flatnonzero(seen.any(axis = 0))
        if not len(ys):
            return None
        sc = dpi / 72.0
        h, w = int(round(self.height * sc)), int(round(self.width * sc))
        fy, fx = h / arr.shape[0], w / arr.shape[1]
        y0, y1 = max(0, int((ys[0] - 0.5) * fy - 0.5)), min(h, int((ys[-1] + 1.5) * fy + 0.5) + 1)
        x0, x1 = max(0, int((xs[0] - 0.5) * fx - 0.5)), min(w, int((xs[-1] + 1.5) * fx + 0.5) + 1)
        px = []
        for k in range(4):
            plane = arr[y0:y1, x0:x1, k] if dpi <= SHADOW_DPI else _upscale(arr[..., k], h, w, (y0, y1, x0, x1))
            px.append((plane * 255.0 + 0.5).astype(np.uint8))
        op = dict(c.state)
        op.update(kind = 'image', w = x1 - x0, h = y1 - y0, rgb = np.dstack(px[:3]).tobytes(), alpha = px[3].tobytes(),
                  ctm = ((x1 - x0) / sc, 0.0, 0.0, (y1 - y0) / sc, x0 / sc, (h - y1) / sc))
        return op

    def to_png(self, dpi = 300.0, background = 'white'):
        """PNG bytes; background None keeps the page transparent."""
        self._layout()
        # The shadow and the page share nothing once laid out, and numpy and zlib release the GIL for most of their work
        with ThreadPoolExecutor(1) as pool:
            shadow = pool.submit(self._shadow_op, dpi, False)
            c = self._canvas(stamp_dpi = dpi, layout = False)
            op = shadow.result()
        if op is not None:
            c.ops.insert(0, op)
        return c.to_png(dpi / 72.0, dpi / 72.0, (1.0, 1.0, 1.0) if background == 'white' else background)

    def to_pdf(self):
        """PDF bytes, with marks and text as vector; only a blurred shadow and very large heatmaps are images."""
        target = BytesIO()
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas(marker_forms = True)
        c.target = target
        if op is not None:
            c.ops.insert(0, op)
        c.save()
        return target.getvalue()

    def to_svg(self):
        """SVG text on a white page, with text as outlines and images embedded."""
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas()
        if op is not None:
            c.ops.insert(0, op)
        labels = [o for o in c.ops if o.get('label')]
        c.ops = [o for o in c.ops if not o.get('label')]
        svg = _with_ids(c.to_svg(), c.ops)
        page = '<rect width="%g" height="%g" fill="#FFFFFF" />\n' % (self.width, self.height)
        svg = svg.replace('<g transform="matrix(', page + '<g transform="matrix(', 1)
        # Tells slate this is a figure, ahead of the gradients' long <defs> and through annotate_figure. It cannot be a
        # comment, as annotate_figure takes every comment and all that follows up to the next </g> for a label
        top = svg.index('\n', svg.index('<svg')) + 1
        svg = svg[:top] + '<metadata>viscose</metadata>\n' + svg[top:]
        return svg.replace('</svg>', _label_groups(c, labels, self.height) + '\n</svg>') if labels else svg

    def save(self, path, dpi = 300.0, background = 'white'):
        """Write the file format that the path's suffix names: png, pdf, or svg."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == '.png':
            path.write_bytes(self.to_png(dpi, background))
        elif suffix == '.pdf':
            path.write_bytes(self.to_pdf())
        elif suffix == '.svg':
            path.write_text(self.to_svg(), encoding = 'utf-8')
        else:
            raise ValueError(f'Unsupported format {suffix!r}; use .png, .pdf, or .svg')
        return path

    def show(self, dpi = 144.0, viewer = True):
        """Show the figure: inline in Jupyter or slate, or in the system's image viewer from a terminal. A library that
        has just saved the figure passes viewer = False, or a script saving many figures opens a window for each."""
        show_png(lambda: self.to_png(dpi), viewer)

    def _repr_png_(self):
        return self.to_png(144.0)


def subplots(nrows = 1, ncols = 1, figsize = (8.0, 8.0), palette = None, width_ratios = None, height_ratios = None):
    """A figure and its axes, as in matplotlib: one Axes, a 1D array for a single row or column, or a 2D array."""
    fig = Figure(figsize[0], figsize[1], palette)
    fig._grid = (nrows, ncols, width_ratios or [1] * ncols, height_ratios or [1] * nrows)
    rows = []
    for r in range(nrows):
        row = []
        for j in range(ncols):
            ax = Axes(fig, fig.pal)
            ax._fixed = None
            ax._cell = (r, j)
            fig.axes.append(ax)
            row.append(ax)
        rows.append(row)
    if nrows == 1 and ncols == 1:
        return fig, rows[0][0]
    grid = np.array(rows, dtype = object)
    return fig, grid.ravel() if nrows == 1 or ncols == 1 else grid


def _given_labels(given, names, n):
    """Tick labels in any of seaborn's forms: a list, True for all, False for none, or an integer for every n-th. Gives
    the labels, the step between those shown, and whether they all stay."""
    if given is False:
        return None, 1, False
    if isinstance(given, (bool, int, np.integer)):
        return list(range(n)) if names is None else names, int(given), True
    return names if given is None else given, 1, False


def clustermap(data, xticklabels = None, yticklabels = None, method = 'average', metric = 'euclidean',
               cmap = 'diverging',
               figsize = (8.0, 8.0), palette = None, title = '', subtitle = '', show_all = False, row_linkage = None,
               col_linkage = None, row_cluster = True, col_cluster = True, vmin = None, vmax = None, center = 0.0,
               xlabel = '', ylabel = '', colorbar = True, colorbar_label = ''):
    """A heatmap with both dendrograms and a color bar, as in seaborn. Needs scipy. A DataFrame brings its own labels,
    and xticklabels and yticklabels can be True for all, False for none, or an integer for every n-th. Returns the
    figure and the heatmap's axes; the figure's row_order and col_order hold the data's rows and columns as drawn,
    first row at the top."""
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
    except ImportError as e:
        raise ImportError('clustermap needs scipy: pip install "viscose[cluster]"') from e
    frame = hasattr(data, 'index') and hasattr(data, 'columns')
    m = np.asarray(data, dtype = float)
    x, x_step, x_all = _given_labels(xticklabels, list(data.columns) if frame else None, m.shape[1])
    y, y_step, y_all = _given_labels(yticklabels, list(data.index) if frame else None, m.shape[0])
    # Scipy walks the tree recursively, one level per row in the worst case
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 2 * max(m.shape) + 200))
    try:
        rows = cols = None
        if row_cluster:
            rows = dendrogram(linkage(m, method, metric,
                                      optimal_ordering = True) if row_linkage is None else row_linkage, no_plot = True)
        if col_cluster:
            cols = dendrogram(linkage(m.T, method, metric,
                                      optimal_ordering = True) if col_linkage is None else col_linkage, no_plot = True)
    finally:
        sys.setrecursionlimit(limit)
    row_order = rows['leaves'][::-1] if row_cluster else list(range(m.shape[0]))
    col_order = cols['leaves'] if col_cluster else list(range(m.shape[1]))
    fig, axes = subplots(2, 2, figsize, palette, width_ratios = [1, 7], height_ratios = [1, 9])
    fig.suptitle(title, subtitle)
    heat = axes[1][1].heatmap(m[row_order][:, col_order],
                              None if x is None else [x[i] if j % x_step == 0 else '' for j, i in enumerate(col_order)],
                              None if y is None else [y[i] if j % y_step == 0 else '' for j, i in enumerate(row_order)],
                              cmap = cmap, show_all = (show_all or x_all, show_all or y_all), vmin = vmin, vmax = vmax,
                              center = center)
    heat.set_xlabel(xlabel)
    heat.set_ylabel(ylabel)
    top = axes[0][1].dendrogram(cols['icoord'], cols['dcoord'], 'top', m.shape[1]) if col_cluster else axes[0][1].off()
    left = axes[1][0].dendrogram(rows['icoord'], rows['dcoord'], 'left',
                                 m.shape[0]) if row_cluster else axes[1][0].off()
    corner = axes[0][0].off()
    if colorbar:
        corner.colorbar(colorbar_label, 'fill', heat)
    fig.axes = [heat, top, left, corner]
    top._align, left._align = {'x': heat}, {'y': heat}
    fig.row_order, fig.col_order = row_order, col_order
    return fig, heat
