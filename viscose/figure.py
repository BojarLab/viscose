"""Figure: lays out axes, draws through glycorender's display list and writes PNG, PDF, or SVG."""
import builtins
import functools
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import numpy as np
from glycorender import pdfwrite, raster
from . import marks, text
from .axes import Axes, LABEL_MIN
from .palettes import get_palette


# Shadow offset and blur in points; a blur cannot stay vector, so the PDF and SVG carry it as an image at this
# resolution
SHADOW_DX, SHADOW_DY, SHADOW_BLUR, SHADOW_DPI = 1.2, 5.0, 4.7, 100.0
# Margins may take this share of an axes' cell before the figure grows
PLOT_SHARE_MARGINS = 0.55
# Coding agents and CI can hold a real console, so a terminal alone does not mean someone wants a window per figure
NO_VIEWER = ('CLAUDECODE', 'CI', 'PYTEST_CURRENT_TEST', 'VISCOSE_NO_VIEWER')
# What clustermap clusters by, as scipy names them
LINKAGES = ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')
METRICS = ('euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra',
           'jensenshannon')
# Elements per block of a min-plus product, which keeps each block in the processor's cache
MINPLUS_BLOCK = 1 << 16
# Points per unit of a page's size, and of a position given in millimeters from a page's top left
UNITS = {'in': 72.0, 'mm': 72.0 / 25.4}


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
    marks', and which over a dark page shows as a glow."""
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
    in page coordinates. glycowork's annotate_figure swaps such groups for glycan drawings, and data-right tells it how
    far on the page those it leaves as text reach."""
    groups = {}
    for op in ops:
        groups.setdefault(op['label'], []).append(op)
    out = []
    for key, members in groups.items():
        a, b, cc, d, e, f = members[0]['ctm']
        paths, right = [], 0.0
        for op in members:
            if op['kind'] == 'path':
                segments = op['path']
                paths.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" stroke-linecap="round" '
                             'stroke-linejoin="round" />'
                             % (_path_data(segments), _hex(op['stroke_rgb']), op['line_width']))
            else:
                segments = text.outline(c, op['ttf'], op['text'], op['x'], op['y'], op['size']).ops
                paths.append('<path d="%s" fill="%s" />' % (_path_data(segments), _hex(op['fill_rgb'])))
            right = max([right] + [x for s in segments for x in s[1::2]])
        comment = '<!-- %s -->\n' % key[2] if '--' not in key[2] else ''
        out.append('%s<g transform="translate(%.2f %.2f) matrix(%.6g %.6g %.6g %.6g 0 0)" data-right="%.2f">%s</g>'
                   % (comment, e, height - f, a, -b, cc, -d, e + a * right, ''.join(paths)))
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


def _in_font(method):
    """Layout and drawing measure and set text in the family of the figure's palette."""
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        with text.family(self.pal.font):
            return method(self, *args, **kwargs)
    return call


def _unit(unit, allowed):
    """A unit's name, as any other would quietly be read as one of these."""
    if unit not in allowed:
        raise ValueError(f"unit has to be {' or '.join(map(repr, allowed))}, got {unit!r}")
    return unit


class Figure:
    def __init__(self, width = 8.0, height = 8.0, palette = None, unit = 'in'):
        """A figure of the given size, in inches or with unit = 'mm' in millimeters, as a journal gives its columns
        (89 and 183 mm for one and two in Nature's). Its palette decides the type, axes, and special effects."""
        self.pal = get_palette(palette)
        k = UNITS[_unit(unit, ('in', 'mm'))]
        self.width, self.height = width * k, height * k
        self._base = (self.width, self.height)
        self.axes = []
        self._calls, self._inside = [], 0
        self._title, self._subtitle = '', ''
        self._grid, self._held = None, False
        self._texts = []

    def suptitle(self, title, subtitle = ''):
        """The figure's title in the palette's manner, with an optional line under it."""
        self._title, self._subtitle = title, subtitle
        return self

    def in_palette(self, palette):
        """A new figure in another palette, as if drawn in that one from the start: the plot calls that made this one
        run again. Shades, `ink`, and `muted` follow the palette, and colors given as hex or by CSS name stay."""
        fig = Figure(palette = palette)
        fig._base, fig.width, fig.height = self._base, *self._base
        fig._title, fig._subtitle, fig._grid, fig._held = self._title, self._subtitle, self._grid, self._held
        fig._texts = list(self._texts)
        twins = {id(ax): Axes(fig, fig.pal) for ax in self.axes}
        for ax in self.axes:
            twin = twins[id(ax)]
            twin._cell, twin._span, twin._fixed = ax._cell, ax._span, ax._fixed
            twin._align = {axis: twins[id(other)] for axis, other in ax._align.items()}
            fig.axes.append(twin)
        swap = lambda v: twins[id(v)] if isinstance(v, Axes) else v
        for ax, name, args, kwargs in self._calls:
            getattr(twins[id(ax)], name)(*map(swap, args), **{k: swap(v) for k, v in kwargs.items()})
        if hasattr(self, 'row_order'):
            fig.row_order, fig.col_order = self.row_order, self.col_order
        return fig

    def add_axes(self, rect, unit = 'fraction'):
        """Axes whose plot area is a fixed rectangle: (left, bottom, width, height) in figure fractions, or with
        unit = 'mm' (left, top, width, height) in millimeters from the page's top left, as a journal's layout is drawn.
        Their labels go around it."""
        ax = Axes(self, self.pal)
        ax._fixed = (*rect, _unit(unit, ('fraction', 'mm')))
        self.axes.append(ax)
        return ax

    def text(self, x, y, s, size = None, color = 'ink', bold = None, ha = 'left', va = 'baseline', rotation = 0.0,
             unit = 'fraction', linespacing = 1.25):
        """Text on the page, outside any axes: at (x, y) in figure fractions from the lower left, or with unit = 'mm'
        in millimeters from the top left, with several lines `linespacing` sizes apart. Without a size it takes the
        palette's for axis names."""
        self._texts.append(dict(x = float(x), y = float(y), s = str(s), size = size, color = color, bold = bold,
                                ha = ha, va = va, rotation = rotation, unit = _unit(unit, ('fraction', 'mm')),
                                title = None, linespacing = linespacing))
        return self

    def letter(self, x, y, char, title = '', size = None, unit = 'mm'):
        """A panel's letter, bold and with its top at (x, y), in millimeters from the page's top left or with
        unit = 'fraction' in figure fractions, and an optional heading on its baseline beside it, in the palette's
        manner of axis names. Without a size it is three points larger than those."""
        self._texts.append(dict(x = float(x), y = float(y), s = str(char), size = size, unit = _unit(unit, ('fraction',
                                'mm')), title = str(title)))
        return self

    def _head(self):
        pal = self.pal
        if not self._title:
            return 10.0
        room = 46.0 + 1.25 * pal.title_size * self._title.count('\n')
        return room + 20.0 + 1.25 * pal.subtitle_size * self._subtitle.count('\n') if self._subtitle else room

    def set_size(self, width, height):
        """A new size in inches, for a figure that follows a window: the page keeps exactly this size from now on, and
        labels that would otherwise widen it turn, thin out, or squeeze the plot."""
        self._base = self.width, self.height = width * 72.0, height * 72.0
        self._held = True
        return self

    def layout(self):
        """Settle limits, ticks, and positions now, for placing things by `Axes.page`; saving does this by itself."""
        self._layout()
        return self

    @_in_font
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
        of squeezing the plot, as matplotlib's bbox_inches = 'tight' does."""
        pad, gap, head = 16.0, 10.0, self._head()
        self.width, self.height = self._base
        if self._title and not self._held:
            pal = self.pal
            self.width = max(self.width, 32.0 + pad + max(text.width(self._title, pal.title_size, pal.title_bold),
                                                          text.width(self._subtitle, pal.subtitle_size)))
        if self._grid is not None:
            nrows, ncols, wr, hr = self._grid
            extra = np.zeros((ncols, 2))
            for ax in self.axes:
                if ax._fixed is None and not self._held:
                    l, r, b, t = ax._margins()
                    (row, col), (rs, cs) = ax._cell, ax._span
                    # What overhangs on the left goes into the first column of the axes, and on the right its last
                    extra[col, 0], extra[col + cs - 1, 1] = (max(extra[col, 0], ax._over[0]),
                                                             max(extra[col + cs - 1, 1], ax._over[1]))
                    share_w, share_h = sum(wr[col:col + cs]) / sum(wr), sum(hr[row:row + rs]) / sum(hr)
                    grow_w = (l + r) / PLOT_SHARE_MARGINS / share_w + 2 * pad + gap * (ncols - 1)
                    grow_h = (b + t) / PLOT_SHARE_MARGINS / share_h + pad + head + gap * (nrows - 1)
                    # Axis names that would still not fit at their smallest
                    name_w, name_h = ax._axis_name_widths(LABEL_MIN)
                    grow_w = max(grow_w, (name_w + l + r) / share_w + 2 * pad + gap * (ncols - 1))
                    grow_h = max(grow_h, (name_h + b + t) / share_h + pad + head + gap * (nrows - 1))
                    self.width = min(max(self.width, grow_w), self._base[0] * 3)
                    self.height = min(max(self.height, grow_h), self._base[1] * 3)
                    # Labels that must all show get the room they need, however much that is
                    sx, sy = ax._tick_sizes()
                    if ax.show_all[0] and ax.xrot:
                        need = len(ax.xticks) * sx * 1.05 + l + r
                        self.width = max(self.width, need / share_w + 2 * pad + gap * (ncols - 1))
                    if ax.show_all[1]:
                        need = len(ax.yticks) * sy * 1.05 + b + t
                        self.height = max(self.height, need / share_h + pad + head + gap * (nrows - 1))
            total_w = self.width - 2 * pad - gap * (ncols - 1)
            total_h = self.height - pad - head - gap * (nrows - 1)
            cw, ch = np.asarray(wr, float) / sum(wr) * total_w, np.asarray(hr, float) / sum(hr) * total_h
            self.width += extra.sum()
        for ax in self.axes:
            l, r, b, t = ax._margins()
            if ax._fixed is not None:
                fx, fy, fw, fh, unit = ax._fixed
                mm = UNITS['mm']
                ax.rect = ((fx * mm, self.height - (fy + fh) * mm, fw * mm, fh * mm) if unit == 'mm' else
                           (fx * self.width, fy * self.height, fw * self.width, fh * self.height))
                ax._room = (0.0, 0.0, self.width, self.height)
            else:
                (row, col), (rs, cs) = ax._cell, ax._span
                cx = pad + sum(cw[:col]) + gap * col + extra[:col].sum() + extra[col, 0]
                cy = pad + sum(ch[row + rs:]) + gap * (nrows - row - rs)
                # A cell spanning columns takes the gaps between them and what they grew by inside it
                wide = cw[col] if cs == 1 else (sum(cw[col:col + cs]) + gap * (cs - 1) + extra[col:col + cs].sum()
                                                - extra[col, 0] - extra[col + cs - 1, 1])
                tall = ch[row] if rs == 1 else sum(ch[row:row + rs]) + gap * (rs - 1)
                ax.rect = (cx + l, cy + b, max(wide - l - r, 10.0), max(tall - b - t, 10.0))
                ax._room = (cx, cy, wide, tall)
                ax._legend_y = cy + 10.0
        # Dendrograms share their matrix's rows or columns, whatever room its labels took
        for ax in self.axes:
            for axis, other in ax._align.items():
                x, y, w, h = ax.rect
                ox, oy, ow, oh = other.rect
                ax.rect = (ox, y, ow, h) if axis == 'x' else (x, oy, w, oh)

    def _width_unlabeled(self):
        """The width of the settled page without the room glycan labels took on its right. Only the last column's growth
        can go, as earlier ones have other columns to their right and axes at fixed fractions move with the page."""
        if self._grid is None or any(ax._fixed is not None for ax in self.axes):
            return self.width
        last = [ax for ax in self.axes if ax._cell[1] + ax._span[1] == self._grid[1]]
        grown = max([ax._over[1] for ax in last], default = 0.0)
        needed = max([ax._over_plain for ax in last], default = -math.inf)
        return min(self.width, max(self.width - grown, self.width + needed))

    @_in_font
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
                y -= 1.25 * pal.title_size * self._title.count('\n')
                text.draw(c, 32.0, y - 20.0, self._subtitle.upper() if caps else self._subtitle, pal.subtitle_size,
                          pal.color(pal.subtitle_color or pal.muted), pal.subtitle_bold,
                          tracking = 0.9 if caps else 0.0)
        for ax in self.axes:
            ax._draw_text(c)
        for t in self._texts:
            x, y = ((t['x'] * UNITS['mm'], self.height - t['y'] * UNITS['mm']) if t['unit'] == 'mm' else
                    (t['x'] * self.width, t['y'] * self.height))
            if t['title'] is None:
                text.draw(c, x, y, t['s'], t['size'] or pal.label_size, pal.color(t['color']),
                          pal.text_bold if t['bold'] is None else t['bold'], t['ha'], t['va'], t['rotation'],
                          linespacing = t['linespacing'])
                continue
            size = t['size'] or pal.label_size + 3.0
            text.draw(c, x, y, t['s'], size, pal.color(pal.title_color or pal.ink), True, va = 'top')
            # The heading starts a letter's width and a space on, on the letter's baseline
            text.draw(c, x + text.width(t['s'], size, True) + 0.45 * size, y - text.CAP_HEIGHT * size,
                      text.spaced(pal, t['title']), pal.label_size, pal.color(pal.label_color or pal.ink),
                      pal.label_bold, tracking = 0.9 if pal.caps else 0.0)
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

    def _raster_canvas(self, dpi):
        self._layout()
        # The shadow and the page share nothing once laid out, and numpy and zlib release the GIL for most of their work
        with ThreadPoolExecutor(1) as pool:
            shadow = pool.submit(self._shadow_op, dpi, False)
            c = self._canvas(stamp_dpi = dpi, layout = False)
            op = shadow.result()
        if op is not None:
            c.ops.insert(0, op)
        return c

    def to_png(self, dpi = 300.0, background = 'paper'):
        """PNG bytes; background is any color, by default the palette's page, and None keeps the page transparent."""
        return self._raster_canvas(dpi).to_png(dpi / 72.0, dpi / 72.0, self.pal.color(background))

    def to_rgba(self, dpi = 144.0, background = 'paper'):
        """The page as an array of shape (rows, columns, 4) of 8-bit RGBA, first row at the top, for showing it in a
        window without encoding a PNG; background is any color, by default the palette's page, and None keeps the page
        transparent."""
        page = raster.render(self._raster_canvas(dpi).ops, self.width, self.height, dpi / 72.0, dpi / 72.0,
                             self.pal.color(background))
        return page if page.shape[2] == 4 else np.dstack([page, np.full(page.shape[:2], 255, np.uint8)])

    def display_list(self, layout = True):
        """The page as glycorender's display list, each op in page points with its own state, for a window that paints
        with a painter of its own, which is much faster at high resolution; a palette's soft shadow and its page's color
        are not part of it. layout = False takes the page as `layout` last settled it."""
        return self._canvas(layout = layout).ops

    def to_pdf(self):
        """PDF bytes, with marks and text as vector; only a blurred shadow and very large heatmaps are images."""
        target = BytesIO()
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas(marker_forms = True)
        c.target = target
        if op is not None:
            c.ops.insert(0, op)
        # A PDF's page is white unless something covers it
        if self.pal.color('paper') != (1.0, 1.0, 1.0):
            marks.polygon(c, [0.0, self.width, self.width, 0.0], [0.0, 0.0, self.height, self.height],
                          self.pal.color('paper'))
            c.ops.insert(0, c.ops.pop())
        c.save()
        return target.getvalue()

    def to_svg(self):
        """SVG text on the palette's page, with text as outlines and images embedded."""
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas()
        if op is not None:
            c.ops.insert(0, op)
        labels = [o for o in c.ops if o.get('label')]
        c.ops = [o for o in c.ops if not o.get('label')]
        svg = _with_ids(c.to_svg(), c.ops)
        # With glycan labels, the width the page needs without them, down to which annotate_figure can crop it
        bare = ' data-width-unlabeled="%.2f"' % self._width_unlabeled() if labels else ''
        page = '<rect width="%g" height="%g" fill="%s"%s />\n' % (self.width, self.height, self.pal.css('paper'), bare)
        svg = svg.replace('<g transform="matrix(', page + '<g transform="matrix(', 1)
        # Tells slate this is a figure, ahead of the gradients' long <defs> and through annotate_figure. It cannot be a
        # comment, as annotate_figure takes every comment and all that follows up to the next </g> for a label
        top = svg.index('\n', svg.index('<svg')) + 1
        svg = svg[:top] + '<metadata>viscose</metadata>\n' + svg[top:]
        return svg.replace('</svg>', _label_groups(c, labels, self.height) + '\n</svg>') if labels else svg

    def save(self, path, dpi = 300.0, background = 'paper'):
        """Write the file format that the path's suffix names: png, pdf, or svg. `dpi` and `background` are for PNG, as
        PDF and SVG are vector on the palette's page."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == '.png':
            path.write_bytes(self.to_png(dpi, background))
        elif suffix == '.pdf':
            path.write_bytes(self.to_pdf())
        elif suffix == '.svg':
            path.write_text(self.to_svg(), encoding = 'utf-8', newline = '\n')
        else:
            raise ValueError(f'Unsupported format {suffix!r}; use .png, .pdf, or .svg')
        return path

    def show(self, dpi = 144.0, viewer = True):
        """Show the figure: inline in Jupyter or slate, or in the system's image viewer from a terminal. A library that
        has just saved the figure passes viewer = False, or a script saving many figures opens a window for each."""
        show_png(lambda: self.to_png(dpi), viewer)

    def _repr_png_(self):
        return self.to_png(144.0)


def subplots(nrows = 1, ncols = 1, figsize = (8.0, 8.0), palette = None, width_ratios = None, height_ratios = None,
             unit = 'in'):
    """A figure and its axes, as in matplotlib: one Axes, a 1D array for a single row or column, or a 2D array.
    `figsize` is in inches, or with unit = 'mm' in millimeters."""
    fig = Figure(figsize[0], figsize[1], palette, unit)
    fig._grid = (nrows, ncols, width_ratios or [1] * ncols, height_ratios or [1] * nrows)
    rows = []
    for r in range(nrows):
        row = []
        for j in range(ncols):
            ax = Axes(fig, fig.pal)
            ax._cell = (r, j)
            fig.axes.append(ax)
            row.append(ax)
        rows.append(row)
    if nrows == 1 and ncols == 1:
        return fig, rows[0][0]
    grid = np.array(rows, dtype = object)
    return fig, grid.ravel() if nrows == 1 or ncols == 1 else grid


def subplot_mosaic(mosaic, figsize = (8.0, 8.0), palette = None, width_ratios = None, height_ratios = None,
                   unit = 'in'):
    """A figure whose axes take rectangles of a grid, as in matplotlib: `mosaic` is a list of rows, each a list or a
    string of labels, or one string with rows split by ';', where a label repeated over a rectangle of cells makes one
    axes spanning them and '.' leaves a cell empty, such as [['a', 'b', 'c'], ['d', 'd', 'd']] or 'abc;ddd'. Returns
    the figure and a dict of axes by label."""
    rows = [list(r) for r in (mosaic.split(';') if isinstance(mosaic, str) else mosaic)]
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('A mosaic needs rows of equal length')
    nrows, ncols = len(rows), len(rows[0])
    fig = Figure(figsize[0], figsize[1], palette, unit)
    fig._grid = (nrows, ncols, width_ratios or [1] * ncols, height_ratios or [1] * nrows)
    found = {}
    for r, row in enumerate(rows):
        for j, key in enumerate(row):
            if key != '.':
                found.setdefault(key, []).append((r, j))
    axes = {}
    for key, cells in found.items():
        (r0, j0), (r1, j1) = np.min(cells, axis = 0), np.max(cells, axis = 0)
        if len(cells) != (r1 - r0 + 1) * (j1 - j0 + 1):
            raise ValueError(f'Each label of a mosaic has to fill a rectangle of cells, and {key!r} does not')
        ax = Axes(fig, fig.pal)
        ax._cell, ax._span = (int(r0), int(j0)), (int(r1 - r0 + 1), int(j1 - j0 + 1))
        fig.axes.append(ax)
        axes[key] = ax
    return fig, axes


def _given_labels(given, names, n):
    """Tick labels in any of seaborn's forms: a list, True for all, False for none, or an integer for every n-th. Gives
    the labels, the step between those shown, and whether they all stay."""
    if given is False:
        return None, 1, False
    if isinstance(given, (bool, int, np.integer)):
        return list(range(n)) if names is None else names, int(given), True
    return names if given is None else given, 1, False


def linkage(data, method = 'average', metric = 'euclidean', optimal_ordering = True):
    """Hierarchical clustering of the rows of data, joining the closest two clusters each time, in scipy's format: per
    merge the two clusters (rows by index, then merges by their step plus the number of rows), their distance, and the
    new size. Methods and metrics are clustermap's. With optimal_ordering, each link's children come in the order that
    puts neighbors as close as the tree allows, found exactly. Distances and merges are scipy's, apart from the order of
    exactly tied merges, so the result also goes into scipy's own functions."""
    m = np.asarray(data, dtype = float)
    if m.ndim != 2 or len(m) < 2 or not np.isfinite(m).all():
        raise ValueError('Clustering needs a table of finite values with at least two rows')
    if method not in LINKAGES:
        raise ValueError(f"Unknown method '{method}'; use one of {', '.join(LINKAGES)}")
    if method in ('centroid', 'median', 'ward') and metric != 'euclidean':
        raise ValueError(f"Method '{method}' needs metric = 'euclidean'")
    if not callable(metric) and metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'; use one of {', '.join(METRICS)}, or a function of two rows")
    if metric == 'jensenshannon' and (m < 0).any():
        raise ValueError('The jensenshannon distance compares abundances, which cannot be negative')
    n = len(m)
    if callable(metric):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = metric(m[i], m[j])
    elif metric in ('cosine', 'correlation'):
        u = m - m.mean(1, keepdims = True) if metric == 'correlation' else m
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            u = u / np.sqrt((u * u).sum(1, keepdims = True))
        d = np.clip(1.0 - u @ u.T, 0.0, 2.0)
    else:
        d = np.empty((n, n))
        p = m / m.sum(1, keepdims = True) if metric == 'jensenshannon' else m
        # Rows a block at a time, so that their differences take at most about 32 MB
        block = max(1, 4_000_000 // max(n * m.shape[1], 1))
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            for i in range(0, n, block):
                a, b = p[i:i + block, None, :], p[None, :, :]
                if metric in ('euclidean', 'sqeuclidean'):
                    d[i:i + block] = ((a - b) ** 2).sum(-1)
                elif metric == 'cityblock':
                    d[i:i + block] = np.abs(a - b).sum(-1)
                elif metric == 'chebyshev':
                    d[i:i + block] = np.abs(a - b).max(-1)
                elif metric == 'braycurtis':
                    d[i:i + block] = np.abs(a - b).sum(-1) / np.abs(a + b).sum(-1)
                elif metric == 'canberra':
                    d[i:i + block] = np.nan_to_num(np.abs(a - b) / (np.abs(a) + np.abs(b)), nan = 0.0).sum(-1)
                else:
                    half = (a + b) / 2
                    terms = np.where(a > 0, a * np.log(a / half), 0.0) + np.where(b > 0, b * np.log(b / half), 0.0)
                    d[i:i + block] = np.sqrt(np.maximum(terms.sum(-1) / 2, 0.0))
        if metric == 'euclidean':
            np.sqrt(d, out = d)
    np.fill_diagonal(d, 0.0)
    if not np.isfinite(d).all():
        raise ValueError(f'The {metric} distance is undefined between some rows, such as constant ones for correlation '
                         'or rows of zeros for cosine and jensenshannon')
    # Centroid, median, and Ward update squared distances
    squared = method in ('centroid', 'median', 'ward')
    work = d ** 2 if squared else d.copy()
    np.fill_diagonal(work, np.inf)
    size, ids, alive = np.ones(n), np.arange(n), np.ones(n, bool)
    near = work.argmin(1)
    best = work[np.arange(n), near]
    z = np.empty((n - 1, 4))
    for step in range(n - 1):
        i = int(best.argmin())
        i, j = sorted((i, int(near[i])))
        a, b, dij, ni, nj = work[i], work[j], work[i, j], size[i], size[j]
        if method == 'single':
            new = np.minimum(a, b)
        elif method == 'complete':
            new = np.maximum(a, b)
        elif method == 'average':
            new = (ni * a + nj * b) / (ni + nj)
        elif method == 'weighted':
            new = (a + b) / 2
        elif method == 'centroid':
            new = (ni * a + nj * b) / (ni + nj) - ni * nj * dij / (ni + nj) ** 2
        elif method == 'median':
            new = a / 2 + b / 2 - dij / 4
        else:
            new = ((ni + size) * a + (nj + size) * b - size * dij) / (ni + nj + size)
        z[step] = min(ids[i], ids[j]), max(ids[i], ids[j]), np.sqrt(max(dij, 0.0)) if squared else dij, ni + nj
        new[i] = new[j] = np.inf
        work[i], work[:, i], work[j], work[:, j] = new, new, np.inf, np.inf
        size[i], ids[i], alive[j], best[j] = ni + nj, n + step, False, np.inf
        # Only rows whose nearest cluster was just merged need a new search; any other can only come closer to it
        stale = alive & ((near == i) | (near == j))
        stale[i] = True
        closer = ~stale & (new < best)
        near[closer], best[closer] = i, new[closer]
        rows = np.flatnonzero(stale)
        near[rows] = work[rows].argmin(1)
        best[rows] = work[rows, near[rows]]
    return _optimal_order(z, d) if optimal_ordering else z


def _minplus(p, q):
    """The min-plus product of two matrices, in blocks small enough to stay in the processor's cache."""
    a, b = p.shape
    c = q.shape[1]
    if b <= min(a, c):
        out, part = p[:, :1] + q[:1], np.empty((a, c))
        for k in range(1, b):
            np.minimum(out, np.add(p[:, k:k + 1], q[k:k + 1], out = part), out = out)
        return out
    out = np.empty((a, c))
    if c < a:
        # Few columns: each reduces along contiguous rows of p plus one column of q
        qt, rows = np.ascontiguousarray(q.T), max(1, MINPLUS_BLOCK // b)
        for k in range(c):
            for i in range(0, a, rows):
                np.min(p[i:i + rows] + qt[k], axis = 1, out = out[i:i + rows, k])
        return out
    cols = min(c, max(1, MINPLUS_BLOCK // b))
    rows = max(1, MINPLUS_BLOCK // (b * cols))
    for j in range(0, c, cols):
        for i in range(0, a, rows):
            np.min(p[i:i + rows, :, None] + q[:, j:j + cols], axis = 1, out = out[i:i + rows, j:j + cols])
    return out


def _ends(c, s, n, lo, mid, hi, cost):
    """Where an order of cluster c that starts at leaf s can end, by rank, and at what cost."""
    if c < n:
        return np.array([s]), np.zeros(1)
    if s < mid[c]:
        return np.arange(mid[c], hi[c]), cost[c][s - lo[c]]
    return np.arange(lo[c], mid[c]), cost[c][:, s - mid[c]]


def _optimal_order(z, d):
    """Turns links so that neighboring leaves lie as close as the tree allows, by the sum of distances between
    neighbors, which Bar-Joseph et al. (2001) find exactly in cubic time. Gives z with each link's children in drawing
    order."""
    n = len(z) + 1
    kids = z[:, :2].astype(int)
    # Leaves in the tree's own order, so that each cluster is one stretch from lo to hi, its first child up to mid
    seq, stack = [], [2 * n - 2]
    while stack:
        c = stack.pop()
        if c < n:
            seq.append(c)
        else:
            stack += [kids[c - n, 1], kids[c - n, 0]]
    rank = np.empty(n, int)
    rank[seq] = np.arange(n)
    lo, hi = np.empty(2 * n - 1, int), np.empty(2 * n - 1, int)
    lo[:n], hi[:n] = rank, rank + 1
    for i, (a, b) in enumerate(kids):
        lo[n + i], hi[n + i] = lo[a], hi[b]
    mid = np.concatenate([rank, hi[kids[:, 0]]])
    near = d[np.ix_(seq, seq)]
    # Per cluster, the least cost of an order that starts at u in its first child and ends at w in its second
    cost = {}
    for i, (a, b) in enumerate(kids):
        if a < n:
            part = near[lo[a]:hi[a], lo[b]:hi[b]]
        else:
            part = np.vstack([_minplus(cost[a], near[mid[a]:hi[a], lo[b]:hi[b]]),
                              _minplus(cost[a].T, near[lo[a]:mid[a], lo[b]:hi[b]])])
        if b >= n:
            k = mid[b] - lo[b]
            part = np.hstack([_minplus(part[:, k:], cost[b].T), _minplus(part[:, :k], cost[b])])
        cost[n + i] = part
    u, w = np.unravel_index(cost[2 * n - 2].argmin(), cost[2 * n - 2].shape)
    stack, turn = [(2 * n - 2, lo[-1] + u, mid[-1] + w)], np.zeros(n - 1, bool)
    while stack:
        c, s, e = stack.pop()
        if c < n:
            continue
        first, second = kids[c - n]
        if s >= mid[c]:
            turn[c - n], first, second = True, second, first
        xs, cx = _ends(first, s, n, lo, mid, hi, cost)
        ys, cy = _ends(second, e, n, lo, mid, hi, cost)
        x, y = np.unravel_index((cx[:, None] + near[np.ix_(xs, ys)] + cy).argmin(), (len(xs), len(ys)))
        stack += [(first, s, xs[x]), (second, ys[y], e)]
    z = z.copy()
    z[turn, :2] = z[turn, 1::-1]
    return z


def _tree(z):
    """What scipy's dendrogram gives for drawing: the leaves in order, and each link's corners, children first."""
    z = np.asarray(z, dtype = float)
    n = len(z) + 1
    leaves, icoord, dcoord, x, h = [], [], [], {}, {}
    stack = [(2 * n - 2, False)]
    while stack:
        c, done = stack.pop()
        if c < n:
            x[c], h[c] = 10.0 * len(leaves) + 5.0, 0.0
            leaves.append(c)
        elif done:
            a, b = int(z[c - n, 0]), int(z[c - n, 1])
            icoord.append([x[a], x[a], x[b], x[b]])
            dcoord.append([h[a], z[c - n, 2], z[c - n, 2], h[b]])
            x[c], h[c] = (x[a] + x[b]) / 2, z[c - n, 2]
        else:
            stack += [(c, True), (int(z[c - n, 1]), False), (int(z[c - n, 0]), False)]
    return {'leaves': leaves, 'icoord': icoord, 'dcoord': dcoord}


def clustermap(data, xticklabels = None, yticklabels = None, method = 'average', metric = 'euclidean',
               cmap = 'diverging', figsize = (8.0, 8.0), palette = None, title = '', subtitle = '', show_all = False,
               row_linkage = None, col_linkage = None, row_cluster = True, col_cluster = True, vmin = None, vmax = None,
               center = 0.0, xlabel = '', ylabel = '', colorbar = True, colorbar_label = ''):
    """A heatmap with both dendrograms and a color bar, as in seaborn. A DataFrame brings its own labels, and
    xticklabels and yticklabels can be True for all, False for none, or an integer for every n-th. Rows and columns are
    clustered by `method` (single, complete, average, weighted, centroid, median, or ward) on the `metric` distance
    (euclidean, sqeuclidean, cityblock, chebyshev, cosine, correlation, braycurtis, canberra, jensenshannon, or a
    function of two rows), and each tree's links turned so that neighbors are as alike as the tree allows. Linkages
    passed in scipy's format are drawn as they are. Returns the figure and the heatmap's axes; the figure's row_order
    and col_order hold the data's rows and columns as drawn, first row at the top."""
    frame = hasattr(data, 'index') and hasattr(data, 'columns')
    m = np.asarray(data, dtype = float)
    x, x_step, x_all = _given_labels(xticklabels, list(data.columns) if frame else None, m.shape[1])
    y, y_step, y_all = _given_labels(yticklabels, list(data.index) if frame else None, m.shape[0])
    # Each row's and column's own name, also where its label stays undrawn
    row_names = list(data.index) if y is None and frame else y
    column_names = list(data.columns) if x is None and frame else x
    # One row or column has nothing to be clustered with
    row_cluster, col_cluster = row_cluster and m.shape[0] > 1, col_cluster and m.shape[1] > 1
    if ((row_cluster and row_linkage is None) or (col_cluster and col_linkage is None)) and not np.isfinite(m).all():
        raise ValueError('Clustering needs finite values: fill in what is missing, pass row_linkage and col_linkage, '
                         'or set row_cluster and col_cluster to False')
    rows = cols = None
    if row_cluster:
        rows = _tree(linkage(m, method, metric) if row_linkage is None else row_linkage)
    if col_cluster:
        cols = _tree(linkage(m.T, method, metric) if col_linkage is None else col_linkage)
    row_order = rows['leaves'][::-1] if row_cluster else list(range(m.shape[0]))
    col_order = cols['leaves'] if col_cluster else list(range(m.shape[1]))
    fig, axes = subplots(2, 2, figsize, palette, width_ratios = [1, 7], height_ratios = [1, 9])
    fig.suptitle(title, subtitle)
    names = (None if row_names is None else [row_names[i] for i in row_order],
             None if column_names is None else [column_names[i] for i in col_order])
    heat = axes[1][1].heatmap(m[row_order][:, col_order],
                              None if x is None else [x[i] if j % x_step == 0 else '' for j, i in enumerate(col_order)],
                              None if y is None else [y[i] if j % y_step == 0 else '' for j, i in enumerate(row_order)],
                              cmap = cmap, show_all = (show_all or x_all, show_all or y_all), vmin = vmin, vmax = vmax,
                              center = center, names = names)
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
