"""Fonts and text: Century Gothic where installed, sub- and superscripts, spaced capitals and halos."""
import contextlib
import functools
import math
import os
import re
import threading
from io import BytesIO
from glycorender import pdfwrite
from glycorender.pdfwrite import TTFont, pdfmetrics


CAP_HEIGHT = 0.72
_RUN = re.compile(r'[_^]\{[^}]*\}|[^_^]+|[_^]')
# Sub- and superscript characters, which fonts mostly lack, as runs of their own
_SCRIPTS = str.maketrans({**{c: f'_{{{v}}}' for c, v in zip('₀₁₂₃₄₅₆₇₈₉₊₋', '0123456789+-')},
                          **{c: f'^{{{v}}}' for c, v in zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')}})


# The family a figure is laid out and drawn in, per thread, as figures can be drawn on worker threads side by side
_family = threading.local()
# Families whose files this machine lacks, so that each is looked for once
_missing = set()


@contextlib.contextmanager
def family(font):
    """Text measured and drawn in a palette's (name, regular file, bold file) family while inside, or in the default
    family for an empty one."""
    before = getattr(_family, 'font', ())
    _family.font = tuple(font)
    try:
        yield
    finally:
        _family.font = before


def fonts():
    """Regular and bold font names, registered once: the current palette's family where this machine has it, else
    Century Gothic, else glycorender's bundled Comfortaa."""
    font = getattr(_family, 'font', ())
    if font and font[0] not in _missing:
        name, regular, bold = font
        if name + '-Bold' not in pdfmetrics.fonts:
            try:
                pdfmetrics.registerFont(TTFont(name, regular))
                pdfmetrics.registerFont(TTFont(name + '-Bold', bold))
            except FileNotFoundError:
                _missing.add(name)
        if name not in _missing:
            return name, name + '-Bold'
    if 'CenturyGothic-Bold' in pdfmetrics.fonts:
        return 'CenturyGothic', 'CenturyGothic-Bold'
    if 'Comfortaa-Bold' in pdfmetrics.fonts:
        return 'Comfortaa', 'Comfortaa-Bold'
    try:
        pdfmetrics.registerFont(TTFont('CenturyGothic', 'GOTHIC.TTF'))
        pdfmetrics.registerFont(TTFont('CenturyGothic-Bold', 'GOTHICB.TTF'))
        return 'CenturyGothic', 'CenturyGothic-Bold'
    except FileNotFoundError:
        root = os.path.join(os.path.dirname(pdfwrite.__file__), 'fonts')
        pdfmetrics.registerFont(TTFont('Comfortaa', os.path.join(root, 'Comfortaa-Regular.ttf')))
        pdfmetrics.registerFont(TTFont('Comfortaa-Bold', os.path.join(root, 'Comfortaa-Bold.ttf')))
        return 'Comfortaa', 'Comfortaa-Bold'


def runs(text):
    """Split 'log_{2} x' or 'log₂ x' into (text, rise, scale) pieces."""
    out = []
    for piece in _RUN.findall(str(text).translate(_SCRIPTS)):
        if piece[0] in '_^' and len(piece) > 2:
            out.append((piece[2:-1], -0.22 if piece[0] == '_' else 0.38, 0.7))
        else:
            out.append((piece, 0.0, 1.0))
    return out


def spaced(pal, text):
    """A palette's treatment of axis names: capitals, where the palette asks for them."""
    return str(text).upper() if pal.caps else str(text)


def width(text, size, bold = False, tracking = 0.0):
    if '\n' in str(text):
        return max(width(line, size, bold, tracking) for line in str(text).split('\n'))
    name = fonts()[1 if bold else 0]
    return sum(pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t) for t, _, k in runs(text))


def outline(c, ttf, text, x, y, size, tracking = 0.0):
    """A string's glyph outlines as one path, so that it can be stroked into a halo."""
    p, pen, gids = c.beginPath(), x, [ttf.gid(ch) for ch in text]
    k = size / ttf.units_per_em
    for n, g in enumerate(gids):
        cur = (0.0, 0.0)
        for cmd, a in ttf.outline(g):
            if cmd == 'M':
                p.moveTo(a[0] * k + pen, a[1] * k + y)
                cur = (a[0] * k + pen, a[1] * k + y)
            elif cmd == 'L':
                p.lineTo(a[0] * k + pen, a[1] * k + y)
                cur = (a[0] * k + pen, a[1] * k + y)
            elif cmd == 'Q':
                qx, qy, ex, ey = a[0] * k + pen, a[1] * k + y, a[2] * k + pen, a[3] * k + y
                p.curveTo(cur[0] + 2 * (qx - cur[0]) / 3, cur[1] + 2 * (qy - cur[1]) / 3, ex + 2 * (qx - ex) / 3,
                          ey + 2 * (qy - ey) / 3, ex, ey)
                cur = (ex, ey)
            elif cmd == 'Z':
                p.close()
        pen += ttf.width(g) * size / 1000.0 + tracking
        if n + 1 < len(gids):
            pen += ttf.kern(g, gids[n + 1]) * k
    return p


def draw(c, x, y, text, size, color, bold = False, ha = 'left', va = 'baseline', rotation = 0.0, tracking = 0.0,
         halo = 0.0, halo_color = (1.0, 1.0, 1.0), linespacing = 1.25):
    """Draw text at (x, y) in page points; several lines, `linespacing` sizes apart, run downward as one block that
    `va` places."""
    text = str(text)
    if not text:
        return
    if '\n' in text:
        parts = text.split('\n')
        lift = {'center': 0.5, 'bottom': 1.0}.get(va, 0.0) * (len(parts) - 1)
        for i, part in enumerate(parts):
            drop = (i - lift) * size * linespacing
            draw(c, x + drop * math.sin(math.radians(rotation)), y - drop * math.cos(math.radians(rotation)), part,
                 size, color, bold, ha, va, rotation, tracking, halo, halo_color)
        return
    name = fonts()[1 if bold else 0]
    w = width(text, size, bold, tracking)
    dx = {'left': 0.0, 'center': -w / 2, 'right': -w}[ha]
    dy = {'baseline': 0.0, 'center': -CAP_HEIGHT * size / 2, 'top': -CAP_HEIGHT * size, 'bottom': 0.22 * size}[va]
    c.saveState()
    c.translate(x, y)
    if rotation:
        c.rotate(rotation)
    pen = dx
    pieces = runs(text)
    if halo:
        c.saveState()
        c.setStrokeColorRGB(*halo_color, alpha = 1.0)
        c.setLineWidth(halo)
        c.setLineCap(1)
        c.setLineJoin(1)
        q = pen
        for t, rise, k in pieces:
            c.drawPath(outline(c, pdfmetrics.fonts[name].ttf, t, q, dy + rise * size, size * k, tracking), fill = 0,
                       stroke = 1)
            q += pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t)
        c.restoreState()
    c.setFillColorRGB(*color, alpha = 1.0)
    for t, rise, k in pieces:
        c.setFont(name, size * k)
        c.drawString(pen, dy + rise * size, t, charSpace = tracking)
        pen += pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t)
    c.restoreState()


# Points a glyph's median symbol, such as a monosaccharide, takes per point of type, and the least share of that which
# tick labels shrink to, to fit between their neighbors, before every other one is left out
GLYPH_SYMBOL = 1.0
GLYPH_MIN = 0.45
# How far a die-cut layer reaches past the ink, in symbols, as GlycoDraw's own stickers do
GLYPH_CUT = 0.18
# Points below which a glyph's text, such as a glycan's linkages, is left out: under half the smallest type any journal
# prints, it reads as texture, while it would weigh most of the drawing
GLYPH_TEXT_MIN = 2.0


def _glycan(label, vertical):
    """A glycan's SNFG drawing from glycowork, or None for a label that names none. Imported here, as glycowork itself
    imports viscose."""
    from glycowork.motif.draw import GlycoDraw
    try:
        return GlycoDraw(label, vertical = vertical, suppress = True, restrict_vocab = True)
    except Exception:
        return None


# What can stand in for a label, by name: a function of (label, vertical) that returns a drawing as SVG, or an object
# with as_svg(), such as a GlycoDraw drawing, or None for a label it cannot draw, which then stays text
GLYPHS = {'glycan': _glycan}


def register_glyphs(name, maker):
    """Makes `name` usable wherever labels take glyphs, such as tick_params(glyphs = name): `maker(label, vertical)`
    returns the label's drawing, as SVG or an object with as_svg(), or None to keep the label as text. `vertical` asks
    for a drawing that runs up the page, as under a crowded x axis."""
    GLYPHS[name] = maker
    glyph.cache_clear()


def check_glyphs(maker):
    """A glyph maker as given, after checking that it names one or is one."""
    if maker and not callable(maker) and maker not in GLYPHS:
        raise ValueError(f"Glyphs come from a function or one of {', '.join(map(repr, GLYPHS))}, got {maker!r}")
    return maker


def _apply(m, x, y):
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def _mul(m, n):
    """The transform `m` followed by `n`."""
    return (m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2],
            m[2] * n[1] + m[3] * n[3], m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])


def _det(m):
    return math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))


class Glyph:
    """A drawing that stands in for a label: its display-list ops in units of its median symbol around its anchor, the
    middle of its root symbol (the rightmost, or the lowest in a vertical drawing, as a glycan's reducing end), with y
    up; `box` is the ink's (left, bottom, right, top), `cut` the ops of a die-cut layer under it, colored when drawn,
    `fonts` the glyphs of each font its text uses, and `symbols` each symbol's (x, y, radius), where a leader may
    end."""
    __slots__ = ('ops', 'cut', 'box', 'fonts', 'symbols')

    def __init__(self, ops, cut, box, fonts, symbols):
        self.ops, self.cut, self.box, self.fonts, self.symbols = ops, cut, box, fonts, symbols

    def extent(self, scale, cut = False):
        """(left, bottom, right, top) around the anchor at `scale` points per symbol, with the cut's reach if drawn."""
        grow = GLYPH_CUT if cut else 0.0
        return tuple((v + grow * s) * scale for v, s in zip(self.box, (-1, -1, 1, 1)))


@functools.lru_cache(4096)
def glyph(maker, label, vertical = False):
    """The Glyph that `maker`, a name in GLYPHS or a function, draws for `label`, or None where it draws none. Each is
    made once."""
    label = str(label)
    drawing = (GLYPHS[maker] if isinstance(maker, str) else maker)(label, vertical) if label.strip() else None
    if drawing is None:
        return None
    from glycorender import convert_svg_to_pdf
    ops = convert_svg_to_pdf(drawing if isinstance(drawing, str) else drawing.as_svg(), BytesIO(),
                             return_canvas = True).ops
    # Each op's ink box on the drawing's page. Symbols are what is both filled and outlined, and their middles find the
    # root; a text's box is what the cut runs around
    symbols, kept, boxes, cut, boxed, width = [], [], [], [], [], 0.0
    for op in ops:
        m = op['ctm']
        if op['kind'] == 'text':
            w = pdfmetrics.stringWidth(op['text'], op['font'], op['size']) + op['char_space'] * len(op['text'])
            pts = [_apply(m, op['x'] + u, op['y'] + v * op['size'])
                   for u, v in ((0.0, -0.22), (w, -0.22), (w, CAP_HEIGHT), (0.0, CAP_HEIGHT))]
            reach = 0.0
            boxed += [('m', *pts[0]), ('l', *pts[1]), ('l', *pts[2]), ('l', *pts[3]), ('h', )]
        elif op['kind'] == 'path' and (op['fill'] or op['stroke']):
            pts = [_apply(m, seg[i], seg[i + 1]) for seg in op['path'] for i in range(1, len(seg), 2)]
            reach = op['line_width'] * _det(m) / 2 if op['stroke'] else 0.0
            cut += [(seg[0], *[v for i in range(1, len(seg), 2) for v in _apply(m, seg[i], seg[i + 1])])
                    for seg in op['path']]
            width = max(width, 2 * reach)
        else:
            continue
        if not pts:
            continue
        px, py = [p[0] for p in pts], [p[1] for p in pts]
        kept.append(op)
        boxes.append((min(px) - reach, min(py) - reach, max(px) + reach, max(py) + reach))
        if op['kind'] == 'path' and op['fill'] and op['stroke']:
            symbols.append((max(max(px) - min(px), max(py) - min(py)), (min(px) + max(px)) / 2,
                            (min(py) + max(py)) / 2))
    if not kept:
        return None
    xs, ys = [v for b in boxes for v in (b[0], b[2])], [v for b in boxes for v in (b[1], b[3])]
    unit = sorted(s[0] for s in symbols)[len(symbols) // 2] if symbols else max(max(xs) - min(xs), max(ys) - min(ys))
    if not symbols:
        root = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    else:
        # The rightmost symbol, or the lowest of a vertical drawing, on the main chain: a fucose hung beside a reducing
        # end shares its place across the chain, so of those, the root is the one with the most symbols in its line
        along, across = (2, 1) if vertical else (1, 2)
        end = min(s[along] for s in symbols) if vertical else max(s[along] for s in symbols)
        ends = [s for s in symbols if abs(s[along] - end) < 0.1 * unit]
        root = max(ends, key = lambda s: sum(abs(t[across] - s[across]) < 0.1 * unit for t in symbols))[1:]
    to_unit = (1.0 / unit, 0.0, 0.0, 1.0 / unit, -root[0] / unit, -root[1] / unit)
    # Shapes of one paint and transform become one path, each joining the latest such path that nothing drawn since
    # overlaps, so that what lies above stays above: a drawing's bonds, its symbols of each kind, and so on. It writes
    # each paint once, which keeps SVG and PDF light, and the rasterizer covers one path where it covered many
    groups, fonts = [], {}
    for op, b in zip(kept, boxes):
        key = None if op['kind'] != 'path' or op.get('fill_grad') else tuple(
            op[k] if k != 'dash' else op[k] and (tuple(op[k][0]), op[k][1]) for k in
            ('ctm', 'fill', 'stroke', 'fill_rgb', 'fill_alpha', 'stroke_rgb', 'stroke_alpha', 'line_width', 'cap',
             'join', 'dash', 'even_odd'))
        k = next((i for i in range(len(groups) - 1, -1, -1) if groups[i][0] == key), None) if key else None
        if k is not None and not any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
                                     for g in groups[k + 1:] for a in g[2]):
            groups[k][1]['path'] = groups[k][1]['path'] + op['path']
            groups[k][2].append(b)
        else:
            groups.append([key, dict(op), [b]])
        if op['kind'] == 'text':
            fonts.setdefault(op['font'], set()).update(op['ttf'].gid(ch) for ch in op['text'])
    ink = tuple(dict(g[1], ctm = _mul(g[1]['ctm'], to_unit), clip = None) for g in groups)
    # The cut is a path on the drawing's page around every shape and another around the boxes of its text, which goes
    # where the text does, each filled and fattened by a round stroke `GLYPH_CUT` symbols past the widest line, as
    # glycorender cuts its stickers
    cut = tuple(dict(kept[0], kind = 'path', path = path, ctm = to_unit, clip = None, fill = True, stroke = True,
                     line_width = width + 2 * GLYPH_CUT * unit, cap = 1, join = 1, dash = None, fill_grad = None,
                     even_odd = False, glyph_text = bool(i)) for i, path in enumerate((cut, boxed)) if path)
    box = tuple((v - o) / unit for v, o in zip((min(xs), min(ys), max(xs), max(ys)), root * 2))
    marks = tuple(((x - root[0]) / unit, (y - root[1]) / unit, size / 2 / unit) for size, x, y in symbols)
    return Glyph(ink, cut, box, fonts, marks)


def draw_glyph(c, g, x, y, scale, cut = None):
    """A glyph with its anchor at (x, y) in page points and `scale` points per symbol, over a die-cut layer in the color
    `cut`, such as a sticker's white on a dark page, or the page's own color as a halo over marks. Its ops keep the
    drawing's own round numbers under one shared transform, which compress better than page points: a glycan takes
    what GlycoDraw's own PDF of it does."""
    m = _mul((scale, 0.0, 0.0, scale, x, y), c.state['ctm'])
    # Text too small to read on the page is left out, with its share of the cut
    legible = [op for op in g.ops if op['kind'] != 'text' or op['size'] * _det(op['ctm']) * scale >= GLYPH_TEXT_MIN]
    lettered = any(op['kind'] == 'text' for op in legible)
    for op in g.cut if cut is not None else ():
        if lettered or not op['glyph_text']:
            c.ops.append(dict(op, ctm = _mul(op['ctm'], m), clip = c.state['clip'], fill_rgb = tuple(cut),
                              stroke_rgb = tuple(cut), fill_alpha = 1.0, stroke_alpha = 1.0))
    for op in legible:
        c.ops.append(dict(op, ctm = _mul(op['ctm'], m), clip = c.state['clip']))
    # The page embeds the glyphs of every font its text uses
    for name, gids in g.fonts.items() if lettered else ():
        c.used[name] = pdfmetrics.fonts[name]
        c.glyphs.setdefault(name, set()).update(gids)
