"""PowerPoint output: figures as slides of native DrawingML shapes, with text that stays text, the palette as the deck's
theme colors, and every shape named after what it draws, so that Morph carries one figure into the next."""
import colorsys
import math
import struct
import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
import numpy as np
from glycorender import raster
from .figure import SHADOW_DX, SHADOW_DY, UNITS, _unit
from .palettes import _oklch


# English Metric Units per point
EMU = 12700
# Morph pairs shapes of consecutive slides whose names start with this and are otherwise equal
MORPH = '!!'
# A palette's soft shadow as PowerPoint's own, moved by the figure's offset and blurred over SHADOW_BLUR points, which
# PowerPoint renders within 2.5 levels of viscose's own shadow, fitted on Aura's bars; and a gradient laid down in this
# many stops, which PowerPoint blends between in linear light
SHADOW_BLUR, GRADIENT_STEPS = 16.5, 16
# Heatmaps and other pictures are scaled up to at least this resolution on the slide, by repeating pixels, as
# PowerPoint smooths a picture it enlarges and would blur the cells into each other
PICTURE_DPI = 300.0
# A color is a shade of the palette's nearer hue, and so of that hue's accent, within THEME_HUE degrees of it in OKLCH
# and from THEME_CHROMA on; grays and colors of other hues stay as they are
THEME_HUE, THEME_CHROMA = 30.0, 0.008
# Slides are 1 to 56 inches on a side
SLIDE_MIN, SLIDE_MAX = 914400, 51206400
_NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
       'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
_CT = 'application/vnd.openxmlformats-officedocument.presentationml.'
_CAP = {0: 'flat', 1: 'rnd', 2: 'sq'}
_JOIN = {0: '<a:miter lim="1000000"/>', 1: '<a:round/>', 2: '<a:bevel/>'}


def _hex(rgb):
    return '%02X%02X%02X' % tuple(max(0, min(255, int(round(v * 255)))) for v in rgb[:3])


def _family(ttf):
    """A font's family name from its name table, as PowerPoint looks the font up by it."""
    d = ttf.data
    off = ttf.tables['name'][0]
    count, strings = struct.unpack('>HH', d[off + 2:off + 6])
    found = {}
    for i in range(count):
        platform, encoding, lang, nid, length, at = struct.unpack('>6H', d[off + 6 + 12 * i:off + 18 + 12 * i])
        if nid in (1, 16) and platform in (0, 3):
            found.setdefault(nid, d[off + strings + at:off + strings + at + length].decode('utf-16-be', 'replace'))
    return found.get(16) or found.get(1) or 'Century Gothic'


def _ascent(ttf):
    """How far below a text box's top PowerPoint sets the first baseline, per point of type: the font's Windows
    ascent, which its line height starts with."""
    d = ttf.data
    os2 = ttf.tables.get('OS/2')
    if os2 is None:
        return ttf.ascent / ttf.units_per_em
    return struct.unpack('>H', d[os2[0] + 74:os2[0] + 76])[0] / ttf.units_per_em


def _apply(m, x, y):
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


class _Theme:
    """The palette as a deck's theme colors. Its ink, page, muted text, and faint guides are the theme's text and
    background colors, and its two hues the first two accents, in the shades that `Palette.contrast` gives the first
    two series, as an Office chart colors its first two series in them; accents 3 to 6 hold the shades its next series
    take, for what a speaker adds. Every color of either hue, a shade, a gradient's blend, or a heatmap's cell, is its
    accent with its own lightness and its hue and saturation relative to the accent, so that a deck's own theme, or
    PowerPoint's Design tab, recolors the figure in the new hues while its shades keep their lightness, and with it
    their contrast and order."""
    def __init__(self, pal):
        dark = pal.is_dark
        self.pal, self.dark = pal, dark
        named = [('tx1', pal.color('ink')), ('bg1', pal.color('paper')), ('tx2', pal.color('muted')),
                 ('bg2', pal.color('faint'))] + [('accent%d' % (i + 1), pal.color(c))
                                                 for i, c in enumerate(pal.contrast(6))]
        self.slots = {}
        for name, rgb in named[:4]:
            self.slots.setdefault(_hex(rgb), name)
        self.colors = dict(named)
        # A dark page is the theme's first dark color and its ink the first light one, as Office's dark themes map them
        self.scheme = {('dk1', 'lt1', 'dk2', 'lt2')[i]: self.colors[k] for i, k in
                       enumerate(('bg1', 'tx1', 'bg2', 'tx2') if dark else ('tx1', 'bg1', 'tx2', 'bg2'))}
        # Each hue's place on the color wheel, and the accent that stands for it
        self.hues = [(math.degrees(_oklch(self.colors[k])[2]), k, colorsys.rgb_to_hls(*self.colors[k]))
                     for k in ('accent1', 'accent2')]

    def color(self, rgb, alpha = 1.0, fixed = False):
        """A color as DrawingML: a theme color, or with `fixed`, such as a glycan's SNFG colors, as it is."""
        h = _hex(rgb)
        a = '' if alpha >= 1.0 else '<a:alpha val="%d"/>' % round(max(alpha, 0.0) * 100000)
        if fixed:
            return '<a:srgbClr val="%s">%s</a:srgbClr>' % (h, a)
        if h in self.slots:
            return '<a:schemeClr val="%s">%s</a:schemeClr>' % (self.slots[h], a)
        _, C, hue = _oklch(rgb)
        hue = math.degrees(hue)
        near = min(self.hues, key = lambda v: abs((hue - v[0] + 180.0) % 360.0 - 180.0))
        if C < THEME_CHROMA or abs((hue - near[0] + 180.0) % 360.0 - 180.0) > THEME_HUE:
            return '<a:srgbClr val="%s">%s</a:srgbClr>' % (h, a)
        # PowerPoint turns and scales the accent's hue and saturation in sRGB's HSL, as colorsys does, and then sets the
        # lightness: scaled to almost nothing, which leaves hue and saturation in place, as it keeps floats between
        # steps where a scale of 0 would make black, and then raised to the color's own
        hh, ll, ss = colorsys.rgb_to_hls(*(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))
        bh, bl, bs = near[2]
        turn = ((hh - bh + 0.5) % 1.0 - 0.5) * 360.0
        return ('<a:schemeClr val="%s"><a:hueOff val="%d"/><a:satMod val="%d"/><a:lumMod val="1"/>'
                '<a:lumOff val="%d"/>%s</a:schemeClr>' % (near[1], round(turn * 60000), round(ss / bs * 100000) if bs
                                                          else 0, round((ll - bl * 1e-5) * 100000), a))

    def clr_map(self):
        pairs = (('bg1', 'dk1'), ('tx1', 'lt1'), ('bg2', 'dk2'), ('tx2', 'lt2')) if self.dark else (
            ('bg1', 'lt1'), ('tx1', 'dk1'), ('bg2', 'lt2'), ('tx2', 'dk2'))
        return ('<p:clrMap %s accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
                'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
                % ' '.join('%s="%s"' % p for p in pairs))

    def xml(self, font):
        scheme = ''.join('<a:%s><a:srgbClr val="%s"/></a:%s>' % (k, _hex(v), k) for k, v in self.scheme.items())
        scheme += ''.join('<a:accent%d><a:srgbClr val="%s"/></a:accent%d>' % (i, _hex(self.colors['accent%d' % i]), i)
                          for i in range(1, 7))
        pal = self.pal
        scheme += '<a:hlink><a:srgbClr val="%s"/></a:hlink><a:folHlink><a:srgbClr val="%s"/></a:folHlink>' % (
            _hex(pal.color('a0')), _hex(pal.color('b0')))
        face = '<a:latin typeface=%s/><a:ea typeface=""/><a:cs typeface=""/>' % quoteattr(font)
        fill = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        line = '<a:ln w="%d"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name=%s><a:themeElements>'
                '<a:clrScheme name=%s>%s</a:clrScheme>'
                '<a:fontScheme name=%s><a:majorFont>%s</a:majorFont><a:minorFont>%s</a:minorFont></a:fontScheme>'
                '<a:fmtScheme name="viscose"><a:fillStyleLst>%s</a:fillStyleLst><a:lnStyleLst>%s</a:lnStyleLst>'
                '<a:effectStyleLst>%s</a:effectStyleLst><a:bgFillStyleLst>%s</a:bgFillStyleLst></a:fmtScheme>'
                '</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'
                % (quoteattr(pal.name), quoteattr(pal.name), scheme, quoteattr(font), face, face, fill * 3,
                   ''.join(line % w for w in (6350, 12700, 19050)), '<a:effectStyle><a:effectLst/></a:effectStyle>' * 3,
                   fill * 3))


def _flatten(segs):
    """Subpaths as lists of points, with each cubic as 16 lines, and whether each is closed."""
    out, pts, closed = [], [], False
    for s in segs:
        if s[0] == 'm':
            if pts:
                out.append((pts, closed))
            pts, closed = [(s[1], s[2])], False
        elif s[0] == 'l':
            pts.append((s[1], s[2]))
        elif s[0] == 'c':
            (x0, y0), t = pts[-1], np.linspace(0.0, 1.0, 17)[1:]
            for u in t:
                v = 1.0 - u
                pts.append((v ** 3 * x0 + 3 * v * v * u * s[1] + 3 * v * u * u * s[3] + u ** 3 * s[5],
                            v ** 3 * y0 + 3 * v * v * u * s[2] + 3 * v * u * u * s[4] + u ** 3 * s[6]))
        elif s[0] == 'h':
            closed = True
    if pts:
        out.append((pts, closed))
    return out


def _clip_polygon(pts, box):
    """Sutherland and Hodgman's clip of a closed outline to (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    for inside, cut in ((lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))),
                        (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1]) * (x1 - p[0]) / (q[0] - p[0]))),
                        (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)),
                        (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0]) * (y1 - p[1]) / (q[1] - p[1]), y1))):
        out = []
        for i, q in enumerate(pts):
            p = pts[i - 1]
            if inside(q):
                if not inside(p):
                    out.append(cut(p, q))
                out.append(q)
            elif inside(p):
                out.append(cut(p, q))
        pts = out
        if not pts:
            return []
    return [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]


def _clip_line(pts, box):
    """The runs of an open line that lie within (x0, y0, x1, y1), by Liang and Barsky's clip of each segment."""
    x0, y0, x1, y1 = box
    runs, run = [], []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy, lo, hi = bx - ax, by - ay, 0.0, 1.0
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0:
                    lo, hi = 1.0, 0.0
            elif p < 0:
                lo = max(lo, q / p)
            else:
                hi = min(hi, q / p)
        if lo > hi:
            if run:
                runs.append(run)
            run = []
            continue
        a, b = (ax + lo * dx, ay + lo * dy), (ax + hi * dx, ay + hi * dy)
        if not run or run[-1] != a:
            if run:
                runs.append(run)
            run = [a]
        run.append(b)
        if hi < 1.0:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return [r for r in runs if len(r) > 1]


def _clip_box(op):
    """The page rectangle all of an op's clips leave, or None."""
    box = None
    for (x, y, w, h), m in op.get('clip') or ():
        xs, ys = zip(*(_apply(m, u, v) for u, v in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))))
        b = (min(xs), min(ys), max(xs), max(ys))
        box = b if box is None else (max(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), min(box[3], b[3]))
    return box


def _sample(stops, t):
    """A gradient's RGBA at t along it, as a PDF shading extends its ends."""
    offs = [o for o, _ in stops]
    return tuple(float(np.interp(t, offs, [c[k] for _, c in stops])) for k in range(4))


class _Slide:
    """One figure's display list as a slide's shapes, on a page `scale` times the figure's size, `shift` EMU from the
    slide's top left."""
    def __init__(self, fig, ops, theme, scale, shift):
        self.fig, self.theme, self.k, self.shift = fig, theme, scale * EMU, shift
        self.height, self.shapes, self.media, self.names, self.ids = fig.height, [], [], {}, 1
        self.shadow = fig.pal.shadow
        self.counts, self.lines = {}, {}
        self._name_all(ops)
        for op in ops:
            # A path that belongs to a line of text is its halo, which the text carries as a glow, so that it follows
            # the text when edited
            if op['kind'] == 'path' and not op.get('line') and (op['fill'] or op['stroke']):
                self._path(op)
            elif op['kind'] == 'image':
                self._picture(op)
            elif op['kind'] == 'text':
                key = op.get('line', (id(op), 'left', None, None))[0]
                if key not in self.lines:
                    self.lines[key] = []
                    self.shapes.append(key)
                self.lines[key].append(op)
        self.shapes = [self._text(self.lines[s]) if not isinstance(s, str) else s for s in self.shapes]

    # --- names ---
    def _name_all(self, ops):
        """Every op's shape name, unique on the slide: what it draws within its axes, as the artist and part, the text,
        or the axis piece that drew it, so that the same thing in the next figure has the same name."""
        seen, groups = {}, {}
        for op in ops:
            scope = op.get('scope', ('page', 'page'))
            if op['kind'] == 'text' and op.get('glyph') is None:
                tag = op.get('line')
                base = '%s %s' % (scope[0], op.get('name') or 'text ' + (tag[3] if tag else op['text']))
                key = ('line', tag[0]) if tag else ('text', id(op))
            elif op.get('line'):
                continue
            elif op.get('artist') is not None:
                part = op.get('part')
                base = '%s %s' % (scope[0], op['artist']) + ('' if part is None else ' %s' % part)
                key = ('artist', base)
            elif op.get('glyph') is not None:
                base, key = '%s glyph %s' % (scope[0], op['glyph']), ('glyph', scope[0], op['glyph'])
            elif op.get('name') is not None:
                base, key = '%s %s' % (scope[0], op['name']), ('name', scope[0], op['name'])
            else:
                n = self.counts.get(scope, 0)
                self.counts[scope] = n + 1
                base, key = '%s %s piece %d' % (scope[0], scope[1], n), None
            if key is not None and key[0] == 'line':
                # A line's runs make one text box, named once
                if key in groups:
                    continue
                groups[key] = 1
            elif key is not None:
                # The layers of one thing, such as a point's rim, body, and highlight, are numbered from the bottom up
                k = groups.get(key, 0)
                groups[key] = k + 1
                op['_first'] = k == 0
                base += ' #%d' % k if k else ''
            n = seen.get(base, 0)
            seen[base] = n + 1
            op['_name'] = MORPH + base + (' (%d)' % n if n else '')

    def _id(self):
        self.ids += 1
        return self.ids

    # --- geometry ---
    def _xy(self, x, y):
        """A page point in slide EMU, y running down."""
        return x * self.k + self.shift[0], (self.height - y) * self.k + self.shift[1]

    def _effects(self, op, rgb):
        """The palette's soft shadow in the mark's own color, on the lowest opaque layer of each mark, as PowerPoint
        draws it with the shape, so that it grows and moves with it."""
        # The shadow is how far the mark darkens white, as in viscose's own: a color at the darkest channel's share of
        # the strength, which over white darkens as the mark does and over a dark page glows as faintly
        dark = 1.0 - np.asarray(rgb, dtype = float)
        m = float(dark.max())
        if not (self.shadow and op.get('mark') and op.get('_first', True)) or m < 1e-3:
            return ''
        dist = math.hypot(SHADOW_DX, SHADOW_DY) * self.k
        return ('<a:effectLst><a:outerShdw blurRad="%d" dist="%d" dir="%d" algn="ctr" rotWithShape="0">%s'
                '</a:outerShdw></a:effectLst>' % (round(SHADOW_BLUR * self.k), round(dist),
                                                  round(math.degrees(math.atan2(SHADOW_DY, SHADOW_DX)) * 60000),
                                                  self.theme.color(1.0 - dark / m, self.shadow * m)))

    def _path(self, op):
        m = op['ctm']
        segs = [(s[0],) + tuple(v for i in range(1, len(s), 2) for v in _apply(m, s[i], s[i + 1])) for s in op['path']]
        pts = [v for s in segs for v in zip(s[1::2], s[2::2])]
        if not pts:
            return
        box = _clip_box(op)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if box is not None and not (min(xs) >= box[0] and max(xs) <= box[2] and min(ys) >= box[1] and
                                    max(ys) <= box[3]):
            if min(xs) > box[2] or max(xs) < box[0] or min(ys) > box[3] or max(ys) < box[1]:
                return
            subs = []
            for poly, closed in _flatten(segs):
                if op['fill'] or closed:
                    found = _clip_polygon(poly, box)
                    if len(found) > 2:
                        subs.append((found, True))
                else:
                    subs += [(run, False) for run in _clip_line(poly, box)]
            if not subs:
                return
            segs = [s for poly, closed in subs for s in [('m',) + poly[0]] + [('l',) + p for p in poly[1:]] +
                    ([('h',)] if closed else [])]
        # In slide EMU from here on
        segs = [(s[0],) + tuple(v for i in range(1, len(s), 2) for v in self._xy(s[i], s[i + 1])) for s in segs]
        pts = [v for s in segs for v in zip(s[1::2], s[2::2])]
        x0, y0 = math.floor(min(p[0] for p in pts)), math.floor(min(p[1] for p in pts))
        cx, cy = max(1, math.ceil(max(p[0] for p in pts)) - x0), max(1, math.ceil(max(p[1] for p in pts)) - y0)
        geometry = self._geometry(segs, x0, y0, cx, cy, op)
        # A glycan's symbols keep SNFG's colors in any theme
        fixed = 'glyph' in op
        rgb = op['fill_rgb'] if op['fill'] else op['stroke_rgb']
        if op['fill'] and op.get('fill_grad'):
            fill, rgb = self._gradient(op, x0, y0, cx, cy)
        elif op['fill']:
            fill = '<a:solidFill>%s</a:solidFill>' % self.theme.color(op['fill_rgb'], op['fill_alpha'], fixed)
        else:
            fill = '<a:noFill/>'
        if op['stroke']:
            w = op['line_width'] * math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) * self.k
            dash = ''
            if op['dash'] and op['dash'][0] and op['line_width'] > 0:
                d = list(op['dash'][0]) * (2 - len(op['dash'][0]) % 2)
                # A PDF's round or square caps lengthen each dash by half the line's width at either end, which
                # PowerPoint keeps within the dash, so the dash takes them and the gap gives them up; no gap left, and
                # the line is solid, as it looks
                if op['cap']:
                    d = [v + op['line_width'] * (1 if i % 2 == 0 else -1) for i, v in enumerate(d)]
                    d = d if min(d[1::2]) > 0 else []
                dash = '<a:custDash>%s</a:custDash>' % ''.join(
                    '<a:ds d="%d" sp="%d"/>' % (round(d[i] / op['line_width'] * 100000),
                                                round(d[i + 1] / op['line_width'] * 100000))
                    for i in range(0, len(d), 2)) if d else ''
            line = '<a:ln w="%d" cap="%s"><a:solidFill>%s</a:solidFill>%s%s</a:ln>' % (
                round(w), _CAP[op['cap']], self.theme.color(op['stroke_rgb'], op['stroke_alpha'], fixed), dash,
                _JOIN[op['join']])
        else:
            line = '<a:ln><a:noFill/></a:ln>'
        self.shapes.append(
            '<p:sp><p:nvSpPr><p:cNvPr id="%d" name=%s/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
            '<a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>%s%s%s%s</p:spPr></p:sp>'
            % (self._id(), quoteattr(op['_name']), x0, y0, cx, cy, geometry, fill, line, self._effects(op, rgb)))

    def _geometry(self, segs, x0, y0, cx, cy, op):
        """A rectangle or an ellipse as PowerPoint's own shape, which Morph moves and resizes, and anything else as a
        freeform in the shape's own EMU."""
        kinds = ''.join(s[0] for s in segs)
        corners = [s[1:3] for s in segs if s[0] in 'ml']
        if kinds in ('mlllh', 'mllllh') and all(
                abs(p[0] - q[0]) < 2 or abs(p[1] - q[1]) < 2 for p, q in zip(corners, corners[1:] + corners[:1])):
            return '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        if kinds == 'mcccch':
            ends = [s[1:3] for s in segs[:1]] + [s[5:7] for s in segs[1:5]]
            mid = [(x0, y0 + cy / 2), (x0 + cx / 2, y0), (x0 + cx, y0 + cy / 2), (x0 + cx / 2, y0 + cy)]
            if all(min(math.hypot(e[0] - p[0], e[1] - p[1]) for p in mid) < 2 + 1e-3 * (cx + cy) for e in ends):
                return '<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>'
        at = lambda x, y: '<a:pt x="%d" y="%d"/>' % (round(x - x0), round(y - y0))
        paths, body = [], []
        for s in segs:
            if s[0] == 'm':
                if body and not op.get('even_odd'):
                    paths.append(body)
                    body = []
                body.append('<a:moveTo>%s</a:moveTo>' % at(*s[1:3]))
            elif s[0] == 'l':
                body.append('<a:lnTo>%s</a:lnTo>' % at(*s[1:3]))
            elif s[0] == 'c':
                body.append('<a:cubicBezTo>%s%s%s</a:cubicBezTo>' % (at(*s[1:3]), at(*s[3:5]), at(*s[5:7])))
            else:
                body.append('<a:close/>')
        if body:
            paths.append(body)
        # A path only stroked is open, and PowerPoint would otherwise fill it to its start
        attrs = ' w="%d" h="%d"%s' % (cx, cy, '' if op['fill'] else ' fill="none"')
        return ('<a:custGeom><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/><a:rect l="0" t="0" r="r" b="b"/>'
                '<a:pathLst>%s</a:pathLst></a:custGeom>' % ''.join('<a:path%s>%s</a:path>' % (attrs, ''.join(b))
                                                                   for b in paths))

    def _gradient(self, op, x0, y0, cx, cy):
        """A linear gradient over the shape's own box, as DrawingML spans one: the stops that lie within the box, and
        the gradient's colors where the box begins and ends along it. Also the color its middle has, for a shadow."""
        kind, geo, stops = op['fill_grad']
        m = op['ctm']
        (ax, ay), (bx, by) = self._xy(*_apply(m, geo[0], geo[1])), self._xy(*_apply(m, geo[2], geo[3]))
        dx, dy = bx - ax, by - ay
        n = dx * dx + dy * dy or 1.0
        ts = [((px - ax) * dx + (py - ay) * dy) / n for px in (x0, x0 + cx) for py in (y0, y0 + cy)]
        t0, t1 = min(ts), max(ts)
        # PowerPoint blends a gradient's stops in linear light, which lightens a blend of a dark and a pale shade in
        # its middle, so the blend that viscose draws in sRGB is laid down in GRADIENT_STEPS stops
        span = [(i / GRADIENT_STEPS, _sample(stops, t0 + (t1 - t0) * i / GRADIENT_STEPS))
                for i in range(GRADIENT_STEPS + 1)]
        alpha = op['fill_alpha']
        stops_xml = ''.join('<a:gs pos="%d">%s</a:gs>' % (round(o * 100000), self.theme.color(c[:3], c[3] * alpha))
                            for o, c in span)
        mid = _sample(stops, (t0 + t1) / 2)
        return ('<a:gradFill rotWithShape="1"><a:gsLst>%s</a:gsLst><a:lin ang="%d" scaled="0"/></a:gradFill>'
                % (stops_xml, round(math.degrees(math.atan2(dy, dx)) % 360 * 60000)), mid[:3])

    def _picture(self, op):
        """An image as a picture, its pixels repeated up to PICTURE_DPI on the slide, cropped to its clip."""
        a, b, c, d, e, f = op['ctm']
        if b or c or not a or not d:
            return
        x0, x1, y0, y1 = sorted((e, e + a)) + sorted((f, f + d))
        box = _clip_box(op)
        crop = [0.0, 0.0, 0.0, 0.0]
        if box is not None:
            if x0 >= box[2] or x1 <= box[0] or y0 >= box[3] or y1 <= box[1]:
                return
            w, h = x1 - x0, y1 - y0
            # Shares of the picture cut off at its left, top, right, and bottom
            crop = [max(0.0, box[0] - x0) / w, max(0.0, y1 - box[3]) / h, max(0.0, x1 - box[2]) / w,
                    max(0.0, box[1] - y0) / h]
        px = np.frombuffer(op['rgb'], np.uint8).reshape(op['h'], op['w'], 3)
        if op.get('alpha') is not None:
            px = np.dstack([px, np.frombuffer(op['alpha'], np.uint8).reshape(op['h'], op['w'], 1)])
        px = px[:, ::-1] if a < 0 else px
        px = px[::-1] if d < 0 else px
        rep = max(1, math.ceil((x1 - x0) * self.k / EMU / 72.0 * PICTURE_DPI / op['w']),
                  math.ceil((y1 - y0) * self.k / EMU / 72.0 * PICTURE_DPI / op['h']))
        rep = min(rep, max(1, int(math.sqrt(16e6 / (op['w'] * op['h'])))))
        if rep > 1:
            px = np.repeat(np.repeat(px, rep, axis = 0), rep, axis = 1)
        self.media.append(raster.encode_png(np.ascontiguousarray(px)))
        (ox, oy), (ex, ey) = self._xy(x0, y1), self._xy(x1, y0)
        fade = '<a:alphaModFix amt="%d"/>' % round(op['fill_alpha'] * 100000) if op['fill_alpha'] < 1 else ''
        src = '<a:srcRect l="%d" t="%d" r="%d" b="%d"/>' % tuple(round(v * 100000) for v in crop) if any(crop) else ''
        self.shapes.append(
            '<p:pic><p:nvPicPr><p:cNvPr id="%d" name=%s/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
            '<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rIdm%d">%s</a:blip>%s<a:stretch><a:fillRect/>'
            '</a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
            % (self._id(), quoteattr(op['_name']), len(self.media), fade, src,
               round(ox + crop[0] * (ex - ox)), round(oy + crop[1] * (ey - oy)),
               max(1, round((ex - ox) * (1 - crop[0] - crop[2]))), max(1, round((ey - oy) * (1 - crop[1] - crop[3])))))

    # --- text ---
    def _text(self, ops):
        """A line of text, its runs of sub- and superscripts and all, as a text box that PowerPoint sets in the same
        font and size from the same baseline, so that it reads as drawn and stays editable."""
        first = ops[0]
        tag = first.get('line') or (None, 'left', first['y'], first['text'], 0.0, None)
        _, ha, base, s, halo, halo_rgb = tag
        m = first['ctm']
        ttf = first['ttf']
        main = max(ops, key = lambda o: o['size'])
        size = main['size']
        start, end, runs = math.inf, -math.inf, []
        for op in ops:
            t, k = op['ttf'], op['size'] / op['ttf'].units_per_em
            gids = [t.gid(ch) for ch in op['text']]
            w = sum(t.width(g) * op['size'] / 1000.0 + op['char_space'] for g in gids) + sum(
                t.kern(g, h) * k for g, h in zip(gids, gids[1:]))
            start, end = min(start, op['x']), max(end, op['x'] + w)
            runs.append((op, (op['y'] - base) / size if op['size'] < size else 0.0))
        scale = math.hypot(m[0], m[1])
        rise = _ascent(ttf)
        # The box in the line's own frame, from the first baseline up by PowerPoint's ascent, as wide as the line
        left, right = start, end
        top, bottom = base + rise * size, base + rise * size - 1.2 * size
        cx, cy = _apply(m, (left + right) / 2, (top + bottom) / 2)
        px, py = self._xy(cx, cy)
        w, h = (right - left) * scale * self.k, (top - bottom) * scale * self.k
        angle = -math.degrees(math.atan2(m[1], m[0])) % 360
        rot = ' rot="%d"' % round(angle * 60000) if round(angle * 60000) % 21600000 else ''
        glow = ''
        if halo:
            glow = '<a:effectLst><a:glow rad="%d">%s</a:glow></a:effectLst>' % (
                round(halo / 2 * scale * self.k), self.theme.color(halo_rgb))
        body = []
        for op, lift in runs:
            bold = bool(op['ttf'].mac_style & 1)
            sz = max(100, round((size if lift else op['size']) * scale * self.k / EMU * 100))
            attrs = ' sz="%d"%s%s' % (sz, ' b="1"' if bold else '', ' spc="%d"' % round(op['char_space'] * scale *
                                                                                        self.k / EMU * 100)
                                      if op['char_space'] else '')
            attrs += ' baseline="%d"' % round(lift * 100000) if lift else ''
            body.append('<a:r><a:rPr lang="en-US"%s kern="0" dirty="0"><a:solidFill>%s</a:solidFill>%s'
                        '<a:latin typeface=%s/><a:cs typeface=%s/></a:rPr><a:t>%s</a:t></a:r>'
                        % (attrs, self.theme.color(op['fill_rgb'], op['fill_alpha'], 'glyph' in op), glow,
                           quoteattr(_family(op['ttf'])), quoteattr(_family(op['ttf'])), escape(op['text'])))
        return ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name=%s/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
                '<a:xfrm%s><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
                '</a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" '
                'bIns="0" anchor="t" rtlCol="0"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr algn="%s">'
                '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcBef><a:spcPts val="0"/></a:spcBef></a:pPr>%s</a:p>'
                '</p:txBody></p:sp>'
                % (self._id(), quoteattr(first['_name']), rot, round(px - w / 2), round(py - h / 2), round(w),
                   round(h), {'left': 'l', 'right': 'r', 'center': 'ctr'}[ha], ''.join(body)))

    def xml(self, morph):
        paper = self.fig.pal.color('paper')
        transition = ''
        if morph:
            ms = round(morph * 1000)
            transition = (
                '<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
                '<mc:Choice xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" Requires="p159">'
                '<p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" spd="slow" '
                'p14:dur="%d"><p159:morph option="byObject"/></p:transition></mc:Choice><mc:Fallback>'
                '<p:transition spd="slow"><p:fade/></p:transition></mc:Fallback></mc:AlternateContent>' % ms)
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sld %s><p:cSld><p:bg><p:bgPr>'
                '<a:solidFill>%s</a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr>'
                '<p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
                '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>%s'
                '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>%s</p:sld>'
                % (_NS, self.theme.color(paper), ''.join(self.shapes), transition))


def deck(figures, path = None, size = None, unit = 'in', morph = 1.5):
    """A PowerPoint deck with a slide per figure, as native shapes: text stays text in the palette's font, colors are
    the deck's theme colors (see `Figure.to_pptx`), and every shape is named after what it draws, so that Morph, which
    pairs shapes of the same name on consecutive slides, carries each figure into the next in `morph` seconds: bars
    grow, points move to their new places, and ticks slide along their axes. Draw the figures with the same calls in
    the same order, on new data, at new limits, or in another palette, and what they share morphs while the rest fades.
    Slides are `size` (width, height) in inches, or with unit = 'mm' in millimeters, by default as large as the largest
    figure, and each figure sits centered on its slide, scaled down only where it would not fit. Returns the file's
    bytes, or writes them to `path` and returns that."""
    figures = list(figures)
    if not figures:
        raise ValueError('A deck needs at least one figure')
    pages = [fig._shapes() for fig in figures]
    if size is None:
        width, height = max(f.width for f in figures), max(f.height for f in figures)
    else:
        k = UNITS[_unit(unit, ('in', 'mm'))]
        width, height = size[0] * k, size[1] * k
    sw, sh = [min(SLIDE_MAX, max(SLIDE_MIN, round(v * EMU))) for v in (width, height)]
    theme = _Theme(figures[0].pal)
    font = figures[0].pal.font[0] if figures[0].pal.font else None
    slides = []
    for i, (fig, ops) in enumerate(zip(figures, pages)):
        if font is None:
            font = next((_family(op['ttf']) for op in ops if op['kind'] == 'text'), 'Century Gothic')
        scale = min(1.0, sw / (fig.width * EMU), sh / (fig.height * EMU))
        shift = ((sw - fig.width * scale * EMU) / 2, (sh - fig.height * scale * EMU) / 2)
        # One theme for the deck, the first figure's palette: another palette's colors stay as they are
        slide = _Slide(fig, ops, theme, scale, shift)
        slides.append(slide)
    out = BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        files = {'ppt/theme/theme1.xml': theme.xml(font)}
        types = ['<Override PartName="/ppt/presentation.xml" ContentType="%spresentation.main+xml"/>' % _CT,
                 '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="%sslideMaster+xml"/>' % _CT,
                 '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="%sslideLayout+xml"/>' % _CT,
                 '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-'
                 'officedocument.theme+xml"/>',
                 '<Override PartName="/ppt/presProps.xml" ContentType="%spresProps+xml"/>' % _CT,
                 '<Override PartName="/ppt/viewProps.xml" ContentType="%sviewProps+xml"/>' % _CT,
                 '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-'
                 'officedocument.extended-properties+xml"/>']
        rels = ['<Relationship Id="rIdm" Type="%sslideMaster" Target="slideMasters/slideMaster1.xml"/>' % _REL,
                '<Relationship Id="rIdt" Type="%stheme" Target="theme/theme1.xml"/>' % _REL,
                '<Relationship Id="rIdp" Type="%spresProps" Target="presProps.xml"/>' % _REL,
                '<Relationship Id="rIdv" Type="%sviewProps" Target="viewProps.xml"/>' % _REL]
        ids = []
        image = 0
        for i, slide in enumerate(slides):
            n = i + 1
            files['ppt/slides/slide%d.xml' % n] = slide.xml(morph if i else 0)
            links = ['<Relationship Id="rIdl" Type="%sslideLayout" Target="../slideLayouts/slideLayout1.xml"/>' % _REL]
            for j, png in enumerate(slide.media):
                image += 1
                files['ppt/media/image%d.png' % image] = png
                links.append('<Relationship Id="rIdm%d" Type="%simage" Target="../media/image%d.png"/>'
                             % (j + 1, _REL, image))
            files['ppt/slides/_rels/slide%d.xml.rels' % n] = _rels(links)
            types.append('<Override PartName="/ppt/slides/slide%d.xml" ContentType="%sslide+xml"/>' % (n, _CT))
            rels.append('<Relationship Id="rIds%d" Type="%sslide" Target="slides/slide%d.xml"/>' % (n, _REL, n))
            ids.append('<p:sldId id="%d" r:id="rIds%d"/>' % (255 + n, n))
        files['[Content_Types].xml'] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" '
            'ContentType="image/png"/>%s</Types>' % ''.join(types))
        files['_rels/.rels'] = _rels([
            '<Relationship Id="rId1" Type="%sofficeDocument" Target="ppt/presentation.xml"/>' % _REL,
            '<Relationship Id="rId2" Type="%sextended-properties" Target="docProps/app.xml"/>' % _REL])
        files['docProps/app.xml'] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns="http://schemas.openxmlformats.'
            'org/officeDocument/2006/extended-properties"><Application>viscose</Application><Slides>%d</Slides>'
            '</Properties>' % len(slides))
        files['ppt/presentation.xml'] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:presentation %s saveSubsetFonts="1">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rIdm"/></p:sldMasterIdLst><p:sldIdLst>%s'
            '</p:sldIdLst><p:sldSz cx="%d" cy="%d"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
            % (_NS, ''.join(ids), sw, sh))
        files['ppt/_rels/presentation.xml.rels'] = _rels(rels)
        files['ppt/presProps.xml'] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                                      '<p:presentationPr %s/>' % _NS)
        files['ppt/viewProps.xml'] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:viewPr %s>'
                                      '<p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/>'
                                      '</p:normalViewPr><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>' % _NS)
        empty = ('<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                 '<p:grpSpPr/></p:spTree>')
        styles = ''.join('<p:%s><a:lvl1pPr><a:defRPr><a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr></p:%s>'
                         % (k, k) for k in ('titleStyle', 'bodyStyle', 'otherStyle'))
        files['ppt/slideMasters/slideMaster1.xml'] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sldMaster %s><p:cSld><p:bg><p:bgRef '
            'idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>%s</p:cSld>%s<p:sldLayoutIdLst><p:sldLayoutId '
            'id="2147483649" r:id="rIdl"/></p:sldLayoutIdLst><p:txStyles>%s</p:txStyles></p:sldMaster>'
            % (_NS, empty, theme.clr_map(), styles))
        files['ppt/slideMasters/_rels/slideMaster1.xml.rels'] = _rels([
            '<Relationship Id="rIdl" Type="%sslideLayout" Target="../slideLayouts/slideLayout1.xml"/>' % _REL,
            '<Relationship Id="rIdt" Type="%stheme" Target="../theme/theme1.xml"/>' % _REL])
        files['ppt/slideLayouts/slideLayout1.xml'] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sldLayout %s type="blank" preserve="1">'
            '<p:cSld name="Figure">%s</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
            % (_NS, empty))
        files['ppt/slideLayouts/_rels/slideLayout1.xml.rels'] = _rels([
            '<Relationship Id="rIdm" Type="%sslideMaster" Target="../slideMasters/slideMaster1.xml"/>' % _REL])
        for name in ['[Content_Types].xml', '_rels/.rels'] + sorted(set(files) - {'[Content_Types].xml',
                                                                                   '_rels/.rels'}):
            data = files[name]
            # Pictures are compressed already
            z.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data if isinstance(data, bytes) else
                       data.encode('utf-8'), zipfile.ZIP_STORED if name.endswith('.png') else zipfile.ZIP_DEFLATED)
    data = out.getvalue()
    if path is None:
        return data
    path = Path(path)
    path.write_bytes(data)
    return path


def _rels(items):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships">%s</Relationships>' % ''.join(items))
