"""Fonts and text: Century Gothic where installed, sub- and superscripts, spaced capitals and halos."""
import math
import os
import re
from glycorender import pdfwrite
from glycorender.pdfwrite import TTFont, pdfmetrics


CAP_HEIGHT = 0.72
_RUN = re.compile(r'[_^]\{[^}]*\}|[^_^]+|[_^]')


def fonts():
    """Regular and bold font names, registered once; glycorender's bundled Comfortaa where Century Gothic is missing."""
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
    """Split 'log_{2} x' into (text, rise, scale) pieces."""
    out = []
    for piece in _RUN.findall(str(text)):
        if piece[0] in '_^' and len(piece) > 2:
            out.append((piece[2:-1], -0.22 if piece[0] == '_' else 0.38, 0.7))
        else:
            out.append((piece, 0.0, 1.0))
    return out


def spaced(pal, text):
    """A palette's treatment of axis names: capitals with their tracking, where the palette asks for them."""
    return str(text).upper() if pal.caps else str(text)


def width(text, size, bold = False, tracking = 0.0):
    if '\n' in str(text):
        return max(width(line, size, bold, tracking) for line in str(text).split('\n'))
    name = fonts()[1 if bold else 0]
    return sum(pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t) for t, _, k in runs(text))


def outline(c, ttf, text, x, y, size):
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
        pen += ttf.width(g) * size / 1000.0
        if n + 1 < len(gids):
            pen += ttf.kern(g, gids[n + 1]) * k
    return p


def draw(c, x, y, text, size, color, bold = False, ha = 'left', va = 'baseline', rotation = 0.0, tracking = 0.0,
         halo = 0.0,
         halo_color = (1.0, 1.0, 1.0)):
    """Draw text at (x, y) in page points; several lines run downward from it."""
    text = str(text)
    if not text:
        return
    if '\n' in text:
        for i, part in enumerate(text.split('\n')):
            drop = i * size * 1.25
            draw(c, x + drop * math.sin(math.radians(rotation)), y - drop * math.cos(math.radians(rotation)), part,
                 size, color, bold, ha, va,
                 rotation, tracking, halo, halo_color)
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
    pieces = [(t, rise, k) for t, rise, k in runs(text)]
    if halo:
        c.saveState()
        c.setStrokeColorRGB(*halo_color, alpha = 1.0)
        c.setLineWidth(halo)
        c.setLineCap(1)
        c.setLineJoin(1)
        q = pen
        for t, rise, k in pieces:
            c.drawPath(outline(c, pdfmetrics.fonts[name].ttf, t, q, dy + rise * size, size * k), fill = 0, stroke = 1)
            q += pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t)
        c.restoreState()
    c.setFillColorRGB(*color, alpha = 1.0)
    for t, rise, k in pieces:
        c.setFont(name, size * k)
        c.drawString(pen, dy + rise * size, t, charSpace = tracking)
        pen += pdfmetrics.stringWidth(t, name, size * k) + tracking * len(t)
    c.restoreState()
