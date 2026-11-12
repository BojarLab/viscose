"""Palettes: a few colors in shades, plus the special effects that belong to each palette, and the color science that
keeps them legible on screen and true in print."""
import itertools
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np


NAMED = {'white': '#FFFFFF', 'black': '#1C1917', 'grey': '#B8B8BE', 'gray': '#B8B8BE', 'none': None}
# CSS names, since a color set by hand, such as a network node's, may use them
CSS = {'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4',
       'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'blanchedalmond': '#FFEBCD', 'blue': '#0000FF',
       'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0',
       'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED',
       'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B',
       'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkgrey': '#A9A9A9',
       'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00',
       'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F',
       'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F', 'darkslategrey': '#2F4F4F',
       'darkturquoise': '#00CED1', 'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF',
       'dimgray': '#696969', 'dimgrey': '#696969', 'dodgerblue': '#1E90FF', 'firebrick': '#B22222',
       'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC',
       'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'green': '#008000',
       'greenyellow': '#ADFF2F', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C',
       'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA',
       'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6',
       'lightcoral': '#F08080', 'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgray': '#D3D3D3',
       'lightgreen': '#90EE90', 'lightgrey': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A',
       'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA', 'lightslategray': '#778899',
       'lightslategrey': '#778899', 'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0', 'lime': '#00FF00',
       'limegreen': '#32CD32', 'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000',
       'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB',
       'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A',
       'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585', 'midnightblue': '#191970', 'mintcream': '#F5FFFA',
       'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'navy': '#000080',
       'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23', 'orange': '#FFA500', 'orangered': '#FF4500',
       'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE',
       'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9', 'peru': '#CD853F',
       'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080',
       'rebeccapurple': '#663399', 'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1',
       'saddlebrown': '#8B4513', 'salmon': '#FA8072', 'sandybrown': '#F4A460', 'seagreen': '#2E8B57',
       'seashell': '#FFF5EE', 'sienna': '#A0522D', 'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD',
       'slategray': '#708090', 'slategrey': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F',
       'steelblue': '#4682B4', 'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347',
       'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00',
       'yellowgreen': '#9ACD32'}


def _hex(c):
    # The short form only counts with its #, so that a three-letter typo stays an unknown color
    c = c[1] * 2 + c[2] * 2 + c[3] * 2 if len(c) == 4 and c[0] == '#' else c.lstrip('#')
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mix(c, f, to = (1.0, 1.0, 1.0)):
    """A color moved towards another; f = 1 keeps it, f = 0 is the other."""
    return tuple(t + (v - t) * f for v, t in zip(c, to))


# OKLab (Ottosson 2020): linear sRGB to cone responses, and their cube roots to lightness and two opponent axes
_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188501, 0.6299787005]])
_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])
# Lightness a dark page's colors span, from the faintest shade to the ink, and the page's own; below about 0.52 an
# orange turns brown
DARK_L = (0.52, 0.93)
DARK_PAPER_L = 0.19


def _oklch(rgb):
    """Lightness, chroma, and hue in radians of an sRGB triple."""
    c = np.asarray(rgb, dtype = float)
    lab = _LAB @ np.cbrt(_LMS @ np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))
    return float(lab[0]), float(np.hypot(lab[1], lab[2])), float(np.arctan2(lab[2], lab[1]))


def _from_oklch(L, C, h):
    """'#RRGGBB' of a lightness, chroma, and hue, with chroma reduced until the color fits in sRGB."""
    inv = np.linalg.inv(_LAB)
    while True:
        rgb = np.linalg.solve(_LMS, (inv @ np.array([L, C * np.cos(h), C * np.sin(h)])) ** 3)
        if (rgb.min() >= -1e-4 and rgb.max() <= 1 + 1e-4) or C < 1e-4:
            break
        C *= 0.95
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return '#%02X%02X%02X' % tuple(int(round(v * 255)) for v in rgb)


def _night(c, chroma = None):
    """A color for a dark page: its lightness mirrored into DARK_L, so that what stood out most on white stands out
    most on the dark page, with its hue kept, except that an orange or yellow made darker turns towards red, as a
    darkened marigold would read as olive."""
    L, C, h = _oklch(_hex(c) if isinstance(c, str) else c)
    lo, hi = DARK_L
    L2 = float(np.clip(hi - (L - 0.25) * (hi - lo) / 0.70, lo, hi))
    deg = np.degrees(h) % 360
    if 50 < deg < 115:
        deg -= (deg - 50) * float(np.clip((0.80 - L2) / 0.28, 0.0, 1.0))
    return _from_oklch(L2, C if chroma is None else chroma, np.radians(deg))


def _linear(rgb):
    c = np.asarray(rgb, dtype = float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    """OKLab of sRGB triples, along the last axis."""
    return np.cbrt(_linear(rgb) @ _LMS.T) @ _LAB.T


# APCA-W3 0.0.98G-4g (Somers, Myndex): what a color reaches against the page as lightness contrast Lc, by its bronze
# levels: text that is read (tick labels, axis names, legends, annotations), large or heavy text (from 24 points, or 16
# bold), and marks that carry data, which need to be discernible, while their shades stay free to recede
LC_TEXT, LC_LARGE, LC_MARK = 60.0, 45.0, 15.0
# Machado, Oliveira, and Fernandes (2009) at full severity, on linear sRGB
CVD = {'protan': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                           [-0.003882, -0.048116, 1.051998]]),
       'deutan': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                           [-0.011820, 0.042940, 0.968881]]),
       'tritan': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                           [0.004733, 0.691367, 0.303900]])}
# How far apart in OKLab two colors of one plot stay for every reader, as normal vision and each simulated deficiency
# see them; the closest pair of Okabe and Ito's palette, made for color-blind readers, is 0.076 apart
CVD_MIN = 0.07


def apca(color, page):
    """APCA's lightness contrast Lc of a color on a page, from 0 up to about 106 for black on white, as an absolute
    value: dark on light and light on dark count alike, each by its own curve."""
    ys = []
    for c in (color, page):
        y = float(np.dot(np.asarray(c, dtype = float) ** 2.4, (0.2126729, 0.7151522, 0.0721750)))
        ys.append(y if y > 0.022 else y + (0.022 - y) ** 1.414)
    fg, bg = ys
    if abs(bg - fg) < 0.0005:
        return 0.0
    sapc = (bg ** 0.56 - fg ** 0.57) * 1.14 if bg > fg else (bg ** 0.65 - fg ** 0.62) * 1.14
    return 0.0 if abs(sapc) < 0.1 else (abs(sapc) - 0.027) * 100.0


def seen(rgb):
    """A color as normal vision and as protanopia, deuteranopia, and tritanopia see it, as OKLab rows."""
    lin = _linear(rgb)
    rows = [lin] + [np.clip(m @ lin, 0.0, 1.0) for m in CVD.values()]
    return np.cbrt(np.array(rows) @ _LMS.T) @ _LAB.T


def apart(c1, c2):
    """How far apart two colors stay for every reader: their least OKLab distance under normal vision and the three
    simulated deficiencies."""
    return float(np.linalg.norm(seen(c1) - seen(c2), axis = 1).min())


def legible(rgb, page, target, step = 0.01):
    """A color that reaches APCA's `target` against `page`: the color itself if it does, else the nearest by lightness
    in its own hue, moved away from the page's, with its chroma kept where sRGB holds it."""
    if apca(rgb, page) >= target:
        return tuple(rgb)
    L, C, h = _oklch(rgb)
    way = -1.0 if _oklch(page)[0] > 0.5 else 1.0
    for k in range(1, int(1.0 / step) + 1):
        moved = _hex(_from_oklch(float(np.clip(L + way * k * step, 0.0, 1.0)), C, h))
        if apca(moved, page) >= target:
            return moved
    return (0.0, 0.0, 0.0) if way < 0 else (1.0, 1.0, 1.0)


def ramp(colors, t):
    """Piecewise linear color scale at t in [0, 1], for scalars and arrays alike; the result has shape (..., 3)."""
    rgb = np.array([_hex(c) if isinstance(c, str) else c for c in colors], dtype = float)
    # A missing value maps to the scale's start; whoever draws decides whether to draw it at all
    pos = np.clip(np.nan_to_num(np.asarray(t, dtype = float)), 0.0, 1.0) * (len(rgb) - 1)
    lo = np.minimum(pos.astype(int), len(rgb) - 2)
    w = (pos - lo)[..., None]
    return rgb[lo] * (1 - w) + rgb[lo + 1] * w


@dataclass
class Palette:
    """Two hues in four shades each (dark to pale), with text colors and effect settings.

    `point` is flat, ring, cut, sphere, or crust (a darker rim around the point's color). `bar` is flat, deep (gradient
    from the dark base), soft (gradient from the pale base), needle, or sear (browning from a pale base with height, on
    one scale across the plot, so that a taller bar reaches a darker tip). `ridge` is flat, deep, tint, or sear.
    `crust` is the width, in line widths, of a darker line under every line, 0 for none. `shadow` is the strength of a
    soft shadow that every mark casts in its own color. `caps` sets axis names in small spaced capitals. `paper` is the
    page's color, for a dark page with a light `ink`, where shades run from the strongest against the page to the
    faintest. `font` is a family as (name, regular file, bold file), found among the system's fonts, for a house or
    journal style; without it, or where its files are missing, Century Gothic or the bundled Comfortaa. `spacing` scales
    the gaps, pads, legend rows, and color bars that the plot code sets itself, which suit type of about ten points, so
    that type of five or six points sits as close to its axes as a journal page wants. `minus` is the sign numbers on
    axes are written with, such as a true minus, '−', in a font that has one. `sequential` and `diverging` take stops
    of their own, pale to dark and from the second pole through the middle to the first, in place of the ones made from
    the hues. `dark_settings` replaces settings of the palette that `dark` makes from this one, such as shades picked
    by hand.
    """
    name: str
    a: list
    b: list
    ink: str
    muted: str
    faint: str
    description: str = ''
    point: str = 'flat'
    point_scale: float = 1.0
    line_width: float = 2.0
    glow: bool = False
    bar: str = 'flat'
    bar_width: float = 0.74
    ridge: str = 'flat'
    ridge_crest: float = 1.0
    heat_gap: float = 1.0
    heat_round: float = 0.0
    shadow: float = 0.0
    spines: tuple = ('left', 'bottom')
    spine_width: float = 1.0
    spine_offset: float = 0.0
    spine_round: bool = False
    spine_color: str = ''
    tick_length: float = 4.0
    tick_size: float = 10.0
    tick_bold: bool = False
    tick_color: str = ''
    label_size: float = 11.0
    label_bold: bool = False
    label_color: str = ''
    grid: bool = False
    caps: bool = False
    title_size: float = 25.0
    title_bold: bool = True
    title_color: str = ''
    subtitle_size: float = 11.0
    subtitle_bold: bool = False
    subtitle_color: str = ''
    text_bold: bool = False
    paper: str = '#FFFFFF'
    font: tuple = ()
    minus: str = '-'
    spacing: float = 1.0
    sequential_stops: tuple = ()
    diverging_stops: tuple = ()
    dark_settings: dict = field(default_factory = dict)
    extra: dict = field(default_factory = dict)
    crust: float = 0.0

    def __post_init__(self):
        # A misspelled effect would otherwise quietly draw the flat one
        for setting, allowed in (('point', ('flat', 'ring', 'cut', 'sphere', 'crust')),
                                 ('bar', ('flat', 'deep', 'soft', 'needle', 'sear')),
                                 ('ridge', ('flat', 'deep', 'tint', 'sear'))):
            if getattr(self, setting) not in allowed:
                raise ValueError(f"A palette's {setting} has to be one of {', '.join(allowed)}, "
                                 f'got {getattr(self, setting)!r}')
        if len(self.a) != 4 or len(self.b) != 4:
            raise ValueError('A palette has two hues of four shades each, from dark to pale')
        if self.font and len(self.font) != 3:
            raise ValueError("A palette's font is (name, regular file, bold file)")
        if any(0 < len(v) < 2 for v in (self.sequential_stops, self.diverging_stops)):
            raise ValueError("A palette's own scale needs at least two stops")

    def color(self, c):
        """An RGB triple from a palette name (a0 to b3, ink, muted, faint, paper), a hex string, a CSS name or a
        tuple."""
        if c is None or isinstance(c, (tuple, list, np.ndarray)):
            return None if c is None else tuple(float(v) for v in c[:3])
        if c in NAMED:
            return None if NAMED[c] is None else _hex(NAMED[c])
        if len(c) == 2 and c[0] in 'ab' and c[1] in '0123':
            return _hex(getattr(self, c[0])[int(c[1])])
        if c in ('ink', 'muted', 'faint', 'paper'):
            return _hex(getattr(self, c))
        if c.lower() in CSS:
            return _hex(CSS[c.lower()])
        try:
            return _hex(c)
        except ValueError:
            raise ValueError(f'Unknown color {c!r}: use a palette name (a0 to b3, ink, muted, faint, paper), '
                             'hex, or a CSS name')

    def css(self, c):
        """A color as '#RRGGBB', for writing it into HTML or SVG by hand."""
        return '#%02X%02X%02X' % tuple(int(round(v * 255)) for v in self.color(c))

    def colors(self, n = 6):
        """Categorical colors in fixed order, grouped by hue: related categories should take neighboring slots."""
        order = [_hex(c) for c in (self.a[0], self.a[1], self.a[2], self.b[1], self.b[2], self.b[0], self.a[3],
                                   self.b[3])]
        if n <= len(order):
            return order[:n]
        # Past eight categories, more shades of the same two hues rather than new hues
        k = (n - len(order) + 1) // 2
        steps = (np.arange(k) + 0.5) / k
        extra = [tuple(c) for hue in (self.a, self.b) for c in ramp(hue, steps)]
        return order + extra[:n - len(order)]

    def contrast(self, n = 2):
        """Categorical colors alternating between the two hues, for categories that have to stand apart, such as sample
        groups; `colors` keeps related categories on neighboring shades instead."""
        order = [_hex(hue[i]) for i in (1, 0, 2, 3) for hue in (self.a, self.b)]
        return order[:n] if n <= len(order) else order + self.colors(n)[len(order):]

    def groups(self, *sizes):
        """Colors for categories grouped by hue, such as groups(3, 2): three shades of one hue and two of the other."""
        if len(sizes) > 2:
            raise ValueError('A palette has two hues, so at most two groups')
        out = []
        for hue, n in zip((self.a, self.b), sizes):
            if n > 4:
                raise ValueError('A hue holds four shades')
            out += [_hex(hue[i]) for i in {0: [], 1: [1], 2: [1, 2], 3: [0, 1, 2], 4: [0, 1, 2, 3]}[n]]
        return out

    def sequential(self, t):
        """Magnitude scale from pale to dark in the main hue, or through the palette's own stops, at t in [0, 1]."""
        return ramp(self.sequential_stops or [self.a[3], self.a[2], self.a[1], self.a[0]], t)

    def sequential_r(self, t):
        """The magnitude scale from dark to pale, for values that matter most when low, such as distances."""
        return self.sequential(1.0 - np.asarray(t, dtype = float))

    def diverging(self, t):
        """Two-pole scale through the page's color: 0 is the second hue at its strongest, 0.5 is the page, 1 is the main
        hue at its strongest; or through the palette's own stops."""
        return ramp(self.diverging_stops or [self.b[0], self.b[1], self.b[3], self.paper, self.a[3], self.a[1],
                                             self.a[0]], t)

    def on(self, c):
        """Whichever of white or a dark text color reads better over a color, by its luminance: the palette's ink, or
        black where a dark page makes its ink light."""
        light = lambda v: np.dot(self.color(v), (0.2126, 0.7152, 0.0722)) >= 0.5
        return 'white' if not light(c) else 'black' if light('ink') else 'ink'

    @property
    def is_dark(self):
        """Whether the page is dark, so that its shades run from light to dark."""
        return float(np.dot(self.color('paper'), (0.2126, 0.7152, 0.0722))) < 0.5

    def dark(self, paper = None):
        """This palette for a dark page, such as an app's dark theme: every color keeps its hue, and its lightness is
        mirrored, so that the shade that stood out most on white stands out most on the dark page, and the effects
        follow. `paper` is the page's color, by default a near-black in the main hue; a palette's `dark_settings`
        replace what is derived here. A dark palette is returned as it is."""
        if self.is_dark:
            return self
        key = (id(self), paper)
        if key not in _darks:
            out = {}
            for k in ('a', 'b'):
                hue = getattr(self, k)
                cs = [_oklch(_hex(s))[1] for s in hue]
                # Pale shades carry little chroma, so the light shades they turn into borrow their mirror's
                out[k] = [_night(s, max(cs[i], cs[3 - i]) * 0.92) for i, s in enumerate(hue)]
            for k in ('ink', 'muted', 'spine_color', 'tick_color', 'label_color', 'title_color', 'subtitle_color'):
                if getattr(self, k):
                    out[k] = _night(getattr(self, k))
            for k in ('sequential_stops', 'diverging_stops'):
                out[k] = tuple(_night(self.color(s)) for s in getattr(self, k))
            _, C, h = _oklch(_hex(self.a[0]))
            L, C, h = _oklch(self.color(paper)) if paper else (DARK_PAPER_L, min(C, 0.015), h)
            out['paper'], out['faint'] = _from_oklch(L, C, h), _from_oklch(L + 0.07, C, h)
            _darks[key] = (self, replace(self, name = self.name + '-dark', **{**out, **self.dark_settings},
                                         dark_settings = {}))
        return _darks[key][1]

    def shades(self, c):
        """The (dark, mid, light) neighbors of a color: its own hue's shades if it is in the palette, or theirs moved
        by as much lightness where it is one of them made lighter or darker in its own hue, as the legibility check
        makes them, so that it keeps the palette's gradients; else mixes."""
        rgb = self.color(c)
        for hue in (self.a, self.b):
            for i, s in enumerate(hue):
                if np.allclose(_hex(s), rgb, atol = 1e-3):
                    return _hex(hue[max(i - 1, 0)]), _hex(s), _hex(hue[min(i + 1, 3)])
        L, C, h = _oklch(rgb)
        for hue in (self.a, self.b):
            for i, s in enumerate(hue):
                Ls, Cs, hs = _oklch(_hex(s))
                turned = abs((h - hs + np.pi) % (2 * np.pi) - np.pi)
                if turned < 0.02 and abs(L - Ls) <= 0.21 and 0.6 * Cs <= C <= Cs + 1e-3:
                    moved = [_oklch(_hex(hue[j])) for j in (max(i - 1, 0), min(i + 1, 3))]
                    dark, light = (_hex(_from_oklch(min(max(Lj + L - Ls, 0.0), 1.0), Cj, hj)) for Lj, Cj, hj in moved)
                    return dark, rgb, light
        return mix(rgb, 0.62, (0.0, 0.0, 0.0)), rgb, mix(rgb, 0.55)


LUMEN = Palette('lumen', ['#063B4C', '#0B7A8C', '#2CC4C9', '#A8EDE8'], ['#B34700', '#F28C0F', '#FFC042', '#FFE7A8'],
                '#063B4C', '#55808A', '#E3F1F1',
                'Peacock and marigold: shade gradients in the marks, spheres, glowing lines, no axis lines.',
                point = 'sphere', point_scale = 1.0, line_width = 2.6, glow = True, bar = 'deep', ridge = 'deep',
                ridge_crest = 1.6, heat_gap = 1.3, shadow = 0.31, spines = (), tick_length = 0.0, tick_size = 10.5,
                tick_color = '#0B7A8C',
                label_size = 11.5, label_color = '#0B7A8C', grid = True, title_color = '#063B4C',
                subtitle_color = '#0B7A8C')
COUTURE = Palette('couture', ['#2A1036', '#6E2A72', '#B65FA0', '#EBC6DC'], ['#8A5A12', '#C9962E', '#E6C46A', '#F6E8B8'],
                  '#2A1036', '#7B6A82', '#EFE8EE',
                  'Plum and gold: hairline axes held away from the data, slender marks, large light titles, spaced '
                  'capitals.',
                  point = 'ring', point_scale = 0.42, line_width = 1.0, bar = 'needle', bar_width = 0.25,
                  ridge = 'tint', ridge_crest = 0.7, heat_gap = 0.35, spine_width = 0.45, spine_offset = 12.0,
                  spine_color = '#2A1036',
                  tick_length = 7.0, tick_size = 8.6, tick_color = '#7B6A82', label_size = 7.8, label_color = '#8A5A12',
                  caps = True, title_size = 31.0, title_bold = False, title_color = '#2A1036', subtitle_size = 7.6,
                  subtitle_color = '#8A5A12', extra = {'subtitle_caps': True})
POSTER = Palette('poster', ['#0B3D2E', '#14704F', '#38A878', '#B6E5CC'], ['#A92B12', '#E8502A', '#F78A5E', '#FCD3BF'],
                 '#0B3D2E', '#3F6A5A', '#E2EFE8',
                 'Forest and vermilion: heavy rounded axes, big bold type, thick solid marks with white gaps.',
                 point = 'cut', point_scale = 1.7, line_width = 5.4, bar = 'flat', bar_width = 0.82, ridge = 'flat',
                 ridge_crest = 3.0, heat_gap = 2.0, spine_width = 3.6, spine_round = True, spine_color = '#0B3D2E',
                 tick_length = 0.0,
                 tick_size = 13.0, tick_bold = True, tick_color = '#0B3D2E', label_size = 13.0, label_bold = True,
                 label_color = '#0B3D2E', title_size = 31.0, title_color = '#0B3D2E', subtitle_size = 12.5,
                 subtitle_bold = True, subtitle_color = '#E8502A', text_bold = True)
AURA = Palette('aura', ['#5A0F2E', '#A3214E', '#DB4F7A', '#F6BFD0'], ['#0C4A4F', '#128C8C', '#4CC9BE', '#C1EFE8'],
               '#3D0A20', '#7F5A68', '#F4E6EB',
               'Burgundy and teal: marks float on soft shadows in their own color, with rounded forms and no axis '
               'lines.',
               point = 'flat', point_scale = 1.25, line_width = 3.6, bar = 'soft', ridge = 'flat', ridge_crest = 1.1,
               heat_gap = 2.0, heat_round = 0.2, shadow = 0.62, spines = (), tick_length = 0.0, tick_size = 10.0,
               tick_color = '#7F5A68', label_color = '#7F5A68', title_color = '#A3214E', subtitle_color = '#7F5A68')
palettes = {p.name: p for p in (LUMEN, COUTURE, POSTER, AURA)}
_default = ['lumen', False]
_library = ['lumen']
_dark = [False, None]
# Dark palettes made by Palette.dark, by (id of the light one, page color), each kept with its light one so that the id
# stays its own
_darks = {}


def _named(name):
    """A palette by instance or name, where a shipped or registered name with '-dark' gives its dark version."""
    if isinstance(name, Palette):
        return name
    if name in palettes:
        return palettes[name]
    if name.endswith('-dark') and name[:-5] in palettes:
        return palettes[name[:-5]].dark(_dark[1])
    raise ValueError(f'Unknown palette {name!r}; available: {", ".join(palettes)}, each also with -dark')


def get_palette(name = None, default = None):
    """A palette by name or instance; None gives the current default. A library with a palette of its own passes it as
    `default`, which gives way only to a palette chosen with set_palette, not to another library's default_palette. In
    dark mode, every palette comes as its dark version."""
    name = default if name is None and not _default[1] else name
    pal = _named(_default[0] if name is None else name)
    return pal.dark(_dark[1]) if _dark[0] else pal


palette = get_palette


def set_palette(name):
    """Choose the palette that new figures use; None leaves the choice to a library's default_palette again."""
    _default[:] = (_library[0], False) if name is None else (_named(name).name, True)


def default_palette(name):
    """A library's choice of palette: it applies unless the user has called set_palette, before or after."""
    _library[0] = _named(name).name
    if not _default[1]:
        _default[0] = _library[0]


def dark_mode(on = True, paper = None):
    """Draw new figures for a dark page, as for an app or a console in a dark theme: every palette, whether chosen by
    name, passed as an instance, or a library's default, comes as its dark version (Palette.dark), on `paper` if given,
    such as the theme's own surface color. dark_mode(False) goes back to light pages. Returns the setting it replaced,
    as (on, paper), so that dark_mode(*previous) restores it, as when saving a figure for print from a dark console."""
    previous = tuple(_dark)
    _dark[:] = (bool(on), paper if on else None)
    return previous


def register_palette(pal):
    """Add a custom palette under its own name."""
    palettes[pal.name] = pal
    return pal


# sRGB in linear light to CIE XYZ under D50, the connection space of ICC profiles, by Bradford adaptation
_XYZ = np.array([[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169],
                 [0.0139322, 0.0971045, 0.7141733]])
_D50 = np.array([0.9642, 1.0, 0.8249])
# A print's colors count as reached within this CIEDE2000 difference, about the least a viewer notices side by side
REACH = 1.0
# Ink coverage the solved separations keep to, 300% as SWOP asks; a grid of this many steps per channel, through which
# large images go; and the steps of the grid over CIELAB that the default press's own profile is built on
INK_LIMIT, IMAGE_GRID, PROFILE_GRID = 3.0, 17, 9


def _xyz_lab(xyz):
    """CIELAB of CIE XYZ under D50, along the last axis."""
    t = np.asarray(xyz, dtype = float) / _D50
    f = np.where(t > (6 / 29) ** 3, np.cbrt(np.maximum(t, 0.0)), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis = -1)


def _lab_xyz(lab):
    """CIE XYZ under D50 of CIELAB, along the last axis."""
    lab = np.asarray(lab, dtype = float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis = -1)
    return np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _D50


def cielab(rgb):
    """CIELAB under D50 of sRGB colors along the last axis, as ICC profiles connect them."""
    return _xyz_lab(_linear(rgb) @ _XYZ.T)


def _srgb(lab):
    """sRGB of CIELAB colors under D50, clipped into its gamut."""
    lin = np.clip(_lab_xyz(lab) @ np.linalg.inv(_XYZ).T, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def _oklab_srgb(ok):
    """sRGB of OKLab colors along the last axis, clipped into its gamut."""
    lin = np.clip((ok @ np.linalg.inv(_LAB).T) ** 3 @ np.linalg.inv(_LMS).T, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def delta_e(lab1, lab2):
    """CIEDE2000 differences between CIELAB colors along the last axis, the measure print is judged by: about 1 is the
    least a viewer notices side by side, and 2 to 3 what a press proof is held to."""
    (L1, a1, b1), (L2, a2, b2) = (np.moveaxis(np.asarray(v, dtype = float), -1, 0) for v in (lab1, lab2))
    cb7 = ((np.hypot(a1, b1) + np.hypot(a2, b2)) / 2) ** 7
    g = 1.5 - 0.5 * np.sqrt(cb7 / (cb7 + 25.0 ** 7))
    c1, c2 = np.hypot(a1 * g, b1), np.hypot(a2 * g, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1 * g)) % 360, np.degrees(np.arctan2(b2, a2 * g)) % 360
    flat = c1 * c2 == 0
    dh = np.where(flat, 0.0, (h2 - h1 + 180) % 360 - 180)
    hb = np.where(flat, h1 + h2, np.where(np.abs(h1 - h2) <= 180, (h1 + h2) / 2,
                                          np.where(h1 + h2 < 360, (h1 + h2 + 360) / 2, (h1 + h2 - 360) / 2)))
    t = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
    lb, cb = (L1 + L2) / 2 - 50, (c1 + c2) / 2
    dl = (L2 - L1) / (1 + 0.015 * lb ** 2 / np.sqrt(20 + lb ** 2))
    dc = (c2 - c1) / (1 + 0.045 * cb)
    dH = 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh) / 2) / (1 + 0.015 * cb * t)
    rt = -np.sin(np.radians(60 * np.exp(-((hb - 275) / 25) ** 2))) * 2 * np.sqrt(cb ** 7 / (cb ** 7 + 25.0 ** 7))
    return np.sqrt(dl ** 2 + dc ** 2 + dH ** 2 + rt * dc * dH)


def _icc_table(data, at):
    """An ICC table of 8 or 16 bits (lut8, lut16) at an offset: its input curves, grid, and output curves in [0, 1],
    and whether it has 16 bits, which sets how it writes CIELAB."""
    kind, nin, nout, g = data[at:at + 4], data[at + 8], data[at + 9], data[at + 10]
    if kind not in (b'mft1', b'mft2'):
        raise ValueError(f'viscose reads the 8- and 16-bit tables (lut8, lut16) that version 2 print profiles hold, '
                         f'not {kind.decode("latin-1")!r}')
    wide = kind == b'mft2'
    n, m = struct.unpack('>HH', data[at + 48:at + 52]) if wide else (256, 256)
    unit, p, parts = np.dtype('>u2' if wide else 'u1'), at + (52 if wide else 48), []
    for count in (nin * n, g ** nin * nout, nout * m):
        parts.append(np.frombuffer(data, unit, count, p).astype(float) / (65535.0 if wide else 255.0))
        p += count * unit.itemsize
    return parts[0].reshape(nin, n), parts[1].reshape((g,) * nin + (nout,)), parts[2].reshape(nout, m), wide


def _icc_apply(table, x, slope = False):
    """An ICC table at points `x` of shape (N, inputs) in [0, 1], by multilinear interpolation on its grid, and with
    `slope` also its derivatives, of shape (N, outputs, inputs), through its curves as well."""
    curves, grid, out, _ = table
    nin, g, n, m = x.shape[1], grid.shape[0], curves.shape[1], out.shape[1]
    u = np.stack([np.interp(x[:, i], np.linspace(0, 1, n), curves[i]) for i in range(nin)], 1) * (g - 1)
    lo = np.minimum(u.astype(int), g - 2)
    # Every corner of each point's cell at once, weighted by the point's place in it
    f, strides = u - lo, g ** np.arange(nin - 1, -1, -1)
    corners = np.array(list(itertools.product((0, 1), repeat = nin)))
    parts = np.where(corners, f[:, None, :], 1 - f[:, None, :])
    found = grid.reshape(-1, grid.shape[-1])[(lo @ strides)[:, None] + corners @ strides]
    y = np.einsum('nk,nkc->nc', parts.prod(axis = 2), found)
    result = np.stack([np.interp(y[:, k], np.linspace(0, 1, m), out[k]) for k in range(y.shape[1])], 1)
    if not slope:
        return result
    # Each piecewise linear curve's slope where it is read, and the grid's by the weights' derivatives
    step = lambda c, v, k: np.diff(c, axis = 1)[np.arange(len(c))[:, None], np.clip((v * (k - 1)).astype(int), 0,
                                                                                    k - 2)].T * (k - 1)
    grads = np.stack([np.prod(np.delete(parts, i, axis = 2), axis = 2) * np.where(corners[:, i], 1.0, -1.0)
                      for i in range(nin)], axis = 2)
    inner = np.einsum('nki,nkc->nci', grads, found) * (g - 1)
    return result, step(out, y.T, m)[:, :, None] * inner * step(curves, x.T, n)[:, None, :]


def _mft2(grid):
    """A lut16 table, as ICC writes it, of a grid of values in [0, 1] whose last axis holds its outputs, with straight
    curves around it."""
    nin, nout, g = grid.ndim - 1, grid.shape[-1], grid.shape[0]
    head = b'mft2' + bytes(4) + bytes((nin, nout, g, 0)) + struct.pack('>9i', 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    line = np.array([0, 65535], '>u2').tobytes()
    clut = (np.clip(grid, 0.0, 1.0) * 65535.0 + 0.5).astype('>u2').tobytes()
    return head + struct.pack('>HH', 2, 2) + line * nin + clut + line * nout


class Press:
    """A print condition, for PDF/X in CMYK and for soft proofs of how a figure prints. By default, U.S. web offset on
    coated paper, CGATS TR 001 (SWOP), as Clinton Ingram's compact profile in the public domain (CC0) characterizes it,
    with separations that viscose solves itself: black for neutrals and shadows, less of it as color grows, and at most
    `ink_limit` of ink in all, 300% as SWOP asks. `profile` takes a printer's or a journal's own ICC output profile
    instead, such as FOGRA39 or GRACoL, as a path or bytes, whose table from CMYK to CIELAB the separations are then
    solved on; `condition` names its registered print condition, as the PDF/X's output intent does, and `info`
    describes it. Colors a press cannot print keep their lightness and hue and give up only the chroma they must, so
    that a palette's hues print as the same hues, and dark inks keep their steps above the press's own black."""
    def __init__(self, profile = None, condition = None, info = None, ink_limit = INK_LIMIT,
                 registry = 'http://www.color.org'):
        own = profile is not None
        data = (bytes(profile) if isinstance(profile, (bytes, bytearray)) else Path(profile).read_bytes()) if own else (
            Path(__file__).parent / 'swop.icc').read_bytes()
        if data[16:20] != b'CMYK' or data[20:24] != b'Lab ':
            raise ValueError('A press needs an ICC profile of CMYK that connects through CIELAB')
        count = struct.unpack('>I', data[128:132])[0]
        tags = {sig: at for sig, at, _ in struct.iter_unpack('>4sII', data[132:132 + 12 * count])}
        if b'A2B1' not in tags and b'A2B0' not in tags:
            raise ValueError("A press's profile needs a table from CMYK to CIELAB (A2B1 or A2B0)")
        self.forward = _icc_table(data, tags.get(b'A2B1', tags.get(b'A2B0')))
        name, at = 'U.S. Web Coated (SWOP)', tags.get(b'desc')
        if own:
            name = data[at + 12:at + 12 + struct.unpack('>I', data[at + 8:at + 12])[0]].split(b'\0')[0].decode(
                'latin-1') if at is not None and data[at:at + 4] == b'desc' else 'Custom'
        self.condition = condition or ('CGATS TR 001' if not own else name)
        self.info, self.registry, self.ink_limit = info or name, registry, ink_limit
        self._data, self._own, self._seen, self._grid, self._icc = data, own, {}, None, None
        # The press's own black, down to which black point compensation scales sRGB's
        self.black = self.lab(np.array([[*[min(1.0, (ink_limit - 1.0) / 3.0)] * 3, 1.0]]))[0]

    def lab(self, cmyk, slope = False):
        """CIELAB of CMYK on this press, by its profile, and with `slope` its derivatives by each ink."""
        found = _icc_apply(self.forward, np.asarray(cmyk, dtype = float), slope)
        (v, d), k = found if slope else (found, None), 65535 / 65280 if self.forward[3] else 1.0
        lab = np.column_stack([100 * k * v[:, 0], 255 * k * v[:, 1] - 128, 255 * k * v[:, 2] - 128])
        return (lab, d * (np.array([100.0, 255.0, 255.0]) * k)[None, :, None]) if slope else lab

    def _ink(self, lab, rgb = None, black = None):
        """CMYK for CIELAB colors on this press, and how far each misses, solved on the profile's table from CMYK to
        CIELAB, which characterizes the press: black from lightness, less of it as color grows, then cyan, magenta,
        and yellow by damped Newton steps from the naive separation of `rgb` where given, trying more and less black
        where the ink limit or the color asks for it, or only the `black` given. A profile's own table back from CIELAB
        is a separation strategy of its maker's, which can pull dark colors to gray, as Windows' SWOP profile does by
        13 CIEDE2000 on a dark red its own press prints."""
        n, shifts = len(lab), np.array([0.0, -0.25, 0.25, 0.5] if black is None else [0.0])
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        plan = (np.clip((75.0 - lab[:, 0]) / 75.0, 0.0, 1.0) ** 1.5 * np.clip(1.0 - chroma / 60.0, 0.0, 1.0)
                if black is None else black)
        # Each black tried for every color at once, its cyan, magenta, and yellow from the same start
        k, target = np.clip(plan[None, :] + shifts[:, None], 0.0, 1.0).ravel(), np.tile(lab, (len(shifts), 1))
        x = np.full((len(k), 3), 0.3) if rgb is None else np.clip((1.0 - np.tile(rgb, (len(shifts), 1)) - k[:, None])
                                                                   / np.maximum(1.0 - k[:, None], 1e-3), 0.0, 1.0)
        for _ in range(8 if rgb is not None else 12):
            found, jac = self.lab(np.column_stack([x, k]), True)
            f, jac = found - target, jac[:, :, :3]
            jt = np.swapaxes(jac, 1, 2)
            x = np.clip(x - np.linalg.solve(jt @ jac + 1e-2 * np.eye(3), jt @ f[..., None])[..., 0], 0.0, 1.0)
        ink = np.column_stack([x, k])
        off = delta_e(self.lab(ink), target).reshape(len(shifts), n)
        # The separation that misses least wins, then the one that keeps to the ink limit and to the planned black
        cost = off + 100.0 * np.maximum(ink.sum(1) - self.ink_limit, 0.0).reshape(len(shifts), n) + 0.5 * np.abs(
            k.reshape(len(shifts), n) - plan)
        pick = cost.argmin(0)
        return ink.reshape(len(shifts), n, 4)[pick, np.arange(n)], off[pick, np.arange(n)]

    def _within(self, rgb):
        """CMYK of sRGB colors, each brought into the press's gamut first: black point compensation scales sRGB's
        black down to the press's own, as Adobe's does in XYZ, and a color still out of reach keeps its OKLab lightness
        and hue and gives up the least chroma, by bisection, that brings it within REACH."""
        black = _lab_xyz(self.black)
        within = lambda c: _xyz_lab(_lab_xyz(cielab(c)) * (1 - black / _D50) + black)
        ink, miss = self._ink(within(rgb), rgb)
        far = np.flatnonzero(miss > REACH)
        if len(far):
            ok = _oklab(rgb[far])
            lo, hi, keep = np.zeros(len(far)), np.ones(len(far)), ink[far]
            # With the black each color first took, which its chroma barely moves
            for _ in range(6):
                mid = (lo + hi) / 2
                less = _oklab_srgb(np.column_stack([ok[:, 0], ok[:, 1:] * mid[:, None]]))
                trial, off = self._ink(within(less), less, ink[far, 3])
                fits = off <= REACH
                lo, hi = np.where(fits, mid, lo), np.where(fits, hi, mid)
                keep[fits] = trial[fits]
            ink[far] = keep
        # Less ink than the finest halftone dot prints as none
        return np.where(ink < 0.002, 0.0, ink)

    def separate(self, rgb):
        """CMYK in [0, 1] for sRGB colors of shape (N, 3) in [0, 1], as `_within` brings them in: each color is worked
        out once, and many at a time, as an image's, go through a grid of IMAGE_GRID steps per channel."""
        rgb = np.asarray(rgb, dtype = float).reshape(-1, 3)
        if len(rgb) > IMAGE_GRID ** 3 // 4:
            if self._grid is None:
                axis = np.linspace(0.0, 1.0, IMAGE_GRID)
                nodes = np.stack(np.meshgrid(axis, axis, axis, indexing = 'ij'), -1).reshape(-1, 3)
                self._grid = self._within(nodes).reshape(IMAGE_GRID, IMAGE_GRID, IMAGE_GRID, 4)
            straight = np.tile(np.linspace(0.0, 1.0, 2), (4, 1))
            return _icc_apply((straight[:3], self._grid, straight, True), np.clip(rgb, 0.0, 1.0))
        keys = ((np.clip(rgb, 0.0, 1.0) * 1023 + 0.5).astype(np.int64) @ np.array([1 << 20, 1 << 10, 1])).tolist()
        new = [k for k in dict.fromkeys(keys) if k not in self._seen]
        if new:
            found = np.array([[k >> 20, k >> 10 & 1023, k & 1023] for k in new]) / 1023.0
            self._seen.update(zip(new, self._within(found)))
        return np.array([self._seen[k] for k in keys])

    def printed(self, rgb):
        """sRGB of how colors print, for a soft proof on screen, with the paper as white."""
        return _srgb(self.lab(self.separate(rgb)))

    def gamut(self, palette = None):
        """How each color of a palette prints: (name, screen, print, CIEDE2000 between them), the print as sRGB hex,
        farthest first. Below about 2 the print holds the color; past that, the press has taken chroma away."""
        pal = get_palette(palette)
        names = [f'{h}{i}' for h in 'ab' for i in range(4)] + ['ink', 'muted', 'faint']
        rgb = np.array([pal.color(n) for n in names])
        out = self.printed(rgb)
        far = delta_e(cielab(rgb), cielab(out))
        rows = [(n, pal.css(n), '#%02X%02X%02X' % tuple(int(round(v * 255)) for v in o), round(float(d), 2))
                for n, o, d in zip(names, out, far)]
        return sorted(rows, key = lambda r: -r[3])

    def icc(self):
        """The ICC profile a PDF/X names in its output intent: the press's own, or for the default press a version 2
        output profile around the public-domain table, with separations and a gamut on a grid of PROFILE_GRID steps
        over CIELAB, solved as `separate` solves them."""
        if self._own:
            return self._data
        if self._icc is not None:
            return self._icc
        axis = np.linspace(0.0, 1.0, PROFILE_GRID)
        v = np.stack(np.meshgrid(axis, axis, axis, indexing = 'ij'), -1).reshape(-1, 3) * 65535 / 65280
        lab = np.column_stack([v[:, 0] * 100, v[:, 1] * 255 - 128, v[:, 2] * 255 - 128])
        ink, miss = self._ink(lab)
        far = np.flatnonzero(miss > REACH)
        # CIELAB the press cannot reach comes in by chroma, at its own lightness and hue
        lo, hi, keep = np.zeros(len(far)), np.ones(len(far)), ink[far]
        for _ in range(6):
            mid = (lo + hi) / 2
            trial, off = self._ink(np.column_stack([lab[far, 0], lab[far, 1:] * mid[:, None]]))
            fits = off <= REACH
            lo, hi = np.where(fits, mid, lo), np.where(fits, hi, mid)
            keep[fits] = trial[fits]
        ink[far] = keep
        grid = (PROFILE_GRID,) * 3
        b2a, gamt = _mft2(ink.reshape(*grid, 4)), _mft2((miss > REACH).astype(float).reshape(*grid, 1))
        count = struct.unpack('>I', self._data[128:132])[0]
        found = {sig: self._data[at:at + size] for sig, at, size in
                 struct.iter_unpack('>4sII', self._data[132:132 + 12 * count])}
        name = ('viscose ' + self.info).encode('latin-1')
        desc = b'desc' + bytes(4) + struct.pack('>I', len(name) + 1) + name + b'\0' + bytes(78)
        tags = [(b'desc', desc), (b'cprt', b'text' + bytes(4) + b'Public domain (CC0)\0'), (b'wtpt', found[b'wtpt']),
                (b'A2B0', found[b'A2B0']), (b'A2B1', found[b'A2B0']), (b'A2B2', found[b'A2B0']), (b'B2A0', b2a),
                (b'B2A1', b2a), (b'B2A2', b2a), (b'gamt', gamt)]
        # Tags that share their data share its place, as ICC allows
        body, table, where, start = b'', b'', {}, 132 + 12 * len(tags)
        for sig, blob in tags:
            if blob not in where:
                where[blob] = start + len(body)
                body += blob + bytes(-len(blob) % 4)
            table += struct.pack('>4sII', sig, where[blob], len(blob))
        head = (struct.pack('>I', start + len(body)) + bytes(4) + bytes((2, 0x10, 0, 0)) + b'prtrCMYKLab '
                + struct.pack('>6H', 2026, 11, 12, 0, 0, 0) + b'acsp' + bytes(24) + struct.pack('>I', 1)
                + struct.pack('>3i', 63190, 65536, 54061) + b'vscs' + bytes(44))
        self._icc = head + struct.pack('>I', len(tags)) + table + body
        return self._icc


_presses = {}


def press(given = True):
    """A Press from what a figure's output was given: True for the default press, an ICC profile's path, or a Press,
    which stays as it is. Presses from paths are made once."""
    if isinstance(given, Press):
        return given
    key = 'swop' if given is True else str(given)
    if key not in _presses:
        _presses[key] = Press(None if given is True else given)
    return _presses[key]
