"""Palettes: a few colors in shades, plus the special effects that belong to each palette."""
from dataclasses import dataclass, field, replace
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

    `point` is flat, ring, cut, or sphere. `bar` is flat, deep (gradient from the dark base), soft (gradient from the
    pale base), or needle. `ridge` is flat, deep, or tint. `shadow` is the strength of a soft shadow that every mark
    casts in its own color. `caps` sets axis names in small spaced capitals. `paper` is the page's color, for a dark
    page with a light `ink`, where shades run from the strongest against the page to the faintest. `font` is a family
    as (name, regular file, bold file), found among the system's fonts, for a house or journal style; without it, or
    where its files are missing, Century Gothic or the bundled Comfortaa. `spacing` scales the gaps, pads, legend
    rows, and color bars that the plot code sets itself, which suit type of about ten points, so that type of five or
    six points sits as close to its axes as a journal page wants. `minus` is the sign numbers on axes are written
    with, such as a true minus, '−', in a font that has one. `sequential` and `diverging` take stops of their own,
    pale to dark and from the second pole through the middle to the first, in place of the ones made from the hues.
    `dark_settings` replaces settings of the palette that `dark` makes from this one, such as shades picked by hand.
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

    def __post_init__(self):
        # A misspelled effect would otherwise quietly draw the flat one
        for setting, allowed in (('point', ('flat', 'ring', 'cut', 'sphere')),
                                 ('bar', ('flat', 'deep', 'soft', 'needle')), ('ridge', ('flat', 'deep', 'tint'))):
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
        """The (dark, mid, light) neighbors of a color: its own hue's shades if it is in the palette, else mixes."""
        rgb = self.color(c)
        for hue in (self.a, self.b):
            for i, s in enumerate(hue):
                if np.allclose(_hex(s), rgb, atol = 1e-3):
                    return _hex(hue[max(i - 1, 0)]), _hex(s), _hex(hue[min(i + 1, 3)])
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
    such as the theme's own surface color. dark_mode(False) goes back to light pages."""
    _dark[:] = (bool(on), paper if on else None)


def register_palette(pal):
    """Add a custom palette under its own name."""
    palettes[pal.name] = pal
    return pal
