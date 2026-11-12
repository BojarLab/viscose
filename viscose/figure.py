"""Figure: lays out axes, draws through glycorender's display list and writes PNG, PDF, or SVG."""
import base64
import builtins
import difflib
import functools
import importlib
import json
import math
import numbers
import os
import re
import struct
import subprocess
import sys
import tempfile
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from dataclasses import fields, replace
import numpy as np
from glycorender import pdfwrite, raster
from . import marks, text
from .axes import Axes, LABEL_MIN, PLOT_SHARE_MARGINS
from .palettes import (AURA, COUTURE, CVD_MIN, LC_LARGE, LC_MARK, LC_TEXT, LUMEN, POSTER, Palette, _darks, _from_oklch,
                       _hex, _named, _oklab, _oklch, apart, apca, dark_mode, get_palette, legible, press as _press)


# Shadow offset and blur in points; a blur cannot stay vector, so the PDF and SVG carry it as an image at this
# resolution
SHADOW_DX, SHADOW_DY, SHADOW_BLUR, SHADOW_DPI = 1.2, 5.0, 4.7, 100.0
# A byte's value in [0, 1], as glycorender reads image bytes
BYTES = (np.arange(256) / 255.0).astype(np.float32)
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
# Marks whose colors carry data, which APCA's floor for what must be discernible applies to; guides, such as grid lines,
# spans, and reference lines, recede on purpose, and translucent marks are pale on purpose
LEVELS = ('points', 'line', 'stems', 'vsegs', 'hsegs', 'bars', 'box', 'violin')
# Palettes made legible by Figure._legible, by id of the palette, each kept with it so that the id stays its own
_legibles = {}
# A figure's source, the calls and data that drew it, travels in its own file as JSON in this format, under this name
# and type: in a PDF as an attachment, in an SVG as base64 in a <metadata>, and in a PNG as a chunk of this type. Arrays
# up to SOURCE_JSON values are JSON numbers, which deflate as tightly as bytes on real data; larger ones follow the JSON
# as bytes, since JSON takes a second per 200,000 floats. A source past SOURCE_FLOOR bytes, deflated, that is larger
# than the rest of its file stays out unless asked for
SOURCE_FORMAT, SOURCE_NAME, SOURCE_MIME, SOURCE_CHUNK = 1, 'figure-source.json', 'application/vnd.viscose+json', b'vcSc'
SOURCE_JSON, SOURCE_FLOOR = 10000, 1 << 16
SOURCE_ABOUT = 'The calls and data that drew this figure, which viscose.open brings back as a figure'
# An SVG in both themes: what of an op's state is its paint, the classes of what only one theme draws, and what the
# drawing's id, which scopes its rules, and its own gradients' ids start with until its text is known
THEME_PAINT = ('fill_rgb', 'fill_alpha', 'fill_grad', 'stroke_rgb', 'stroke_alpha')
THEME_LIGHT, THEME_DARK, THEME_PREFIX = 'vl', 'vd', '\x01'
# What an op notes about what drew it, for PowerPoint and an interactive SVG, which is the same in either theme
TAGS = ('scope', 'artist', 'part', 'name', 'line', 'glyph', 'hover', 'series', 'hit', 'tick')
# How far everything else in a panel fades while an interactive SVG lights up what the pointer names
HOVER_FADE = 0.2
# An interactive SVG's script, which needs no library. It runs where the SVG does as a document or inline in a page,
# though not as an <img>, and only on the pointer: hover text for the element under it, in a box in the page's colors,
# and every element of the same names lit up in every panel, and in every other such figure on the page, while the rest
# of each panel that holds one fades, with the soft shadow. __TABLE__ is [texts, rows], each row holding, for an
# element, its own names, its hover text, the series it belongs to, as places in texts, which holds each text once, its
# axes, and whether it stands for that series, as a line or a legend entry does; data-h on an element or on the group of
# its layers gives its row
HOVER_SCRIPT = '''<script><![CDATA[(function () {
var NS = 'http://www.w3.org/2000/svg', me = document.currentScript, svg = me && me.parentNode;
if (!svg || !svg.querySelectorAll || !window.CustomEvent) return;
var T = __TABLE__, word = function (k) { return T[0][k]; };
var H = T[1].map(function (h) {
  return [h[0].map(word), h[1].map(word), h[2] === null ? null : h[2].map(word), h[3], h[4]];
});
var names = {}, panels = {}, own = {}, lit = [], tip = null, at = null, quiet = false;
function add(map, k, e) { (map[k] = map[k] || []).push(e); }
Array.prototype.forEach.call(svg.querySelectorAll('[data-h]'), function (e) {
  var i = +e.getAttribute('data-h'), h = H[i];
  add(own, i, e);
  if (h[3] !== null) add(panels, h[3], e);
  h[0].forEach(function (k) { add(names, k, e); });
  (h[2] || []).forEach(function (k) { add(names, k, e); });
});
function light(keys, mine) {
  lit.forEach(function (e) { e.classList.remove('vc-on', 'vc-off'); });
  lit = [];
  var on = new Set(mine || []), held = new Set();
  (keys || []).forEach(function (k) { (names[k] || []).forEach(function (e) { on.add(e); }); });
  svg.classList.toggle('vc-lit', on.size > 0);
  on.forEach(function (e) { var p = H[+e.getAttribute('data-h')][3]; if (p !== null) held.add(p); });
  held.forEach(function (p) {
    panels[p].forEach(function (e) { e.classList.add(on.has(e) ? 'vc-on' : 'vc-off'); lit.push(e); });
  });
}
document.addEventListener('viscose-focus', function (ev) { if (!quiet) light(ev.detail); });
function focus(keys, mine) {
  light(keys, mine);
  quiet = true;
  document.dispatchEvent(new CustomEvent('viscose-focus', {detail: keys}));
  quiet = false;
}
function place(ev) {
  var m = svg.getScreenCTM();
  if (!tip || !m) return;
  var p = svg.createSVGPoint();
  p.x = ev.clientX;
  p.y = ev.clientY;
  p = p.matrixTransform(m.inverse());
  var b = tip.getBBox(), gap = 14 / Math.abs(m.a || 1), v = svg.viewBox.baseVal, x = p.x + gap, y = p.y + gap;
  if (x + b.width > v.x + v.width) x = p.x - gap - b.width;
  if (y + b.height > v.y + v.height) y = p.y - gap - b.height;
  tip.setAttribute('transform', 'translate(' + (Math.max(v.x, x) - b.x) + ' ' + (Math.max(v.y, y) - b.y) + ')');
}
function show(said, ev) {
  if (tip) tip.remove();
  tip = null;
  if (!said.length) return;
  var m = svg.getScreenCTM(), k = 1 / Math.abs(m && m.a || 1), box = document.createElementNS(NS, 'rect');
  var t = document.createElementNS(NS, 'text');
  tip = document.createElementNS(NS, 'g');
  tip.setAttribute('class', 'vc-tip');
  tip.setAttribute('pointer-events', 'none');
  t.setAttribute('font-size', 12 * k);
  said.forEach(function (s, n) {
    var span = document.createElementNS(NS, 'tspan');
    span.setAttribute('x', 0);
    if (n) span.setAttribute('dy', 15 * k);
    else span.setAttribute('class', 'vc-head');
    span.textContent = s;
    t.appendChild(span);
  });
  tip.appendChild(box);
  tip.appendChild(t);
  svg.appendChild(tip);
  var b = t.getBBox(), pad = 6 * k;
  [['x', b.x - pad], ['y', b.y - pad], ['width', b.width + 2 * pad], ['height', b.height + 2 * pad], ['rx', 3 * k],
   ['stroke-width', 0.75 * k]].forEach(function (a) { box.setAttribute(a[0], a[1]); });
  place(ev);
}
function leave() {
  if (at === null) return;
  at = null;
  show([]);
  focus(null);
}
svg.addEventListener('pointerover', function (ev) {
  var e = ev.target.closest && ev.target.closest('[data-h]');
  if (!e || !svg.contains(e)) return leave();
  var i = +e.getAttribute('data-h'), h = H[i];
  if (i === at) return;
  at = i;
  focus(h[0].length ? h[0] : h[4] ? h[2] || [] : [], own[i]);
  show(h[1], ev);
});
svg.addEventListener('pointermove', function (ev) { if (at !== null) place(ev); });
svg.addEventListener('pointerleave', leave);
})();]]></script>
'''


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
    # How far the marks darken white in each channel, blurred, a channel per thread; a soft shadow is this darkening at
    # the given strength
    with ThreadPoolExecutor(3) as pool:
        dark = np.stack(list(pool.map(lambda i: raster._box_blur((1.0 - img[..., i]) * alpha, radius), range(3))),
                        axis = -1)
    dx, dy = int(round(SHADOW_DX * sc)), int(round(SHADOW_DY * sc))
    # Moved, with whatever leaves the page gone rather than back in at the opposite edge
    moved = np.zeros_like(dark)
    moved[dy:, dx:] = dark[:dark.shape[0] - dy, :dark.shape[1] - dx]
    dark = moved * strength
    out_a = dark.max(axis = -1)
    safe = np.where(out_a > 1e-4, out_a, 1.0)
    rgb = np.where(out_a[..., None] > 1e-4, 1.0 - dark / safe[..., None], 1.0)
    return np.dstack([np.clip(rgb, 0, 1), np.clip(out_a, 0, 1)])


def _path_data(segments):
    return ''.join('Z' if s[0] == 'h' else {'m': 'M', 'l': 'L', 'c': 'C'}[s[0]]
                   + ' '.join('%.2f' % v for v in s[1:]) for s in segments)


def _css(rgb):
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
            # The class that turns it dark, in an SVG of both themes
            cls = ' class="%s"' % op['theme'] if op.get('theme') else ''
            if op['kind'] == 'path':
                segments = op['path']
                paths.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f" stroke-linecap="round" '
                             'stroke-linejoin="round"%s />'
                             % (_path_data(segments), _css(op['stroke_rgb']), op['line_width'], cls))
            else:
                segments = text.outline(c, op['ttf'], op['text'], op['x'], op['y'], op['size']).ops
                paths.append('<path d="%s" fill="%s"%s />' % (_path_data(segments), _css(op['fill_rgb']), cls))
            right = max([right] + [x for s in segments for x in s[1::2]])
        comment = '<!-- %s -->\n' % key[2] if '--' not in key[2] else ''
        out.append('%s<g transform="translate(%.2f %.2f) matrix(%.6g %.6g %.6g %.6g 0 0)" data-right="%.2f">%s</g>'
                   % (comment, e, height - f, a, -b, cc, -d, e + a * right, ''.join(paths)))
    return '\n'.join(out)


def _with_ids(svg, ops):
    """Wrap each element's run of output in a group with its `gid`. The emitted body has one line per drawn op."""
    named = [o.get('gid') for o in _drawn(ops)]
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


def _drawn(ops):
    """The ops that svgout.emit writes a line for, in order: it skips paths without paint and text without ink."""
    return [o for o in ops if (o['fill'] or o['stroke'] if o['kind'] == 'path' else o['kind'] == 'image' or
                               o['text'].strip())]


def _hovered(svg, ops):
    """Each drawn op's line of svgout.emit's output marked with its element in an interactive SVG, by its row in the
    table that HOVER_SCRIPT reads, and that table: per element its own names, hover text, series, axes, and whether it
    stands for its series. An element of several ops, such as a point drawn in layers, is one group, so that it fades
    as one."""
    start = svg.index('\n', svg.index('<g transform="matrix(')) + 1
    end = svg.rindex('\n</g>\n</svg>')
    lines, drawn = svg[start:end].split('\n'), _drawn(ops)
    if len(lines) != len(drawn):
        raise RuntimeError(f'An interactive SVG needs a line per drawn op, got {len(lines)} for {len(drawn)}')
    found, table, rows = {}, [], []
    for n, op in enumerate(drawn):
        # The soft shadow fades while anything is lit, as faded marks over their full shadows would read as smudges,
        # and an unpainted disc over a point drawn in a group fills in while it is lit
        if op.get('shadow') or op.get('hit'):
            cls, at = 'vc-shadow' if op.get('shadow') else 'vc-hit', lines[n].find(' class="')
            lines[n] = (lines[n][:at] + ' class="%s ' % cls + lines[n][at + 8:] if at >= 0 else
                        re.sub(r'^<(\w+)', r'<\1 class="%s"' % cls, lines[n], count = 1))
        hover, series = op.get('hover'), op.get('series')
        if hover is None and series is None:
            rows.append(None)
            continue
        scope = op.get('scope')
        key = (scope, op.get('artist'), op.get('part'), op.get('name'), hover, series)
        if key not in found:
            found[key] = len(table)
            # A series is a label, or the labels of a group, such as a category and a hue level
            series = None if series is None else [str(k) for k in (series if isinstance(series, tuple) else (series, ))]
            names, said = hover if hover is not None else ((), (' · '.join(series), ))
            # A line, or a legend entry or label that names a series, stands for all of it; a tick label lights up
            # what it names, and keeps to no panel, as it never fades
            table.append([[str(k) for k in names], [str(s) for s in said], series,
                          None if scope is None or op.get('tick') else int(scope[0][1:]),
                          int(hover is None or hover == ((), ()))])
        rows.append(found[key])
    out = []
    for n, (line, i) in enumerate(zip(lines, rows)):
        if i is not None:
            first, last = n == 0 or rows[n - 1] != i, n == len(rows) - 1 or rows[n + 1] != i
            line = (re.sub(r'^<(\w+)', r'<\1 data-h="%d"' % i, line, count = 1) if first and last else
                    ('<g data-h="%d">' % i if first else '') + line + ('</g>' if last else ''))
        out.append(line)
    return svg[:start] + '\n'.join(out) + svg[end:], table


def _hover_style(pal, dark = None):
    """An interactive SVG's rules, scoped to the drawing's id: how the rest of a panel fades, with the soft shadow, and
    the hover box in the page's colors, those of `dark` in a dark theme, and nothing of either in print."""
    scope = '#' + THEME_PREFIX
    family = ', '.join("'%s'" % f for f in dict.fromkeys(([pal.font[0]] if pal.font else []) + [
        'Century Gothic', 'CenturyGothic', 'Comfortaa'])) + ', sans-serif'
    box = lambda p: ('{0} .vc-tip rect{{fill:{1};stroke:{2}}}{0} .vc-tip text{{fill:{3}}}'
                     '{0} .vc-tip .vc-head{{fill:{4}}}'.format(scope, p.css('paper'), p.css('faint'), p.css('muted'),
                                                               p.css('ink')))
    # A lit disc or line over marks drawn in groups, whose class sits on the element, or within it where a theme's class
    # was there first
    hit = '{0} .vc-hit.vc-on,{0} .vc-hit.vc-on *,{0} .vc-on .vc-hit'.format(scope)
    rules = ('{0} [data-h],{0} .vc-shadow{{transition:opacity .15s}}{0} .vc-off,{0}.vc-lit .vc-shadow{{opacity:{1}}}'
             '{3}{{fill-opacity:1;stroke-opacity:1}}{0} .vc-tip text{{font-family:{2}}}'
             '{0} .vc-tip .vc-head{{font-weight:bold}}'.format(scope, HOVER_FADE, family, hit))
    night = '@media (prefers-color-scheme:dark){%s}' % box(dark) if dark else ''
    paper = ('@media print{{{0} .vc-off,{0}.vc-lit .vc-shadow{{opacity:1}}{1}{{fill-opacity:0;stroke-opacity:0}}'
             '{0} .vc-tip{{display:none}}}}'.format(scope, hit))
    return '<style>%s%s%s%s</style>\n' % (rules, box(pal), night, paper)


def _frozen(v):
    return tuple(_frozen(u) for u in v) if isinstance(v, (list, tuple)) else v


def _shape_key(op):
    """What an op draws, apart from its paint, which is all that differs between a figure and its twin in another
    palette as long as they draw the same shapes; an image's pixels are its shape, and a gradient's stops its paint."""
    key = tuple((k, _frozen(v)) for k, v in sorted(op.items())
                if k not in THEME_PAINT + TAGS + ('ttf', 'label', 'mark'))
    return key + (op['fill_grad'][:2] if op.get('fill_grad') else None, )


def _paints(op):
    """An op's fill and stroke as svgout.emit writes them, with their opacities: a color, or a gradient's key."""
    fill = op['kind'] == 'text' or op['kind'] == 'path' and op['fill']
    stroke = op['kind'] == 'path' and op['stroke']
    return ((op['fill_grad'] or _css(op['fill_rgb']), op['fill_alpha']) if fill else None,
            (_css(op['stroke_rgb']), op['stroke_alpha']) if stroke else None)


def _gradient_def(key, name):
    """A gradient as svgout.emit writes one, under its own name."""
    kind, geo, stops = key
    inner = ''.join('<stop offset="%s" stop-color="%s" stop-opacity="%s" />' % (pdfwrite._fmt(o), _css(c[:3]),
                                                                                 pdfwrite._fmt(c[3])) for o, c in stops)
    tag, axes = ('radialGradient', ('cx', 'cy', 'r')) if kind == 'radial' else ('linearGradient', ('x1', 'y1', 'x2',
                                                                                                   'y2'))
    return '<%s id="%s" gradientUnits="userSpaceOnUse" %s>%s</%s>' % (
        tag, name, ' '.join('%s="%s"' % (a, pdfwrite._fmt(v)) for a, v in zip(axes, geo)), inner, tag)


class _Themes:
    """How a page in its light palette turns into its dark version: each op of one paired with the op of the other
    that draws the same shape, a class for each pair whose paint differs, naming the dark paint as CSS custom
    properties, and the ops only one of them draws, shown only in its theme."""
    def __init__(self, light, dark):
        ids, own = {}, lambda ops: [ids.setdefault(_shape_key(o), len(ids)) for o in ops]
        a, b = own(light), own(dark)
        self.ops, self.colors, self.rules, self.grads = [], {}, {}, {}
        if a == b:
            blocks = [('equal', 0, len(a), 0, len(b))]
        else:
            blocks = difflib.SequenceMatcher(None, a, b, autojunk = False).get_opcodes()
        for tag, i0, i1, j0, j1 in blocks:
            if tag == 'equal':
                self.ops += [dict(o, theme = self.pair(o, p)) for o, p in zip(light[i0:i1], dark[j0:j1])]
                continue
            # A shape of one theme alone, such as a sticker's cut on the dark page or a shadow, which glows there
            self.ops += [dict(o, theme = THEME_LIGHT) for o in light[i0:i1]]
            self.ops += [dict(o, theme = THEME_DARK) for o in dark[j0:j1]]

    def pair(self, light, dark):
        """The class that turns `light`'s paint into `dark`'s, or None where they are the same."""
        was, now = _paints(light), _paints(dark)
        if was == now:
            return None
        rule = []
        for prop, old, new in zip(('fill', 'stroke'), was, now):
            if new is None or old == new:
                continue
            if isinstance(new[0], tuple):
                grad = self.grads.setdefault(new[0], '%sg%d' % (THEME_PREFIX, len(self.grads)))
                rule.append('%s:url(#%s)' % (prop, grad))
            else:
                rule.append('%s:var(%s)' % (prop, self.colors.setdefault(new[0], '--c%d' % len(self.colors))))
            if new[1] != old[1]:
                rule.append('%s-opacity:%s' % (prop, pdfwrite._fmt(new[1])))
        return self.rules.setdefault(';'.join(rule), 'k%d' % len(self.rules)) if rule else None

    def style(self):
        """The rules, scoped to the drawing's own id, as a page may hold several drawings inline."""
        scope = '#' + THEME_PREFIX
        props = ';'.join('%s:%s' % (name, color) for color, name in self.colors.items())
        body = ''.join('%s .%s{%s}' % (scope, cls, rule) for rule, cls in self.rules.items())
        return ('<style>@media (prefers-color-scheme:dark){%s{%s}%s%s .%s{display:none}%s .%s{display:inline}}'
                '</style>\n' % (scope, props, body, scope, THEME_LIGHT, scope, THEME_DARK))

    def defs(self):
        return ''.join(_gradient_def(key, name) for key, name in self.grads.items())


def _classed(svg, ops):
    """Each drawn op's line of svgout.emit's output with its theme's class, and an op of the dark theme alone hidden
    unless the dark rule shows it, so that a reader without CSS sees the light page whole."""
    start = svg.index('\n', svg.index('<g transform="matrix(')) + 1
    end = svg.rindex('\n</g>\n</svg>')
    lines, drawn = svg[start:end].split('\n'), _drawn(ops)
    if len(lines) != len(drawn):
        raise RuntimeError(f'An SVG in both themes needs a line per drawn op, got {len(lines)} for {len(drawn)}')
    out = []
    for line, op in zip(lines, drawn):
        cls = op.get('theme')
        if cls:
            at = line.index(' />')
            line = line[:at] + ' class="%s"%s' % (cls, ' display="none"' if cls == THEME_DARK else '') + line[at:]
        out.append(line)
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


def _encode(v, ctx):
    """An argument of a plot call as JSON: numbers, text, lists, and dicts with text keys as they are, everything else
    as a dict tagged by a key that starts with '$'. Arrays of numbers go into ctx's table once each, by content. Raises
    TypeError for what has no such form."""
    enc = lambda u: _encode(u, ctx)
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        return float(v) if math.isfinite(v) else {'$f': repr(float(v))}
    if isinstance(v, Axes):
        return {'$ax': ctx['axes'][id(v)]}
    if isinstance(v, tuple):
        return {'$t': [enc(u) for u in v]}
    if isinstance(v, list):
        return [enc(u) for u in v]
    if isinstance(v, range):
        return {'$range': [v.start, v.stop, v.step]}
    # A set, such as what a highlight names, in an order that stays from one process to the next, as its own does not
    if isinstance(v, (set, frozenset)):
        return {'$set': [enc(u) for u in sorted(v, key = repr)]}
    if isinstance(v, dict):
        if all(isinstance(k, str) and k[:1] != '$' for k in v):
            return {k: enc(u) for k, u in v.items()}
        return {'$d': [[enc(k), enc(u)] for k, u in v.items()]}
    if isinstance(v, Palette):
        return {'$pal': v.name}
    kind = type(v).__name__
    if type(v).__module__.startswith('pandas') and kind in ('Series', 'DataFrame', 'Index', 'RangeIndex',
                                                              'Categorical'):
        # A default index, 0 to n - 1, goes without saying
        plain = lambda index: (type(index).__name__ == 'RangeIndex' and index.start == 0 and index.step == 1)
        if kind == 'DataFrame':
            out = {'$frame': [[enc(c), enc(v[c].reset_index(drop = True))] for c in v.columns]}
            return out if plain(v.index) else {**out, 'index': enc(v.index)}
        if kind in ('Index', 'RangeIndex'):
            return {'$index': enc(np.asarray(v)), 'name': enc(v.name)}
        cat = v if kind == 'Categorical' else v.cat if str(v.dtype) == 'category' else None
        out = {'$series' if kind == 'Series' else '$categorical': enc(np.asarray(v, dtype = object) if cat is not None
                                                                     else v.to_numpy())}
        if cat is not None:
            out.update(categories = enc(list(cat.categories)), ordered = bool(cat.ordered))
        if kind == 'Series':
            out['name'] = enc(v.name)
            if not plain(v.index):
                out['index'] = enc(v.index)
        return out
    if isinstance(v, np.ndarray):
        if v.dtype.kind in 'OUS':
            return {'$o': [enc(u) for u in v.ravel().tolist()], 'dtype': v.dtype.str, 'shape': list(v.shape)}
        if v.dtype.kind not in 'biuf':
            raise TypeError(f'it holds an array of {v.dtype}')
        key = (v.dtype.str, v.shape, v.tobytes())
        if key not in ctx['seen']:
            ctx['seen'][key] = len(ctx['arrays'])
            entry = {'dtype': v.dtype.str, 'shape': list(v.shape)}
            if v.size <= SOURCE_JSON:
                values = v.ravel().tolist()
                if v.dtype.kind == 'f' and not np.isfinite(v).all():
                    values = [None if u != u else repr(u) if u in (math.inf, -math.inf) else u for u in values]
                entry['values'] = values
            else:
                # The smallest type that holds every value exactly, its bytes grouped by significance, which deflates
                # floats far better than bytes in their order
                store = v.dtype
                if v.dtype.kind == 'b':
                    store = np.dtype('u1')
                elif v.dtype.kind in 'iu':
                    lo, hi = int(v.min()), int(v.max())
                    store = next((np.dtype(t) for t in ('i1', 'i2', 'i4', 'i8') if np.iinfo(t).min <= lo and
                                  hi <= np.iinfo(t).max), v.dtype)
                elif v.dtype.itemsize == 8:
                    with np.errstate(over = 'ignore'):
                        store = np.dtype('f4') if np.array_equal(v.astype('f4'), v, equal_nan = True) else v.dtype
                store = store.newbyteorder('<')
                raw = np.frombuffer(v.astype(store).tobytes(), np.uint8).reshape(-1, store.itemsize).T.tobytes()
                entry.update(store = store.str, offset = sum(map(len, ctx['tail'])), length = len(raw))
                ctx['tail'].append(raw)
            ctx['arrays'].append(entry)
        return {'$a': ctx['seen'][key]}
    if callable(v):
        return {'$fn': f"{getattr(v, '__module__', None)}.{getattr(v, '__qualname__', kind)}"}
    raise TypeError(f'it cannot write a {type(v).__module__}.{kind}')


def _decode(v, arrays, axes, functions):
    """What _encode wrote, back as it was: arrays as their table holds them, and a function by its name in
    `functions`, as a file names a function but never holds its code."""
    dec = lambda u: _decode(u, arrays, axes, functions)
    if isinstance(v, list):
        return [dec(u) for u in v]
    if not isinstance(v, dict):
        return v
    if '$a' in v:
        return arrays[v['$a']]
    if '$t' in v:
        return tuple(dec(u) for u in v['$t'])
    if '$d' in v:
        # A key that was a tuple comes back as one, as JSON has only lists
        return {dec(k): dec(u) for k, u in v['$d']}
    if '$f' in v:
        return float(v['$f'])
    if '$range' in v:
        return range(*v['$range'])
    if '$set' in v:
        return {dec(u) for u in v['$set']}
    if '$ax' in v:
        return axes[v['$ax']]
    if '$pal' in v:
        return v['$pal']
    if '$o' in v:
        values = [dec(u) for u in v['$o']]
        if np.dtype(v['dtype']).kind != 'O':
            return np.array(values, dtype = v['dtype']).reshape(v['shape'])
        # Element by element, as np.array would turn a list of lists into more dimensions
        out = np.empty(len(values), dtype = object)
        out[:] = values
        return out.reshape(v['shape'])
    if '$fn' in v:
        name = v['$fn']
        for key in (name, name.split('.')[-1]):
            if key in (functions or {}):
                return functions[key]
        raise ValueError(f'The figure was drawn with the function {name}, which a file holds only by name; pass it '
                         f"as open(..., functions = {{'{name.split('.')[-1]}': ...}})")
    if any(k in v for k in ('$series', '$frame', '$index', '$categorical')):
        pd = importlib.import_module('pandas')
        index = dec(v['index']) if 'index' in v else None
        if '$frame' in v:
            frame = pd.DataFrame({dec(c): dec(s) for c, s in v['$frame']})
            return frame if index is None else frame.set_axis(index)
        if '$index' in v:
            return pd.Index(dec(v['$index']), name = dec(v['name']))
        values = dec(v.get('$series', v.get('$categorical')))
        if 'categories' in v:
            values = pd.Categorical(values, categories = dec(v['categories']), ordered = v['ordered'])
        return values if '$categorical' in v else pd.Series(values, index = index, name = dec(v['name']))
    return {k: dec(u) for k, u in v.items()}


def _carried(made, force, size):
    """The source a file takes, as (JSON with its tail, that deflated), from the future `made`, and why it stays out
    if it does: it cannot be written, or, unless `force`, it is past SOURCE_FLOOR and larger than the file without it,
    whose size `size` gives when asked."""
    if made is None:
        return None, None
    try:
        raw, packed = made.result()
    except TypeError as e:
        if force:
            raise
        return None, f"the figure's source stays out of the file, as {e}"
    if not force and len(packed) > SOURCE_FLOOR and len(packed) > size():
        return None, (f"the figure's source ({len(packed) / 1024:.0f} kB) is larger than the file and stays out of it; "
                      'source = True keeps it')
    return (raw, packed), None


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
        # Set by a window while a zoom, pan, or resize runs, so that spread glyphs move with their points rather than
        # being placed again at every step (see Axes._place_glyphs); files never draw in draft
        self._draft = False
        self._texts = []
        # Every text and mark meets APCA's contrast and every color of a plot stays apart for every reader, by shades
        # adjusted where they fail, which `adjusted` lists as (was, is, why)
        self.legible, self.adjusted = True, []
        # Glyphs take the room of a dark page's sticker cut on a light page too, as in an SVG of both themes
        self._stickers = False

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
        fig._texts, fig.legible = list(self._texts), self.legible
        twins, cats = {id(ax): Axes(fig, fig.pal) for ax in self.axes}, {}
        for ax in self.axes:
            twin = twins[id(ax)]
            twin._cell, twin._span, twin._fixed, twin._outer = ax._cell, ax._span, ax._fixed, dict(ax._outer)
            twin._share = {axis: [twins[id(other)] for other in group] for axis, group in ax._share.items()}
            # Categories shared as the axes share them, to fill again as the calls run
            twin._cats = {axis: cats.setdefault(id(found), {}) for axis, found in ax._cats.items()}
            twin._align = {axis: twins[id(other)] for axis, other in ax._align.items()}
            fig.axes.append(twin)
        swap = lambda v: twins[id(v)] if isinstance(v, Axes) else v
        # A color the caller took from this palette, such as fig.pal.groups(3, 2), is that shade of the other one, as
        # it would have been had the figure been drawn there; the page's color stays, as white text means white
        was = next((asked for asked, made, _ in _legibles.values() if made is self.pal), self.pal)
        own = {was.color(n): n for n in ('ink', 'muted', 'faint', *(h + str(i) for h in 'ba' for i in range(4)))}
        def shade(v):
            if not (isinstance(v, str) and v.startswith('#') or isinstance(v, tuple) and len(v) in (3, 4)):
                return v
            try:
                return own.get(was.color(v), v)
            except (ValueError, TypeError):
                return v
        def tinted(k, v):
            if k == 'edge' and isinstance(v, tuple) and len(v) == 2:
                return shade(v[0]), v[1]
            if k in ('color', 'colors'):
                return ({u: shade(w) for u, w in v.items()} if isinstance(v, dict) else [shade(w) for w in v]
                        if isinstance(v, list) else shade(v))
            if k == 'emphasis' and isinstance(v, dict):
                return {u: shade(w) for u, w in v.items()}
            return swap(v)
        for ax, name, args, kwargs in self._calls:
            getattr(twins[id(ax)], name)(*map(swap, args), **{k: tinted(k, v) for k, v in kwargs.items()})
        if hasattr(self, 'row_order'):
            fig.row_order, fig.col_order = self.row_order, self.col_order
        return fig

    def _source(self):
        """The calls and data that drew the figure, as JSON followed by the bytes of large arrays, and all of that
        deflated: what in_palette copies, with each Axes as its place in `axes` and the palette by name where it is a
        shipped one or its dark version, else by its settings. Raises TypeError for an argument it cannot write."""
        ctx = {'axes': {id(ax): i for i, ax in enumerate(self.axes)}, 'arrays': [], 'seen': {}, 'tail': []}
        enc = lambda v: _encode(v, ctx)
        calls = []
        for ax, name, args, kwargs in self._calls:
            # A table in long form keeps only the columns the call names, as a DataFrame may hold many more
            named = [kwargs[k] for k in ('x', 'y', 'hue', 'names') if isinstance(kwargs.get(k), str)]
            data = kwargs.get('data', args[0] if args else None)
            if named and name in ('boxplot', 'stripplot', 'violinplot') and hasattr(data, 'columns') and all(
                    k in data.columns for k in named):
                data = data[list(dict.fromkeys(named))]
                args, kwargs = ((data, *args[1:]), kwargs) if 'data' not in kwargs else (args, {**kwargs, 'data': data})
            calls.append([ctx['axes'][id(ax)], name, enc(list(args)), enc(kwargs)])
        # The palette as asked for, before Figure._legible fixed its text colors, which it does again on reopening
        pal = next((was for was, made, _ in _legibles.values() if made is self.pal), self.pal)
        shipped = {p.name: p for p in (LUMEN, COUTURE, POSTER, AURA)}
        base = shipped.get(pal.name[:-5]) if pal.name.endswith('-dark') else None
        palette = (pal.name if shipped.get(pal.name) == pal else {'dark': base.name, 'paper': pal.paper} if base and
                   base.dark(pal.paper) == pal else {'settings': {f.name: enc(getattr(pal, f.name)) for f in
                                                                   fields(Palette) if f.init}})
        axes = [{'cell': enc(ax._cell), 'span': enc(ax._span), 'fixed': enc(ax._fixed), 'outer': enc(ax._outer),
                 'share': {k: [ctx['axes'][id(other)] for other in group] for k, group in ax._share.items()},
                 'cats': {k: next(i for i, other in enumerate(self.axes) if other._cats.get(k) is found)
                          for k, found in ax._cats.items()},
                 'align': {k: ctx['axes'][id(other)] for k, other in ax._align.items()}} for ax in self.axes]
        head = {'viscose': getattr(sys.modules.get('viscose'), '__version__', None), 'format': SOURCE_FORMAT,
                'about': SOURCE_ABOUT, 'palette': palette,
                'figure': {'size': enc(self._base), 'title': self._title, 'subtitle': self._subtitle,
                           'grid': enc(self._grid), 'held': self._held, 'texts': enc(self._texts),
                           'legible': bool(self.legible)},
                'axes': axes, 'calls': calls}
        if hasattr(self, 'row_order'):
            head['figure']['order'] = enc([list(map(int, self.row_order)), list(map(int, self.col_order))])
        head['arrays'] = ctx['arrays']
        # JSON escapes every line break inside it, so the first one ends it
        raw = json.dumps(head, ensure_ascii = False, allow_nan = False, separators = (',', ':')).encode()
        raw += b'\n' + b''.join(ctx['tail']) if ctx['tail'] else b''
        # Level 1 packs a megabyte of floats 1% less tightly in half the time, and level 9 gains nothing over 6
        return raw, zlib.compress(raw, 6 if len(raw) < 1 << 20 else 1)

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
        # Brackets stretch an axis shared by several, so all start over before any sets its limits
        for ax in self.axes:
            ax._lift = 1.0
        for ax in self.axes:
            ax._limits()
        # A first pass places the axes so that labels can be fitted to them, and later ones place them around the
        # result, with the y limits that the brackets need in their height
        for _ in range(5):
            self._place()
            lifted = [ax._fit_brackets() for ax in self.axes]
            if not any([ax._fit_labels() for ax in self.axes]) and not any(lifted):
                break
        if self.legible:
            self._legible()

    def _legible(self):
        """Every text and mark meets APCA's contrast for what it is against the page (LC_TEXT for text that is read,
        LC_LARGE for large or heavy text, LC_MARK for marks that carry data), and colors of different hues that tell an
        axes' data apart stay CVD_MIN apart for every reader, as normal vision and protanopia, deuteranopia, and
        tritanopia see them; a hue's own shades differ in lightness, which all readers see. A color that fails takes the
        nearest of its hue's shades that passes, by lightness, or else its own hue at most 0.2 lighter or darker with
        most of its chroma, in every axes alike; colors from a scale, whose shades are data, and glyphs, which keep
        their own, stay as they are. A color that passes is never touched, so a figure checks out the same each time it
        is laid out, and a pale shade set on purpose stays as long as it can be seen."""
        pal = self.pal
        page = pal.color('paper')
        # The palette's own text, of which the axis lines and guides keep the colors they had
        if id(pal) not in _legibles:
            texts = {'tick_color': (pal.tick_color or pal.muted, LC_TEXT), 'label_color': (pal.label_color or pal.ink,
                     LC_TEXT), 'subtitle_color': (pal.subtitle_color or pal.muted, LC_TEXT), 'ink': (pal.ink, LC_TEXT),
                     'title_color': (pal.title_color or pal.ink, LC_LARGE if pal.title_size >= 24 else LC_TEXT)}
            fixed = {k: _css(legible(pal.color(c), page, target)) for k, (c, target) in texts.items()
                     if apca(pal.color(c), page) < target}
            _legibles[id(pal)] = (pal, replace(pal, **fixed) if fixed else pal,
                                  [(_css(pal.color(texts[k][0])), v, 'contrast') for k, v in fixed.items()])
        _, self.pal, notes = _legibles[id(pal)]
        self.adjusted += [n for n in notes if n not in self.adjusted]
        for ax in self.axes:
            ax.pal = self.pal
        swap = {}
        hues = [[_hex(s) for s in hue] for hue in (pal.a, pal.b)]
        # Which of the palette's hues a color is a shade of, if any; a hue's shades differ in lightness, which every
        # reader sees, and are ordered so on purpose
        hue = lambda rgb: next((i for i, h in enumerate(hues) if any(np.allclose(s, rgb, atol = 1e-3) for s in h)),
                               None)
        origin = {}
        family = lambda rgb: hue(origin.get(rgb, rgb))
        def best(rgb, target, others, taken, kin = ()):
            # The least change in OKLab from the color as drawn that reaches the target, stays apart from the other
            # hues' colors, keeps clear of the colors in `taken`, and keeps its place by lightness among the shades in
            # `kin`, as two categories must never merge or trade places: its own hue at most 0.2 lighter or darker
            # with most of its chroma, which keeps the palette's effects through Palette.shades, or another shade of
            # its hue where that is nearer, so that a figure keeps the look it was drawn with
            L, C, h = _oklch(rgb)
            own = [] if hue(rgb) is None else [s for s in hues[hue(rgb)] if not np.allclose(s, rgb, atol = 1e-3)]
            steps = [_hex(_from_oklch(L + k * 0.02, C, h)) for k in range(-10, 11) if k and 0 <= L + k * 0.02 <= 1]
            near = lambda v: float(np.linalg.norm(_oklab(v) - _oklab(rgb)))
            ranked = sorted([(near(v), v) for v in own] + [(near(v), v) for v in steps if _oklch(v)[1] >= 0.6 * C])
            for d, cand in ranked:
                lc = _oklch(cand)[0]
                if (apca(cand, page) >= target and all(apart(cand, o) >= CVD_MIN for o in others)
                        and all(np.linalg.norm(_oklab(cand) - _oklab(t)) >= 0.03 for t in taken)
                        and all((lc - _oklch(t)[0]) * (L - _oklch(t)[0]) > 0 for t in kin)):
                    return d, tuple(float(v) for v in cand)
            return None
        def move(rgb, cand, why):
            swap[rgb] = cand
            origin[cand] = rgb
            # Colors a rounding apart move alike, and are listed once
            if (_css(rgb), _css(cand), why) not in self.adjusted:
                self.adjusted.append((_css(rgb), _css(cand), why))
        for ax in self.axes:
            # Each color an axes shows, with what it must reach, and those that tell its data apart, in order
            need, kinds = {}, []
            for a in ax.artists:
                # Labels that name a series take its color, made legible as they are drawn
                if a['kind'] == 'text' and a.get('direct'):
                    continue
                if a['kind'] == 'text' and not a.get('glyph'):
                    size, bold = a['size'] or ax.pal.tick_size, pal.text_bold if a['bold'] is None else a['bold']
                    found, target = [a['rgb']], LC_LARGE if size >= 24 or bold and size >= 16 else LC_TEXT
                elif a['kind'] in LEVELS and a.get('alpha', 1.0) >= 1.0 and not a.get('mapped'):
                    # A mark with a rim is seen by whichever of its fill and rim stands out more against the page, as a
                    # ring's white fill or a dot's white rim must not count, nor be moved for everything of their color
                    found = a['rgb'] if isinstance(a['rgb'], list) else [
                        max((v for v in (a['rgb'], (a.get('edge') or (None, ))[0]) if v is not None),
                            key = lambda v: abs(apca(v, page)), default = None)]
                    target = LC_MARK
                    kinds += [tuple(float(v) for v in rgb) for rgb in found if rgb is not None]
                else:
                    continue
                for rgb in found:
                    if rgb is not None:
                        key = tuple(float(v) for v in rgb)
                        need[key] = max(need.get(key, 0.0), target)
            # A mark keeps clear of the other marks; text may share a color with them
            for rgb, target in need.items():
                found = best(rgb, target, [], [swap.get(o, o) for o in kinds if o != rgb] if rgb in kinds else []) if (
                    rgb not in swap and apca(rgb, page) < target) else None
                if found:
                    move(rgb, found[1], 'contrast')
            kinds = list(dict.fromkeys(swap.get(rgb, rgb) for rgb in kinds))
            apart_hues = lambda i, k: family(kinds[i]) is None or family(kinds[i]) != family(kinds[k])
            # Only colors of different hues can run together for a reader; of a pair that does, whichever needs the
            # smaller change moves
            for j in range(1, len(kinds)):
                for _ in range(len(kinds)):
                    clash = [i for i in range(j) if apart_hues(i, j) and apart(kinds[i], kinds[j]) < CVD_MIN]
                    if not clash:
                        break
                    options = []
                    for k in [j] + clash:
                        was = origin.get(kinds[k], kinds[k])
                        found = best(was, need.get(was, LC_MARK), [kinds[i] for i in range(len(kinds))
                                                                   if i != k and apart_hues(i, k)],
                                     kinds[:k] + kinds[k + 1:], [kinds[i] for i in range(len(kinds))
                                                                 if i != k and not apart_hues(i, k)])
                        if found:
                            options.append((found[0], k, was, found[1]))
                    if not options:
                        break
                    _, k, was, cand = min(options)
                    move(was, cand, 'color vision')
                    kinds[k] = cand
        if not swap:
            return
        recolor = lambda rgb: swap.get(tuple(float(v) for v in rgb), rgb) if rgb is not None else None
        for ax in self.axes:
            for a in ax.artists:
                if isinstance(a.get('rgb'), (tuple, list)) and not a.get('mapped'):
                    a['rgb'] = [recolor(v) for v in a['rgb']] if isinstance(a['rgb'], list) else recolor(a['rgb'])
                if a.get('edge'):
                    a['edge'] = (recolor(a['edge'][0]), a['edge'][1])
            ax.legend_items = [(s, recolor(rgb), *rest) for s, rgb, *rest in ax.legend_items]

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
                    # Labels that must all show get the room they need, however much that is, and glyphs as much as
                    # the page may grow, before any is left out
                    glyphs = [bool(ax._tick_opts[axis].get('glyphs')) for axis in 'xy']
                    if ax.xrot and (ax.show_all[0] or glyphs[0]):
                        need = (ax._tick_need('x') + l + r) / share_w + 2 * pad + gap * (ncols - 1)
                        self.width = max(self.width, need if ax.show_all[0] else min(need, self._base[0] * 3))
                    if ax.show_all[1] or glyphs[1]:
                        need = (ax._tick_need('y') + b + t) / share_h + pad + head + gap * (nrows - 1)
                        self.height = max(self.height, need if ax.show_all[1] else min(need, self._base[1] * 3))
            total_w = self.width - 2 * pad - gap * (ncols - 1)
            total_h = self.height - pad - head - gap * (nrows - 1)
            cw, ch = np.asarray(wr, float) / sum(wr) * total_w, np.asarray(hr, float) / sum(hr) * total_h
            # Axes that share an axis line up: each column takes its widest margins, each row its tallest, and the plot
            # areas divide what is left by the ratios
            aligned = any(len(group) > 1 for ax in self.axes for group in ax._share.values())
            if aligned:
                sides, levels = np.zeros((ncols, 2)), np.zeros((nrows, 2))
                for ax in self.axes:
                    if ax._fixed is None:
                        l, r, b, t = ax._margins()
                        (row, col) = ax._cell
                        sides[col], levels[row] = np.maximum(sides[col], (l, r)), np.maximum(levels[row], (b, t))
                cw = np.asarray(wr, float) / sum(wr) * (total_w - sides.sum()) + sides.sum(axis = 1)
                ch = np.asarray(hr, float) / sum(hr) * (total_h - levels.sum()) + levels.sum(axis = 1)
            self.width += extra.sum()
        for ax in self.axes:
            l, r, b, t = ax._margins()
            if self._grid is not None and aligned and ax._fixed is None:
                (l, r), (b, t) = sides[ax._cell[1]], levels[ax._cell[0]]
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
    def _canvas(self, stamp_dpi = None, marker_forms = False, layout = True, shapes = False, hover = False):
        """Everything drawn onto a fresh display list; `stamp_dpi` names a raster output, where many discs become an
        image, `marker_forms` a PDF, where markers of one size can be one stored shape, `shapes` PowerPoint, where each
        op notes what drew it, which names its shape, and `hover` an interactive SVG, where each mark also notes its
        names and hover text."""
        if layout:
            self._layout()
        c = pdfwrite.Canvas(BytesIO(), pagesize = (self.width, self.height))
        c.stamp_dpi = stamp_dpi
        c.marker_forms = marker_forms
        c.shapes = shapes or hover
        c.hover = hover
        pal = self.pal
        for i, ax in enumerate(self.axes):
            n = len(c.ops)
            ax._draw_axes(c)
            marks.tag(c, n, scope = ('a%d' % i, 'frame'))
        for i, ax in enumerate(self.axes):
            n = len(c.ops)
            ax._draw_marks(c)
            marks.tag(c, n, scope = ('a%d' % i, 'marks'))
        if self._title:
            y = self.height - 38.0
            n = len(c.ops)
            text.draw(c, 30.0, y, self._title, pal.title_size, pal.color(pal.title_color or pal.ink), pal.title_bold)
            marks.tag(c, n, name = 'title')
            if self._subtitle:
                caps = pal.extra.get('subtitle_caps')
                y -= 1.25 * pal.title_size * self._title.count('\n')
                n = len(c.ops)
                text.draw(c, 32.0, y - 20.0, self._subtitle.upper() if caps else self._subtitle, pal.subtitle_size,
                          pal.color(pal.subtitle_color or pal.muted), pal.subtitle_bold,
                          tracking = 0.9 if caps else 0.0)
                marks.tag(c, n, name = 'subtitle')
        for i, ax in enumerate(self.axes):
            n = len(c.ops)
            ax._draw_text(c)
            marks.tag(c, n, scope = ('a%d' % i, 'text'))
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

    def _shadow(self, dpi, layout = True):
        """The palette's soft shadow for an output at `dpi`: its straight-alpha RGBA at no more than SHADOW_DPI, the
        (top, bottom, left, right) box on the output's own pixel grid where some alpha survives rounding, beyond which
        it can be left out, the output's rows and columns, and the page's drawing state. None for a palette without one
        or a page without marks."""
        if not self.pal.shadow:
            return None
        res = min(dpi, SHADOW_DPI)
        c = self._canvas(stamp_dpi = res, layout = layout)
        marks = [op for op in c.ops if op.get('mark')]
        if not marks:
            return None
        arr = _shadow_layer(marks, self.width, self.height, res, self.pal.shadow)
        seen = arr[..., 3] * 255.0 + 0.5 >= 1.0
        ys, xs = np.flatnonzero(seen.any(axis = 1)), np.flatnonzero(seen.any(axis = 0))
        if not len(ys):
            return None
        sc = dpi / 72.0
        h, w = int(round(self.height * sc)), int(round(self.width * sc))
        fy, fx = h / arr.shape[0], w / arr.shape[1]
        y0, y1 = max(0, int((ys[0] - 0.5) * fy - 0.5)), min(h, int((ys[-1] + 1.5) * fy + 0.5) + 1)
        x0, x1 = max(0, int((xs[0] - 0.5) * fx - 0.5)), min(w, int((xs[-1] + 1.5) * fx + 0.5) + 1)
        return arr, (y0, y1, x0, x1), h, w, c.state

    def _shadow_op(self, dpi, layout = True):
        """The palette's soft shadow as an image op at no more than SHADOW_DPI to go under everything else, for PDF and
        SVG, or None."""
        found = self._shadow(min(dpi, SHADOW_DPI), layout)
        if found is None:
            return None
        arr, (y0, y1, x0, x1), h, w, state = found
        px = [(arr[y0:y1, x0:x1, k] * 255.0 + 0.5).astype(np.uint8) for k in range(4)]
        sc = min(dpi, SHADOW_DPI) / 72.0
        return dict(state, kind = 'image', w = x1 - x0, h = y1 - y0, rgb = np.dstack(px[:3]).tobytes(),
                    alpha = px[3].tobytes(), ctm = ((x1 - x0) / sc, 0.0, 0.0, (y1 - y0) / sc, x0 / sc, (h - y1) / sc))

    def _shadow_under(self, dpi):
        """The palette's soft shadow for a raster output, as the premultiplied layer that glycorender lays under the
        page: (top, left, planes) on the output's own pixels, scaled up from SHADOW_DPI where finer, or None."""
        found = self._shadow(dpi, False)
        if found is None:
            return None
        arr, (y0, y1, x0, x1), h, w, _ = found
        out = np.empty((4, y1 - y0, x1 - x0), np.float32)
        # In 8 bits, as an image would carry it, then premultiplied as glycorender would composite that image: a soft
        # gradient then keeps runs of equal values, which pack far smaller than values rounded pixel by pixel. A plane
        # and a band of rows per thread, as the whole page is millions of pixels each
        step = max(1, -(-(y1 - y0) // 3))
        def plane(job):
            k, r0 = job
            r1 = min(y1, r0 + step)
            v = arr[r0:r1, x0:x1, k] if dpi <= SHADOW_DPI else marks._upscale(arr[..., k], h, w, (r0, r1, x0, x1))
            v = np.multiply(v, 255.0, dtype = np.float32)
            v += 0.5
            np.take(BYTES, v.astype(np.uint8), out = out[k, r0 - y0:r1 - y0])
        def premultiply(job):
            k, r0 = job
            out[k, r0 - y0:r0 - y0 + step] *= out[3, r0 - y0:r0 - y0 + step]
        with ThreadPoolExecutor(12) as pool:
            list(pool.map(plane, [(k, r0) for k in range(4) for r0 in range(y0, y1, step)]))
            list(pool.map(premultiply, [(k, r0) for k in range(3) for r0 in range(y0, y1, step)]))
        return y0, x0, out

    def _raster(self, dpi, background):
        """The page's pixels at `dpi`, with its soft shadow computed on a thread beside the drawing and laid under it;
        they share nothing once laid out, and numpy releases the GIL for most of its work."""
        self._layout()
        with ThreadPoolExecutor(1) as pool:
            shadow = pool.submit(self._shadow_under, dpi)
            c = self._canvas(stamp_dpi = dpi, layout = False)
            return raster.render(c.ops, self.width, self.height, dpi / 72.0, dpi / 72.0, self.pal.color(background),
                                 under = shadow.result)

    def to_png(self, dpi = 300.0, background = 'paper', press = None):
        """PNG bytes; background is any color, by default the palette's page, and None keeps the page transparent.
        With a `press` (see to_pdf), a soft proof: the page as that press prints it, with its paper as white."""
        page = self._raster(dpi, background)
        if press:
            # Each color once, as a page holds few beside its many pixels
            flat = page[..., :3].reshape(-1, 3)
            found, back = np.unique(flat[:, 0].astype(np.int32) << 16 | flat[:, 1].astype(np.int32) << 8 | flat[:, 2],
                                    return_inverse = True)
            proof = _press(press).printed(np.stack([found >> 16, found >> 8 & 255, found & 255], 1) / 255.0)
            page[..., :3] = (np.clip(proof, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[back.ravel()].reshape(
                page.shape[:2] + (3,))
        return raster.encode_png(page, (), (dpi, dpi))

    def to_rgba(self, dpi = 144.0, background = 'paper'):
        """The page as an array of shape (rows, columns, 4) of 8-bit RGBA, first row at the top, for showing it in a
        window without encoding a PNG; background is any color, by default the palette's page, and None keeps the page
        transparent."""
        page = self._raster(dpi, background)
        return page if page.shape[2] == 4 else np.dstack([page, np.full(page.shape[:2], 255, np.uint8)])

    def display_list(self, layout = True):
        """The page as glycorender's display list, each op in page points with its own state, for a window that paints
        with a painter of its own, which is much faster at high resolution; a palette's soft shadow and its page's color
        are not part of it. layout = False takes the page as `layout` last settled it."""
        return self._canvas(layout = layout).ops

    def _shapes(self):
        """The display list that PowerPoint's shapes are made from, each op noting what drew it, without the soft
        shadow, which each mark casts there as PowerPoint's own."""
        return self._canvas(shapes = True).ops

    def to_pptx(self, morph = 1.5):
        """A PowerPoint file of one slide as large as the page, in native shapes: text stays text in the palette's font,
        to edit in place; the palette is the deck's theme, its ink, muted text, page, and faint guides the theme's text
        and background colors and the six shades that series take first its accents, in the order an Office chart
        gives them, so that a deck's own theme or the Design tab recolors the figure as it does a chart; and every shape
        is named after what it draws, so that Morph carries one figure into the next (see vc.deck)."""
        from .drawingml import deck
        return deck([self], morph = morph)

    def to_pdf(self, press = None):
        """PDF bytes, with marks and text as vector; only a blurred shadow and very large heatmaps are images. With a
        `press`, a PDF/X-4 for print in that press's CMYK, with its ICC profile as the output intent: True for U.S. web
        offset on coated paper (CGATS TR 001, SWOP), or a printer's or a journal's own profile as a path or a Press.
        Every color goes through Press.separate, so that one the press cannot print keeps its lightness and hue and
        gives up only the chroma it must, rather than whatever a print shop's conversion would make of it."""
        return self._pdf(press)[0]

    def _pdf(self, press = None, made = None, force = False):
        """PDF bytes as to_pdf makes them, carrying the source that the future `made` gives, if it comes (see
        _carried), as an attachment, and why it stays out if it does."""
        target = BytesIO()
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas(marker_forms = True)
        c.target = target
        if press:
            ink = _press(press)
            c.cmyk = ink.separate
            c.pdfx = dict(profile = ink.icc(), condition = ink.condition, info = ink.info, registry = ink.registry,
                          title = ''.join(t for t, _, _ in text.runs(self._title.split('\n')[0])) or 'Figure',
                          creator = 'viscose')
        if op is not None:
            c.ops.insert(0, op)
        # A PDF's page is white unless something covers it
        if self.pal.color('paper') != (1.0, 1.0, 1.0):
            marks.polygon(c, [0.0, self.width, self.width, 0.0], [0.0, 0.0, self.height, self.height],
                          self.pal.color('paper'))
            c.ops.insert(0, c.ops.pop())
        # The file without its source, written only when a large source has to be weighed against it
        bare = []
        def size():
            c.save()
            bare.append(target.getvalue())
            return len(bare[0])
        carried, why = _carried(made, force, size)
        if carried is None and bare:
            return bare[0], why
        target.seek(0)
        target.truncate()
        if carried:
            c.attachments = [dict(name = SOURCE_NAME, data = carried[0], packed = carried[1], mime = SOURCE_MIME,
                                  description = SOURCE_ABOUT, relationship = 'Source')]
        c.save()
        return target.getvalue(), why

    def _svg_page(self, hover = False):
        """The display list that to_svg writes, the shadow under everything, and apart from it the glycan labels; with
        `hover`, each op notes what drew it, for an interactive SVG."""
        op = self._shadow_op(SHADOW_DPI)
        c = self._canvas(hover = hover)
        if op is not None:
            c.ops.insert(0, dict(op, shadow = True) if hover else op)
        labels = [o for o in c.ops if o.get('label')]
        c.ops = [o for o in c.ops if not o.get('label')]
        return c, labels

    def _themes(self, themes):
        """The figure in its light palette and in that palette's dark version, on the page color `themes` names or the
        dark version's own, both drawn again by in_palette, which dark mode must leave alone while it runs, and both
        with the room of the dark page's sticker cuts around glyphs, so that they share one page."""
        pal = next((was for was, made, _ in _legibles.values() if made is self.pal), self.pal)
        paper = None if themes is True else themes
        light = next((was for was, made in _darks.values() if made is pal), None) if pal.is_dark else pal
        if light is None:
            raise ValueError(f'An SVG in both themes needs the light palette that {pal.name!r} is the dark version '
                             'of, and this one was made by hand; draw the figure in a light palette instead')
        previous = dark_mode(False)
        try:
            # A figure drawn dark keeps its own page, such as a console's surface, unless given another
            dark = pal if pal.is_dark and paper is None else light.dark(paper)
            pair = self.in_palette(light), self.in_palette(dark)
        finally:
            dark_mode(*previous)
        # One page for both, so the light one leaves the room of the dark one's sticker cuts too
        for fig in pair:
            fig._stickers = True
        return pair

    def to_svg(self, themes = False, interactive = False):
        """SVG text on the palette's page, with text as outlines and images embedded. themes = True writes the figure
        in its palette and that palette's dark version (Palette.dark) at once, as one drawing whose colors follow the
        reader's light or dark mode through CSS, on a website, on GitHub, or in docs alike; a color for `themes` is the
        dark page's, such as the site's own dark background. Whatever reads no CSS, as most editors, sees the light
        page. interactive = True adds a few lines of script and style, with no library: the pointer over a point, bar,
        box, cell, or line shows what it is and its values, and lights up everything of the same names in every panel,
        a sample by its `names` across a PCA, a heatmap's column, and its points over boxes alike, a series or a
        category by its label, and so in every other interactive figure on the same page, while the rest of each panel
        that holds one fades. It works where the SVG runs as a document or inline in a page, not as an <img>, and the
        page in print, in an editor, or in an <img> is the figure as drawn."""
        figure = self
        if themes:
            figure, dark = self._themes(themes)
            d, dark_labels = dark._svg_page(interactive)
        c, labels = figure._svg_page(interactive)
        if themes:
            theme = _Themes(c.ops, d.ops)
            c.ops = theme.ops
            sheet = lambda f: dict(c.state, kind = 'path', fill = True, stroke = False, fill_rgb = f.pal.color('paper'))
            paper = theme.pair(sheet(figure), sheet(dark))
            # Glycan labels are the same text in either theme
            flat = lambda op: dict(op, fill = False) if op['kind'] == 'path' else op
            for op, other in zip(labels, dark_labels):
                op['theme'] = theme.pair(flat(op), flat(other))
        svg, table = _classed(c.to_svg(), c.ops) if themes else c.to_svg(), None
        if interactive:
            svg, table = _hovered(svg, c.ops)
        svg = _with_ids(svg, c.ops)
        # With glycan labels, the width the page needs without them, down to which annotate_figure can crop it
        bare = ' data-width-unlabeled="%.2f"' % figure._width_unlabeled() if labels else ''
        page = '<rect width="%g" height="%g" fill="%s"%s%s />\n' % (
            figure.width, figure.height, figure.pal.css('paper'), bare, ' class="%s"' % paper if themes and paper
            else '')
        svg = svg.replace('<g transform="matrix(', page + '<g transform="matrix(', 1)
        # Tells slate this is a figure, ahead of the gradients' long <defs> and through annotate_figure. It cannot be a
        # comment, as annotate_figure takes every comment and all that follows up to the next </g> for a label
        top = svg.index('\n', svg.index('<svg')) + 1
        svg = (svg[:top] + '<metadata>viscose</metadata>\n' + (theme.style() if themes else '') +
               (_hover_style(figure.pal, dark.pal if themes else None) if table else '') + svg[top:])
        if labels:
            svg = svg.replace('</svg>', _label_groups(c, labels, figure.height) + '\n</svg>')
        if table:
            # The script goes last, after all it reads, with each text once, such as a glycan named in every cell of
            # its row, and nothing in its table that could close it or its section
            texts = {}
            word = lambda found: None if found is None else [texts.setdefault(s, len(texts)) for s in found]
            table = [[word(names), word(said), word(series), axes, whole] for names, said, series, axes, whole in table]
            rows = json.dumps([list(texts), table], ensure_ascii = False, separators = (',', ':'))
            rows = rows.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
            end = svg.rindex('</svg>')
            svg = svg[:end] + HOVER_SCRIPT.replace('__TABLE__', rows) + svg[end:]
        if not themes and not table:
            return svg
        if themes and theme.grads:
            # The dark theme's own gradients, beside the light one's, which svgout.emit wrote
            svg = (svg.replace('</defs>', theme.defs() + '</defs>', 1) if '</defs>' in svg else
                   svg.replace('<rect ', '<defs>%s</defs>\n<rect ' % theme.defs(), 1))
        svg = svg.replace('<svg ', '<svg id="%s" ' % THEME_PREFIX, 1)
        return svg.replace(THEME_PREFIX, 'vc%08x' % zlib.crc32(svg.encode()))

    def save(self, path, dpi = 300.0, background = 'paper', press = None, source = None, themes = False,
             interactive = False):
        """Write the file format that the path's suffix names: png, pdf, svg, or pptx, a slide of PowerPoint's own
        shapes whose text stays editable and which Morph carries into the next figure (see to_pptx and vc.deck). `dpi`
        and `background` are for PNG, as PDF, SVG, and PowerPoint are vector on the palette's page. `press` makes a PDF
        a PDF/X-4 in that press's CMYK, and a PNG its soft proof (see to_pdf), which SVG, a screen format, has no place
        for. A PNG, PDF, or SVG carries the figure's source, the calls and data that drew it, as JSON: vc.open brings
        the figure back from the file alone, in another palette, dark, or at a journal's column width, and anyone can
        read its numbers. A figure's source takes a few kB, and is written while the page draws. source = False leaves
        it out, as a PDF/X does, which is the page alone for print; a source past 64 kB that is larger than the rest of
        the file, such as of 100,000 points, stays out with a warning, unless source = True. themes = True makes an SVG
        follow its reader's light or dark mode, and interactive = True gives it hover text and linked highlighting
        (see to_svg)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if themes and suffix != '.svg':
            raise ValueError('Only an SVG carries both themes, as CSS; a PNG or a PDF is one page, so draw it in the '
                             "palette's dark version for a dark one")
        if interactive and suffix != '.svg':
            raise ValueError('Only an SVG runs the script that makes a figure interactive; save it as .svg')
        if press and suffix == '.svg':
            raise ValueError('SVG is for screens and has no CMYK; for print, save a PDF with press, or a PNG for its '
                             'proof')
        if press and suffix == '.pptx':
            raise ValueError('A slide is for screens and has no CMYK; for print, save a PDF with press')
        if suffix == '.pptx':
            path.write_bytes(self.to_pptx())
            return path
        if suffix not in ('.png', '.pdf', '.svg'):
            raise ValueError(f'Unsupported format {suffix!r}; use .png, .pdf, .svg, or .pptx')
        force = source is True
        with ThreadPoolExecutor(1) as pool:
            made = None if source is False or press and suffix == '.pdf' else pool.submit(self._source)
            if suffix == '.pdf':
                data, why = self._pdf(press, made, force)
            elif suffix == '.png':
                data = self.to_png(dpi, background, press)
                carried, why = _carried(made, force, lambda: len(data))
                if carried:
                    # A private chunk that editors may copy, just before the end
                    chunk = SOURCE_CHUNK + carried[1]
                    data = data[:-12] + struct.pack('>I', len(carried[1])) + chunk + struct.pack(
                        '>I', zlib.crc32(chunk)) + data[-12:]
            else:
                data = self.to_svg(themes, interactive)
                carried, why = _carried(made, force, lambda: len(data))
                if carried:
                    end = data.rindex('</svg>')
                    data = (data[:end] + '<metadata id="viscose-source">' + base64.b64encode(carried[1]).decode('ascii')
                            + '</metadata>\n' + data[end:])
                data = data.encode('utf-8')
        if why:
            warnings.warn(f'{path.name}: {why}', stacklevel = 2)
        path.write_bytes(data)
        return path

    def show(self, dpi = 144.0, viewer = True):
        """Show the figure: inline in Jupyter or slate, or in the system's image viewer from a terminal. A library that
        has just saved the figure passes viewer = False, or a script saving many figures opens a window for each."""
        show_png(lambda: self.to_png(dpi), viewer)

    def _repr_png_(self):
        return self.to_png(144.0)


def subplots(nrows = 1, ncols = 1, figsize = (8.0, 8.0), palette = None, width_ratios = None, height_ratios = None,
             unit = 'in', sharex = False, sharey = False):
    """A figure and its axes, as in matplotlib: one Axes, a 1D array for a single row or column, or a 2D array.
    `figsize` is in inches, or with unit = 'mm' in millimeters. `sharex` and `sharey` give axes the same limits, scale,
    ticks, and categories, each at one place in all of them: True or 'all' across the figure, 'row' within each row,
    'col' within each column. Then only the outer
    panels show those tick labels and the axis's name, the bottom row's for x shared by columns, the first column's for
    y shared by rows, and the plot areas line up across rows and columns."""
    fig = Figure(figsize[0], figsize[1], palette, unit)
    fig._grid = (nrows, ncols, width_ratios or [1] * ncols, height_ratios or [1] * nrows)
    for name, share in (('sharex', sharex), ('sharey', sharey)):
        if share not in (False, True, 'none', 'all', 'row', 'col'):
            raise ValueError(f"{name} has to be True, False, 'all', 'none', 'row', or 'col', got {share!r}")
    rows = [[Axes(fig, fig.pal) for _ in range(ncols)] for _ in range(nrows)]
    for r in range(nrows):
        for j in range(ncols):
            ax = rows[r][j]
            ax._cell = (r, j)
            fig.axes.append(ax)
            for axis, share in (('x', sharex), ('y', sharey)):
                if share in (True, 'all', 'row', 'col'):
                    ax._share[axis] = ([a for row in rows for a in row] if share in (True, 'all') else
                                       rows[r] if share == 'row' else [row[j] for row in rows])
                    # One list of categories for all, in order of first appearance in any, so that a category has one
                    # place in every panel, whichever groups each holds
                    ax._cats[axis] = ax._share[axis][0]._cats[axis]
            # As matplotlib labels them: x below the bottom row of a column that shares it, y beside the first column
            ax._outer = {'x': sharex in (True, 'all', 'col') and r < nrows - 1,
                         'y': sharey in (True, 'all', 'row') and j > 0}
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
               center = 0.0, xlabel = '', ylabel = '', colorbar = True, colorbar_label = '', glyphs = None,
               highlight = None):
    """A heatmap with both dendrograms and a color bar, as in seaborn. A DataFrame brings its own labels, and
    xticklabels and yticklabels can be True for all, False for none, or an integer for every n-th. Rows and columns are
    clustered by `method` (single, complete, average, weighted, centroid, median, or ward) on the `metric` distance
    (euclidean, sqeuclidean, cityblock, chebyshev, cosine, correlation, braycurtis, canberra, jensenshannon, or a
    function of two rows), and each tree's links turned so that neighbors are as alike as the tree allows. Linkages
    passed in scipy's format are drawn as they are. `glyphs` draws the rows' labels as glyphs where it can, such as
    'glycan' for SNFG drawings (see Axes.tick_params), or the (columns', rows') with a pair. `highlight` keeps the rows
    and columns it names in full shade, and the other cells recede (see Axes.highlight). Returns the figure and the
    heatmap's axes; the figure's row_order and col_order hold the data's rows and columns as drawn, first row at the
    top."""
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
                              center = center, names = names, highlight = highlight)
    heat.set_xlabel(xlabel)
    heat.set_ylabel(ylabel)
    for axis, maker in zip('xy', glyphs if isinstance(glyphs, tuple) else (None, glyphs)):
        if maker:
            heat.tick_params(axis, glyphs = maker)
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


def open(file, palette = None, size = None, unit = 'in', functions = None):
    """A figure back from a PDF, SVG, or PNG that viscose saved, from the calls and data the file carries (see
    Figure.save), with no script needed: as it was drawn, or in another `palette`, such as a name, a dark version, or
    a palette with larger type through dataclasses.replace, at another `size`, (width, height) in inches or with
    unit = 'mm' in millimeters, such as a journal's column. The file can also come as its bytes. Nothing in a file runs
    as code: it names Axes' plot methods and holds JSON and arrays, so a function that a plot call took, such as a
    heatmap's `fmt`, comes back only from `functions`, a dict by the name the error gives."""
    data = bytes(file) if isinstance(file, (bytes, bytearray, memoryview)) else Path(file).read_bytes()
    raw = None
    if data[:4] == b'%PDF':
        # The last such attachment, as an edited PDF appends; a stream is never inside an object stream, so its
        # dictionary is found as written, whichever program wrote the file last
        spots = list(re.finditer(rb'/Subtype\s*/application#2Fvnd\.viscose\+json', data))
        if spots:
            begin = re.compile(rb'stream\r?\n').search(data, spots[-1].end())
            entry = data[data.rfind(b'obj', 0, spots[-1].start()):begin.start()]
            length = re.search(rb'/Length\s+(\d+)(\s+\d+\s+R)?', entry)
            body = (data[begin.end():begin.end() + int(length.group(1))] if length and not length.group(2) else
                    data[begin.end():data.index(b'endstream', begin.end())].rstrip(b'\r\n'))
            raw = zlib.decompressobj().decompress(body) if b'FlateDecode' in entry else body
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        at = 8
        while at + 8 <= len(data) and raw is None:
            n, kind = struct.unpack('>I4s', data[at:at + 8])
            raw = zlib.decompress(data[at + 8:at + 8 + n]) if kind == SOURCE_CHUNK else None
            at += 12 + n
    else:
        found = re.search(rb'<metadata id="viscose-source">([A-Za-z0-9+/=\s]*)</metadata>', data)
        raw = zlib.decompress(base64.b64decode(found.group(1))) if found else None
    if raw is None:
        raise ValueError('This file carries no figure source: it was saved with source = False or as a PDF/X, by '
                         'another program, or before viscose 0.3.0, or an editor dropped it')
    text_part, _, tail = raw.partition(b'\n')
    head = json.loads(text_part)
    if head.get('format', 0) > SOURCE_FORMAT:
        raise ValueError(f"This figure's source is in a newer format, from viscose {head.get('viscose')}")
    arrays = []
    for t in head['arrays']:
        if 'values' in t:
            values = t['values']
            if np.dtype(t['dtype']).kind == 'f':
                values = [math.nan if u is None else float(u) if isinstance(u, str) else u for u in values]
            arrays.append(np.array(values, dtype = t['dtype']).reshape(t['shape']))
        else:
            store = np.dtype(t['store'])
            grouped = np.frombuffer(tail, np.uint8, t['length'], t['offset']).reshape(store.itemsize, -1)
            arrays.append(np.frombuffer(grouped.T.tobytes(), store).astype(t['dtype']).reshape(t['shape']))
    # A figure and axes that hold what in_palette copies, which then draws them anew
    stub, axes, cats = Figure.__new__(Figure), [Axes.__new__(Axes) for _ in head['axes']], {}
    dec = lambda v: _decode(v, arrays, axes, functions)
    for ax, h in zip(axes, head['axes']):
        ax._cell, ax._span, ax._fixed, ax._outer = dec(h['cell']), dec(h['span']), dec(h['fixed']), dec(h['outer'])
        ax._share = {k: [axes[i] for i in group] for k, group in h['share'].items()}
        ax._cats = {k: cats.setdefault((k, i), {}) for k, i in h['cats'].items()}
        ax._align = {k: axes[i] for k, i in h['align'].items()}
    found = head['figure']
    stub.axes = axes
    stub._base = dec(found['size']) if size is None else tuple(v * UNITS[_unit(unit, ('in', 'mm'))] for v in size)
    stub._title, stub._subtitle, stub._held, stub.legible = found['title'], found['subtitle'], found['held'], found[
        'legible']
    stub._grid, stub._texts = dec(found['grid']), dec(found['texts'])
    for _, name, _, _ in head['calls']:
        if name[:1] == '_' or not hasattr(getattr(Axes, name, None), '__wrapped__'):
            raise ValueError(f'{name!r} is not a plot method of Axes')
    stub._calls = [(axes[i], name, dec(args), dec(kwargs)) for i, name, args, kwargs in head['calls']]
    if 'order' in found:
        stub.row_order, stub.col_order = dec(found['order'])
    spec = head['palette']
    # The palette it was drawn in, whose shades the calls may name by color, as in_palette carries them over
    stub.pal = (_named(spec) if isinstance(spec, str) else _named(spec['dark']).dark(spec['paper']) if 'dark' in spec
                else Palette(**{k: dec(v) for k, v in spec['settings'].items()}))
    return stub.in_palette(stub.pal.name if palette is None and isinstance(spec, str) else
                           stub.pal if palette is None else palette)
