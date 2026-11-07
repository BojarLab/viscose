"""Mark drawing in page points. This is where each palette's special effects live."""
import math
from io import BytesIO
import numpy as np
from glycorender import pdfwrite, raster


KAPPA = 0.5522847498
# Vertices on the unit circle's scale; None is a true circle
_CROSS = [(-1, -.35), (-.35, -.35), (-.35, -1), (.35, -1), (.35, -.35), (1, -.35), (1, .35), (.35, .35), (.35, 1),
          (-.35, 1), (-.35, .35), (-1, .35)]
MARKERS = {'o': None,
           'X': [(x * .71 - y * .71, x * .71 + y * .71) for x, y in _CROSS],
           's': [(-.88, -.88), (.88, -.88), (.88, .88), (-.88, .88)],
           'D': [(0, -1.2), (1.2, 0), (0, 1.2), (-1.2, 0)],
           '^': [(-1.05, -.8), (1.05, -.8), (0, 1.15)],
           'v': [(-1.05, .8), (0, -1.15), (1.05, .8)],
           'P': _CROSS,
           '*': [(math.sin(i * math.pi / 5) * (1.25 if i % 2 == 0 else .55),
                  math.cos(i * math.pi / 5) * (1.25 if i % 2 == 0 else .55)) for i in range(10, 0, -1)],
           '>': [(-.8, -1.05), (1.15, 0), (-.8, 1.05)],
           '<': [(.8, 1.05), (-1.15, 0), (.8, -1.05)],
           # A hexagon standing on a corner and a pentagon on a side, with the area of the unit circle
           'h': [(1.1 * math.cos(math.radians(90 + 60 * i)), 1.1 * math.sin(math.radians(90 + 60 * i)))
                 for i in range(6)],
           'p': [(1.15 * math.cos(math.radians(90 + 72 * i)), 1.15 * math.sin(math.radians(90 + 72 * i)))
                 for i in range(5)]}
# From this many discs on, a raster output stamps them instead of tracing each one's curves, this many at a time: a
# smaller batch finds more of its pixels opaque already, a larger one costs less around them
STAMP_MIN, STAMP_BATCH = 50, 100
# Dense plots of ringed or layered points are drawn in at most about this many overlapping groups
POINT_GROUPS = 200
# A stamped marker other than a disc takes its coverage from a table per size, with its center on the nearest of this
# many positions per pixel and axis; a scatter with more sizes than this stays vector
SPRITE_PHASES, SPRITE_SIZES = 8, 4


def _style(c, rgb = None, alpha = 1.0, stroke = None, lw = None, stroke_alpha = 1.0, cap = 0, join = 0, dash = None):
    if rgb is not None:
        c.setFillColorRGB(*rgb, alpha = alpha)
    if stroke is not None:
        c.setStrokeColorRGB(*stroke, alpha = stroke_alpha)
        c.setLineWidth(lw)
        c.setLineCap(cap)
        c.setLineJoin(join)
        c.setDash(dash)


def _marker_path(p, unit, x, y, r):
    if unit is None:
        k = r * KAPPA
        p.ops.extend((('m', x - r, y), ('c', x - r, y + k, x - k, y + r, x, y + r),
                      ('c', x + k, y + r, x + r, y + k, x + r, y), ('c', x + r, y - k, x + k, y - r, x, y - r),
                      ('c', x - k, y - r, x - r, y - k, x - r, y), ('h',)))
    else:
        p.ops.append(('m', x + unit[0][0] * r, y + unit[0][1] * r))
        p.ops.extend([('l', x + ux * r, y + uy * r) for ux, uy in unit[1:]])
        p.ops.append(('h',))


def _stamp_buffer(c, xs, ys, rs, marker):
    """A transparent layer at the output resolution, over the part of the clip that the marks can reach."""
    sc = c.stamp_dpi / 72.0
    (bx, by, bw, bh), _ = c.state['clip'][0]
    reach = float(np.max(rs)) + 3.0 if marker == 'o' else 1.4 * float(np.max(rs)) + 9.0
    px0 = int(math.floor(max(bx, float(np.min(xs)) - reach) * sc))
    py0 = int(math.floor(max(by, float(np.min(ys)) - reach) * sc))
    nx = int(math.ceil(min(bx + bw, float(np.max(xs)) + reach) * sc)) - px0
    ny = int(math.ceil(min(by + bh, float(np.max(ys)) + reach) * sc)) - py0
    if nx <= 0 or ny <= 0:
        return None
    return dict(sc = sc, px0 = px0, py0 = py0, nx = nx, ny = ny, rgb = np.zeros((ny * nx, 3), np.float32),
                alpha = np.zeros(ny * nx, np.float32), last = np.zeros(ny * nx, np.int32), sprites = {})


def _in_order(found):
    """Arrays per marker size, whose last one names the marker of each entry, as one set in drawing order."""
    if len(found) == 1:
        return found[0]
    parts = [np.concatenate(v) for v in zip(*found)]
    order = np.argsort(parts[-1], kind = 'stable')
    return [v[order] for v in parts]


def _under(buf, spots, fill, edge, rgb, alpha, stroke):
    """Paint coverage per entry under what the layer already holds. Drawing runs from the top down, so that pixels gone
    opaque cost nothing."""
    opacity, premul, last = buf['alpha'], buf['rgb'], buf['last']
    color, pen = np.asarray(rgb, np.float32), np.asarray((0.0, 0.0, 0.0) if stroke is None else stroke, np.float32)
    # The stroke lies over the fill and is opaque, as it is when a path is both filled and stroked
    body = fill * (1.0 - edge) * np.float32(alpha)
    a = body + edge
    drawn = a > 0.0
    spots, body, edge, a = spots[drawn], body[drawn], edge[drawn], a[drawn]
    # Several marks can reach one pixel, and writing them all at once would keep only the last: each round takes the
    # topmost one left at each pixel, as the last index written to a pixel stays there
    while len(spots):
        n = np.arange(len(spots), dtype = np.int32)
        last[spots] = n
        top = last[spots] == n
        s = spots[top]
        under = 1.0 - opacity[s]
        premul[s] += (body[top, None] * color + edge[top, None] * pen) * under[:, None]
        opacity[s] += a[top] * under
        left = ~top
        spots, body, edge, a = spots[left], body[left], edge[left], a[left]
        left = opacity[spots] < 0.999
        spots, body, edge, a = spots[left], body[left], edge[left], a[left]


def _stamp(buf, xs, ys, rs, rgb, alpha, stroke, lw):
    """Paint discs of one color into the layer, from each disc's coverage of the pixels around its center."""
    sc, nx, ny, opacity = buf['sc'], buf['nx'], buf['ny'], buf['alpha']
    cx = (xs * sc - buf['px0']).astype(np.float32)
    cy = (buf['py0'] + ny - ys * sc).astype(np.float32)
    cr, hw = (rs * sc).astype(np.float32), float(lw * sc / 2.0)
    reach = np.ceil(cr + hw + 1.0).astype(int)
    found = []
    for k in np.unique(reach):
        # The pixels around a center that a disc of this reach can touch, from any position within its pixel
        off = np.arange(-k, k + 1)
        near = off[:, None] ** 2 + off[None, :] ** 2 <= (k + 2) ** 2
        oy, ox = (v[near] for v in np.meshgrid(off, off, indexing = 'ij'))
        i = np.flatnonzero(reach == k)
        px, py = np.floor(cx[i]), np.floor(cy[i])
        ix, iy = px.astype(np.int32)[:, None] + ox, py.astype(np.int32)[:, None] + oy
        on = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        spots = (iy * nx + ix)[on]
        seen = opacity[spots] < 0.999
        spots = spots[seen]
        # Each pixel's offset from its disc's center, from the same table
        dx = (ox + 0.5).astype(np.float32) - (cx[i] - px)[:, None]
        dy = (oy + 0.5).astype(np.float32) - (cy[i] - py)[:, None]
        found.append((spots, np.sqrt(dx * dx + dy * dy)[on][seen], np.repeat(i, on.sum(axis = 1))[seen]))
    spots, d, who = _in_order(found)
    fill = np.clip(cr[who] + 0.5 - d, 0.0, 1.0)
    edge = np.clip(hw + 0.5 - np.abs(d - cr[who]), 0.0, 1.0) if stroke is not None else np.zeros_like(fill)
    _under(buf, spots, fill, edge, rgb, alpha, stroke)


def _stamp_shapes(buf, xs, ys, rs, rgb, alpha, stroke, lw, marker):
    """Paint markers of one color into the layer, from a table of each one's coverage, each within a sixteenth of a
    pixel of where it belongs."""
    sc, nx, ny, opacity = buf['sc'], buf['nx'], buf['ny'], buf['alpha']
    qx = np.floor((xs * sc - buf['px0']) * SPRITE_PHASES + 0.5).astype(int)
    qy = np.floor((buf['py0'] + ny - ys * sc) * SPRITE_PHASES + 0.5).astype(int)
    phase = (qy % SPRITE_PHASES) * SPRITE_PHASES + qx % SPRITE_PHASES
    found = []
    for r in np.unique(rs):
        key = (marker, float(r), float(lw), stroke is not None)
        if key not in buf['sprites']:
            # One marker's fill and stroke coverage, as seen from the pixels that either ever reaches, per position of
            # its center within a pixel, drawn by the rasterizer itself
            unit = MARKERS[marker]
            # A miter stays within five line widths of its corner at the default limit
            k = int(math.ceil((1.4 * float(r) + 5.0 * lw) * sc)) + 2
            page = (2 * k + 1) / sc
            pen = pdfwrite.Canvas(BytesIO(), pagesize = (page, page))
            _style(pen, (0.0, 0.0, 0.0), 1.0, (0.0, 0.0, 0.0), lw, 1.0)
            cover, inked = (np.zeros((SPRITE_PHASES ** 2, 2 * k + 1, 2 * k + 1), np.float32) for _ in range(2))
            for py in range(SPRITE_PHASES):
                for px in range(SPRITE_PHASES):
                    p = pen.beginPath()
                    _marker_path(p, unit, (k + px / SPRITE_PHASES) / sc, (k + 1 - py / SPRITE_PHASES) / sc, float(r))
                    pen.ops = []
                    pen.drawPath(p, fill = 1, stroke = 0)
                    cover[py * SPRITE_PHASES + px] = raster.render(pen.ops, page, page, sc, sc)[..., 3] / 255.0
                    if stroke is not None:
                        pen.ops = []
                        pen.drawPath(p, fill = 0, stroke = 1)
                        inked[py * SPRITE_PHASES + px] = raster.render(pen.ops, page, page, sc, sc)[..., 3] / 255.0
            oy, ox = np.nonzero(cover.max(axis = 0) + inked.max(axis = 0))
            buf['sprites'][key] = oy - k, ox - k, cover[:, oy, ox], inked[:, oy, ox]
        oy, ox, fill, edge = buf['sprites'][key]
        i = np.flatnonzero(rs == r)
        ix, iy = (qx[i] // SPRITE_PHASES)[:, None] + ox, (qy[i] // SPRITE_PHASES)[:, None] + oy
        on = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        spots = (iy * nx + ix)[on]
        seen = opacity[spots] < 0.999
        found.append((spots[seen], fill[phase[i]][on][seen], edge[phase[i]][on][seen],
                      np.repeat(i, on.sum(axis = 1))[seen]))
    spots, fill, edge, _ = _in_order(found)
    _under(buf, spots, fill, edge, rgb, alpha, stroke)


def _stamp_op(c, buf):
    """The finished layer as one image, whose pixels lie on the output's own."""
    sc, nx, ny = buf['sc'], buf['nx'], buf['ny']
    # Only the covered pixels: the rest of a sparse layer stays zero
    drawn = np.flatnonzero(buf['alpha'] > 0)
    a, rgb = buf['alpha'][drawn], np.zeros((ny * nx, 3), np.uint8)
    rgb[drawn] = (np.clip(buf['rgb'][drawn] / a[:, None], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    alpha = np.zeros(ny * nx, np.uint8)
    alpha[drawn] = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    op = dict(c.state)
    op.update(kind = 'image', w = nx, h = ny, rgb = rgb.tobytes(), alpha = alpha.tobytes(), fill_alpha = 1.0,
              ctm = (nx / sc, 0.0, 0.0, ny / sc, buf['px0'] / sc, buf['py0'] / sc))
    c.ops.append(op)


def shapes(c, xs, ys, rs, rgb, alpha = 1.0, stroke = None, lw = 0.0, marker = 'o', form = False):
    """Markers of one color as a single compound path, which keeps thousands of points light in every format, or as one
    stored shape placed at each point, to a hundredth of a point, if `form` says that they have one size."""
    if not len(xs):
        return
    unit = MARKERS[marker]
    p = c.beginPath()
    place = [(0.0, 0.0, float(rs[0]))] if form else zip(xs.tolist(), ys.tolist(), rs.tolist())
    for x, y, r in place:
        _marker_path(p, unit, x, y, r)
    c.saveState()
    _style(c, rgb, alpha, stroke, lw, 1.0)
    if form:
        c.drawMarkers(p, np.round(xs, 2).tolist(), np.round(ys, 2).tolist(), fill = 0 if rgb is None else 1,
                      stroke = 1 if stroke is not None else 0)
    else:
        c.drawPath(p, fill = 0 if rgb is None else 1, stroke = 1 if stroke is not None else 0)
    c.restoreState()


def polyline(c, xs, ys, rgb, lw, alpha = 1.0, cap = 1, join = 1, dash = None):
    polylines(c, [(xs, ys)], rgb, lw, alpha, cap, join, dash)


def polylines(c, lines, rgb, lw, alpha = 1.0, cap = 1, join = 1, dash = None):
    """Several (xs, ys) lines of one style as one path."""
    p = c.beginPath()
    for xs, ys in lines:
        p.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            p.lineTo(x, y)
    c.saveState()
    _style(c, stroke = rgb, lw = lw, stroke_alpha = alpha, cap = cap, join = join, dash = dash)
    c.drawPath(p, fill = 0, stroke = 1)
    c.restoreState()


def polygon(c, xs, ys, rgb, alpha = 1.0, grad = None):
    """A filled outline; grad is ((x1, y1, x2, y2), colors), with colors evenly spaced and of three or four values."""
    # A flat outline has no area, though a PDF reader may paint one as a hairline
    if max(ys) <= min(ys):
        return
    p = c.beginPath()
    p.moveTo(xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        p.lineTo(x, y)
    p.close()
    c.saveState()
    if grad is not None:
        c.setFillColorRGB(0.0, 0.0, 0.0, alpha = alpha)
        c.setFillGradient('linear', grad[0], [(i / (len(grad[1]) - 1), col) for i, col in enumerate(grad[1])])
    else:
        _style(c, rgb, alpha)
    c.drawPath(p, fill = 1, stroke = 0)
    c.restoreState()


def rounded(c, x, y, w, h, r, rgb):
    """A rectangle with rounded corners."""
    r = min(r, w / 2, h / 2)
    k = r * (1 - KAPPA)
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.curveTo(x + w - k, y, x + w, y + k, x + w, y + r)
    p.lineTo(x + w, y + h - r)
    p.curveTo(x + w, y + h - k, x + w - k, y + h, x + w - r, y + h)
    p.lineTo(x + r, y + h)
    p.curveTo(x + k, y + h, x, y + h - k, x, y + h - r)
    p.lineTo(x, y + r)
    p.curveTo(x, y + k, x + k, y, x + r, y)
    p.close()
    c.saveState()
    _style(c, rgb)
    c.drawPath(p, fill = 1, stroke = 0)
    c.restoreState()


def _layers(pal, xs, ys, r, rgb, edge):
    """A group of points in the palette's manner, as (xs, ys, radii, color, stroke, stroke width) from the bottom up."""
    if edge is not None:
        return [(xs, ys, r, rgb, edge[0], edge[1])]
    if pal.point == 'sphere':
        dark, base, light = pal.shades(rgb)
        big = r > 2.0
        return [(xs, ys, r * 1.2, dark, None, 0.0), (xs, ys, r, base, None, 0.0),
                (xs[big] - r[big] * 0.22, ys[big] + r[big] * 0.22, r[big] * 0.5, light, None, 0.0)]
    if pal.point == 'ring':
        return [(xs, ys, r, rgb, pal.color('paper'), 0.35)]
    if pal.point == 'cut':
        # A ring as wide as a small point's radius would eat the point
        return [(xs, ys, r, rgb, pal.color('paper'), min(1.1, 0.25 * float(np.min(r))))]
    return [(xs, ys, r, rgb, None, 0.0)]


def points(c, pal, xs, ys, area, rgb, alpha = 1.0, marker = 'o', edge = None):
    """Scatter marks in the palette's manner; `area` is in points squared, as in matplotlib. `edge` is (color,
    width), and with `rgb` None it is all that shows."""
    n = len(xs)
    if not n:
        return
    xs, ys = np.asarray(xs, dtype = float), np.asarray(ys, dtype = float)
    r = np.broadcast_to(np.sqrt(np.maximum(np.asarray(area, dtype = float), 0.0) * pal.point_scale) / 2.0, xs.shape)
    # An area of zero draws nothing, though a ring around it would show
    drawn = r > 0.0
    xs, ys, r = xs[drawn], ys[drawn], r[drawn]
    n = len(xs)
    if not n:
        return
    # Later points must cover earlier ones, or one path's rings would all lie over all its fills. Groups do that and
    # keep dense plots light; points that look alike when merged need no order.
    step = max(1, n // POINT_GROUPS) if edge is not None or pal.point != 'flat' or alpha < 1.0 else n
    if not (getattr(c, 'stamp_dpi', None) and n >= STAMP_MIN and c.state['clip']
            and (marker == 'o' or len(np.unique(r)) <= SPRITE_SIZES)):
        form = getattr(c, 'marker_forms', False) and np.min(r) == np.max(r)
        for start in range(0, n, step):
            chosen = slice(start, start + step)
            for lx, ly, lr, color, stroke, lw in _layers(pal, xs[chosen], ys[chosen], r[chosen], rgb, edge):
                shapes(c, lx, ly, lr, color, alpha, stroke, lw, marker, form)
        return
    buf = _stamp_buffer(c, xs, ys, r, marker)
    if buf is None:
        return
    rgb, alpha = ((0.0, 0.0, 0.0), 0.0) if rgb is None else (rgb, alpha)
    step = min(step, STAMP_BATCH)
    for start in reversed(range(0, n, step)):
        chosen = slice(start, start + step)
        for lx, ly, lr, color, stroke, lw in reversed(_layers(pal, xs[chosen], ys[chosen], r[chosen], rgb, edge)):
            if len(lx) and marker == 'o':
                _stamp(buf, lx, ly, lr, color, alpha, stroke, lw)
            elif len(lx):
                _stamp_shapes(buf, lx, ly, lr, color, alpha, stroke, lw, marker)
    _stamp_op(c, buf)


def line(c, pal, xs, ys, rgb, lw = None, alpha = 1.0, dash = None):
    lw = pal.line_width if lw is None else lw
    if pal.glow:
        for k, a in ((4.2, 0.06), (2.7, 0.1), (1.6, 0.2)):
            polyline(c, xs, ys, rgb, lw * k, a * alpha)
    polyline(c, xs, ys, rgb, lw, alpha, cap = 0 if pal.bar == 'needle' else 1, dash = dash)


def arrow(c, pal, x0, y0, x1, y1, rgb, lw, alpha = 1.0, head = None, bend = 0.0):
    """A shaft ending in a filled head at (x1, y1), scaled to the palette's line weight, with the head `head` points
    long if given. With a `bend`, the shaft is an arc whose middle stands that share of its length off the straight
    line, to the left of its direction for a positive bend."""
    width = lw * pal.line_width / 2.0
    n = math.hypot(x1 - x0, y1 - y0)
    if not n:
        return
    # The arc's control point; the head points along the arc where it ends
    cx, cy = (x0 + x1) / 2 - bend * (y1 - y0), (y0 + y1) / 2 + bend * (x1 - x0)
    tx, ty = (x1 - cx, y1 - cy) if bend else (x1 - x0, y1 - y0)
    ux, uy = tx / math.hypot(tx, ty), ty / math.hypot(tx, ty)
    head = min(max(6.0, width * 3.2) if head is None else head, n)
    bx, by = x1 - ux * head, y1 - uy * head
    if head < n and bend:
        t = np.linspace(0.0, 1.0, 24)[:, None]
        q = (1 - t) ** 2 * np.array([x0, y0]) + 2 * t * (1 - t) * np.array([cx, cy]) + t ** 2 * np.array([bx, by])
        polyline(c, q[:, 0], q[:, 1], rgb, width, alpha, cap = 0)
    elif head < n:
        polyline(c, [x0, bx], [y0, by], rgb, width, alpha, cap = 0)
    half = head * 0.42
    polygon(c, [x1, bx - uy * half, bx + uy * half], [y1, by + ux * half, by - ux * half], rgb, alpha)


def bar(c, pal, x, y, w, h, rgb, horizontal = False, alpha = 1.0, flip = False, reach = 0.7):
    """A bar in the palette's manner; (x, y) is its lower left corner, and the base is at y, or at x if horizontal, or
    at the opposite end if `flip`. A needle's top reaches past either side by `reach` bar widths."""
    if w <= 0 or h <= 0:
        return
    dark, base, light = pal.shades(rgb)
    xs, ys = [x, x + w, x + w, x], [y, y, y + h, y + h]
    axis = (x, y, x + w, y) if horizontal else (x, y, x, y + h)
    axis = (axis[2], axis[3], axis[0], axis[1]) if flip else axis
    if pal.bar == 'deep':
        polygon(c, xs, ys, None, alpha, (axis, (dark, base, light)))
    elif pal.bar == 'soft':
        polygon(c, xs, ys, None, alpha, (axis, (light, base)))
    elif pal.bar == 'needle':
        polygon(c, xs, ys, rgb, alpha)
        if horizontal:
            tip = x if flip else x + w
            polyline(c, [tip, tip], [y - h * reach, y + h * (1 + reach)], dark, 0.9, alpha, cap = 0)
        else:
            tip = y if flip else y + h
            polyline(c, [x - w * reach, x + w * (1 + reach)], [tip, tip], dark, 0.9, alpha, cap = 0)
    else:
        polygon(c, xs, ys, rgb, alpha)


def box(c, pal, x, w, stats, rgb, alpha = 1.0):
    """A box plot's whiskers, box in the palette's manner of bars, and median; (x, w) is the box's left edge and width,
    and `stats` holds the lower whisker, quartile, median, quartile, and upper whisker in page points."""
    low, q1, median, q3, high = stats
    lw, mid = max(pal.line_width * 0.5, 0.9), x + w / 2
    # A shade darker than the box, as a pale one would otherwise leave them hard to see
    polylines(c, [([mid, mid], [low, q1]), ([mid, mid], [q3, high]), ([mid - w / 4, mid + w / 4], [low, low]),
                  ([mid - w / 4, mid + w / 4], [high, high])], pal.shades(rgb)[0], lw, alpha, cap = 0)
    # A needle's wide top would read as part of the data here
    if pal.bar == 'needle':
        polygon(c, [x, x + w, x + w, x], [q1, q1, q3, q3], rgb, alpha)
    else:
        bar(c, pal, x, q1, w, q3 - q1, rgb, False, alpha)
    polyline(c, [x, x + w], [median, median], pal.color(pal.on(rgb)), max(lw, 1.4), alpha, cap = 0)


def violin(c, pal, xs, ys, cx, half, rgb, alpha = 1.0):
    """A violin in the palette's manner of bars, its shades running across it from its middle at `cx` out to `half` on
    either side; a needle's palette draws it pale inside a hairline, as its bars are slender."""
    dark, base, light = pal.shades(rgb)
    across = (cx - half, ys[0], cx + half, ys[0])
    if pal.bar == 'deep':
        polygon(c, xs, ys, None, alpha, (across, (dark, light, dark)))
    elif pal.bar == 'soft':
        polygon(c, xs, ys, None, alpha, (across, (light, base, light)))
    elif pal.bar == 'needle':
        polygon(c, xs, ys, light, alpha)
        p = c.beginPath()
        p.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            p.lineTo(x, y)
        p.close()
        c.saveState()
        _style(c, stroke = rgb, lw = 0.9, stroke_alpha = alpha, cap = 1, join = 1)
        c.drawPath(p, fill = 0, stroke = 1)
        c.restoreState()
    else:
        polygon(c, xs, ys, rgb, alpha)


def inner_box(c, pal, x, w, stats, rgb, alpha = 1.0):
    """The quartiles and median inside a violin, as a slim bar `w` wide on a line between the whiskers, with the median
    a dot in the page's color; `stats` are the lower whisker, quartile, median, quartile, and upper whisker in page
    points."""
    low, q1, median, q3, high = stats
    dark = pal.shades(rgb)[0]
    # A needle's violin is pale, so its own color stands out on it; others take a darker shade, or ink if none is darker
    ink = rgb if pal.bar == 'needle' else dark if dark != pal.color(rgb) else pal.color(pal.on(rgb))
    polyline(c, [x, x], [low, high], ink, max(w * 0.22, 0.8), alpha, cap = 0)
    if q3 > q1:
        polygon(c, [x - w / 2, x + w / 2, x + w / 2, x - w / 2], [q1, q1, q3, q3], ink, alpha)
    shapes(c, np.array([x]), np.array([median]), np.array([w * 0.36]), pal.color('paper'), alpha)


def segment(c, pal, x, y, w, h, rgb, gap, alpha = 1.0):
    """One piece of a stacked bar, with a gap to the piece below it."""
    dark, base, light = pal.shades(rgb)
    y0, h0 = y + gap, h - gap
    if h0 <= 0:
        return
    xs, ys = [x, x + w, x + w, x], [y0, y0, y0 + h0, y0 + h0]
    if pal.bar in ('deep', 'soft'):
        polygon(c, xs, ys, None, alpha, ((x, y0, x, y0 + h0), (base, light)))
    else:
        polygon(c, xs, ys, rgb, alpha)


def ridge(c, pal, xs, base, ys, rgb, top_rgb, peak):
    """One ridge of a ridgeline, with the page's color behind it so that it hides the ridges at its back."""
    paper = pal.color('paper')
    px = np.r_[xs, xs[::-1]]
    py = np.r_[ys, np.full(len(xs), base)]
    if pal.ridge == 'deep':
        polygon(c, px, py, paper)
        polygon(c, px, py, None, 1.0, ((xs[0], base, xs[0], base + peak), (rgb, top_rgb)))
        polyline(c, xs, ys, paper, pal.ridge_crest)
    elif pal.ridge == 'tint':
        polygon(c, px, py, top_rgb)
        polyline(c, xs, ys, rgb, pal.ridge_crest)
    else:
        polygon(c, px, py, rgb)
        polyline(c, xs, ys, paper, pal.ridge_crest)


def heat_cell(c, pal, x, y, w, h, rgb):
    # A gap that suits a large cell would swallow a small one
    g = min(pal.heat_gap, 0.2 * min(w, h)) / 2.0
    if pal.heat_round:
        rounded(c, x + g, y + g, w - 2 * g, h - 2 * g, pal.heat_round * min(w, h), rgb)
    else:
        polygon(c, [x + g, x + w - g, x + w - g, x + g], [y + g, y + g, y + h - g, y + h - g], rgb)
