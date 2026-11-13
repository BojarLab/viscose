"""Axes: records what to draw, then lays it out and draws it once the figure is rendered."""
import dataclasses
import functools
import math
import numbers
from io import BytesIO
import numpy as np
from glycorender import pdfwrite, raster
from . import marks, text
from .palettes import LC_TEXT, legible, mix


# Past this many cells a heatmap is one image, as matplotlib's rasterized heatmaps are, not thousands of shapes
HEAT_IMAGE_CELLS = 2500
# The kinds of marks that a highlight keeps in full shade or lets recede
FOCUS_KINDS = ('points', 'line', 'band', 'bars', 'box', 'violin', 'heat', 'stems', 'vsegs')
# How wide, in points, an interactive SVG lets the pointer find a named arrow
ARROW_GRIP = 9.0
LEGEND_ROW = 16.0
COLORBAR_WIDTH = 11.0
LEGEND_SPOTS = ('direct', 'below', 'right', 'upper left', 'upper right', 'lower left', 'lower right')
# A group of points is named at its middle only where this share of its points lies nearer its own middle than any other
# group's, as the middle of groups that mingle names no place of theirs
DIRECT_OWN = 0.7
# Axis names shrink to fit their cell down to this size, and past it the page grows
LABEL_MIN = 6.5
# Margins may take this share of an axes' cell before the figure grows
PLOT_SHARE_MARGINS = 0.55
# Brackets may stretch the y axis by up to this factor, which leaves the data at least this share of the height
BRACKET_LIFT_MAX = 4.0
# The sizes, as shares of their type's, that spread glyph annotations may take, from the largest that fits
GLYPH_SIZES = (1.75, 1.5, 1.3, 1.15, 1.0, 0.9, 0.8, 0.72, 0.65, 0.58, 0.52, 0.46, 0.4)


def nice_ticks(lo, hi, n = 6, steps = (1, 2, 5, 10)):
    """Round tick positions covering [lo, hi], about n of them, at a step of one of `steps` times a power of ten, such
    as (1, 2, 2.5, 5, 10) for matplotlib's quarters too."""
    if not math.isfinite(hi - lo) or hi <= lo:
        return [lo]
    mag = 10 ** math.floor(math.log10((hi - lo) / max(n - 1, 1)))
    # The round step whose tick count comes closest to n, counting the ticks that actually fall inside the range
    step = min((s * mag for s in steps),
               key = lambda v: abs(math.floor(hi / v + 1e-9) - math.ceil(lo / v - 1e-9) + 1 - n))
    first = math.ceil(lo / step - 1e-9)
    # Rounded to the step's own scale, which keeps 0.6 from reading 0.6000000000000001 and 3e-12 from reading 0
    digits = 10 - math.floor(math.log10(step))
    return [round(i * step, digits) for i in range(first, int(math.floor(hi / step + 1e-9)) + 1)]


def fmt(v, powers = False, digits = 3):
    """A tick value without trailing zeros or a stray minus zero, with `digits` significant digits beside a power of ten
    and three more without one. `powers` writes it with a power of ten however small it is, for the ticks beside a large
    one."""
    if v and (powers or abs(v) >= 1e5 or abs(v) < 1e-3):
        e = int(math.floor(math.log10(abs(v))))
        m = '%.*g' % (digits, v / 10.0 ** e)
        # Rounding can carry over into the next power of ten
        if m.lstrip('-') == '10':
            m, e = m.replace('10', '1'), e + 1
        return f'{m[:-1]}10^{{{e}}}' if m.lstrip('-') == '1' else f'{m}×10^{{{e}}}'
    s = '%.*g' % (digits + 3, v)
    return '0' if s == '-0' else s


def tick_labels(ticks):
    """Ticks in one notation, as 80000 next to 10^5 or 0.001 next to 8×10^-4 reads badly."""
    powers = any(v and (abs(v) >= 1e5 or abs(v) < 1e-3) for v in ticks)
    # As many digits as it takes to tell the ticks apart, as on an axis from 1234567 to 1234570
    for digits in range(3, 16):
        labels = [fmt(v, powers, digits) for v in ticks]
        if len(set(labels)) == len(set(ticks)):
            break
    return labels


def _minus(pal, labels):
    """Numbers written with the palette's minus sign, such as a true minus where its font has one."""
    return labels if pal.minus == '-' else [s.replace('-', pal.minus) for s in labels]


def _keyset(keys):
    """What a highlight names, as a set of labels: one label or several, or None for no highlight."""
    if keys is None:
        return None
    return {str(keys)} if isinstance(keys, str) or not np.iterable(keys) else {str(k) for k in keys}


def _numeric(values):
    """Whether values are numbers rather than category labels."""
    return np.asarray(values).dtype.kind in 'fiu'


def _floats(values):
    """Values as a flat array of floats, with anything that is not a number, such as a missing value, as NaN."""
    try:
        return np.asarray(values, dtype = float).ravel()
    except (TypeError, ValueError):
        return np.array([float(u) if isinstance(u, numbers.Real) else np.nan
                         for u in np.asarray(values, dtype = object).ravel()], dtype = float)


def _levels(values, order):
    """Category or hue labels as text, None where one is missing, and their levels in order: as given, else a pandas
    categorical's own, else numbers by size, else in order of first appearance."""
    order = list(values.cat.categories) if order is None and hasattr(values, 'cat') else order
    raw = list(np.asarray(values, dtype = object).ravel())
    missing = [u is None or type(u).__name__ in ('NAType', 'NaTType') or isinstance(u, numbers.Real) and u != u
               for u in raw]
    labels = np.array([None if gone else str(u) for u, gone in zip(raw, missing)], dtype = object)
    found = list(dict.fromkeys(s for s in labels if s is not None))
    numeric = all(isinstance(u, numbers.Real) and not isinstance(u, (bool, np.bool_))
                  for u, gone in zip(raw, missing) if not gone)
    if order is not None:
        return labels, [str(u) for u in order]
    return labels, sorted(found, key = float) if numeric else found


def _runs(ok):
    """Slices of the stretches where a mask holds, which a NaN breaks a line or a band into."""
    edges = np.flatnonzero(np.diff(np.r_[False, ok, False]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


def _gap(positions):
    """The smallest distance between two positions that differ, as ticks set by hand can repeat."""
    gaps = np.abs(np.diff(positions))
    gaps = gaps[gaps > 0]
    return float(gaps.min()) if gaps.size else math.inf


def _extra_lines(s):
    """How many lines a text has past its first."""
    return str(s).count('\n')


def _check_cmap(cmap):
    """A palette brings its color scales; any other name would fail, or worse succeed, as an attribute lookup."""
    if cmap not in ('diverging', 'sequential', 'sequential_r'):
        raise ValueError(f"cmap has to be 'diverging', 'sequential', or 'sequential_r', got {cmap!r}: a palette brings "
                         "these scales")


# Each scale's map from data to even spacing along the axis, and back
WARPS = {'linear': (lambda v: v, lambda u: u),
         'log': (lambda v: np.log10(np.maximum(v, 1e-300)), lambda u: 10.0 ** u),
         'sqrt': (lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda u: np.sign(u) * u * u)}


def _warp(scale, functions):
    """A scale's (forward, inverse) maps; any other name would quietly draw a linear axis."""
    if scale not in (*WARPS, 'function'):
        raise ValueError(f"An axis scale has to be 'linear', 'log', 'sqrt', or 'function', got {scale!r}")
    if (scale == 'function') != (functions is not None):
        raise ValueError("A 'function' scale takes functions = (forward, inverse), and no other scale takes them")
    return WARPS[scale] if functions is None else tuple(functions)


def _check_limits(lo, hi):
    """Limits as floats, None for an end that stays automatic, or None for both; a NaN or an infinity would otherwise
    only fail once the figure is saved."""
    if any(v is not None and not math.isfinite(v) for v in (lo, hi)):
        raise ValueError(f'Axis limits have to be finite, got {lo} and {hi}')
    return None if lo is None and hi is None else tuple(None if v is None else float(v) for v in (lo, hi))


def _check_marker(marker):
    """Any other name would otherwise only fail once the figure is saved."""
    if marker not in marks.MARKERS:
        raise ValueError(f"Unknown marker {marker!r}; available: {', '.join(marks.MARKERS)}")


def _check_tick_labels(ticks, labels):
    """Too few labels would otherwise only fail once the figure is saved, and too many would pass unnoticed."""
    if labels is not None and ticks is None:
        raise ValueError('Tick labels need the ticks they belong to')
    if labels is not None and len(labels) != len(ticks):
        raise ValueError(f'Ticks need one label each, got {len(ticks)} ticks and {len(labels)} labels')


def _check_lengths(x, y):
    """Different lengths would otherwise only fail once the figure is saved."""
    if x.shape != y.shape:
        raise ValueError(f'x and y have to be of the same length, got {x.size} and {y.size} values')


def _check_names(names, n):
    """A name per mark, or none, as a wrong count would only fail under the mouse."""
    if names is not None and len(names) != n:
        raise ValueError(f'names needs one name per mark, got {len(names)} for {n}')


def _spread_glyphs(placements, canvas, marks = (), ports = None, pad = 4.0, step = 3.0, radii = None):
    """Places each drawing at the free spot nearest its anchor, all clear of each other, with short leaders that do not
    cross, from glycowork's _spread_glycans, which annotate_figure placed glycans with. `placements` holds (x, y, w, h,
    anchor x, anchor y) per drawing, (x, y) being its ink's natural top left corner, in points with y running down
    from the top of `canvas`, (width, height), the room they stay in; `marks` holds (x, y, weight) of what a drawing
    should rather not hide, and `ports` per drawing the (dx, dy, radius) from its corner of the symbols a leader may
    end on, its middle by default, and `radii` each anchor's own, a point's marker, from whose edge nearness counts.
    Gives the corners, the leaders as (x0, y0, x1, y1) or None, and how many leaders cross another, run through a
    drawing, run long, or a drawing that found room only past the canvas's right."""
    P = np.array(placements, dtype = float).reshape(-1, 6)
    n, (cw, ch) = len(P), canvas
    width = cw
    ports = [np.array([(w / 2, h / 2, min(w, h) / 2)]) if q is None or not len(q) else
             np.array(q, dtype = float).reshape(-1, 3) for q, (_, _, w, h, _, _) in zip(ports or [None] * n, P)]
    pos, ends = P[:, :2].copy(), np.full((n, 2), np.nan)
    placed = np.zeros(n, dtype = bool)
    edge, radii = min(2.0, pad), np.zeros(n) if radii is None else np.asarray(radii, dtype = float)
    soft = np.concatenate([np.asarray(marks, dtype = float).reshape(-1, 3),
                           np.column_stack([P[:, 4:6], np.full(n, 150.0)])])
    grid_w, grid_h = int(cw * 3 / step) + 8, int(ch / step) + 8
    cover = np.zeros((grid_h + 1, grid_w + 1))
    inside = ((soft[:, 0] >= 0) & (soft[:, 1] >= 0) & (soft[:, 0] < (grid_w - 1) * step) &
              (soft[:, 1] < (grid_h - 1) * step))
    np.add.at(cover, ((soft[inside, 1] / step).astype(int) + 1, (soft[inside, 0] / step).astype(int) + 1),
              soft[inside, 2])
    cover = cover.cumsum(0).cumsum(1)
    def hits_box(x0, y0, x1, y1, left, top, right, bottom):
        # Whether segments run through boxes, all broadcast together
        dx, dy = x1 - x0, y1 - y0
        lo = np.zeros(np.broadcast(dx, dy, left, right).shape)
        hi, hit = np.ones_like(lo), np.ones_like(lo, dtype = bool)
        for p, q in ((-dx, x0 - left), (dx, right - x0), (-dy, y0 - top), (dy, bottom - y0)):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                r = q / p
            hit &= (p != 0) | (q >= 0)
            lo, hi = np.where(p < 0, np.maximum(lo, r), lo), np.where(p > 0, np.minimum(hi, r), hi)
        return hit & (lo <= hi)
    def crossed(x0, y0, x1, y1, u0, v0, u1, v1):
        # Whether segments cross each other, all broadcast together
        side = lambda ax, ay, bx, by, cx, cy: (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return ((side(u0, v0, u1, v1, x0, y0) * side(u0, v0, u1, v1, x1, y1) < 0) &
                (side(x0, y0, x1, y1, u0, v0) * side(x0, y0, x1, y1, u1, v1) < 0))
    def port(i, xs, ys):
        # The symbol a leader runs to from the ink at (xs, ys), and how far its edge lies
        q = ports[i]
        cx, cy = xs[:, None] + q[:, 0], ys[:, None] + q[:, 1]
        gap = np.hypot(cx - P[i, 4], cy - P[i, 5]) - q[:, 2] - radii[i]
        k, row = gap.argmin(1), np.arange(len(xs))
        return cx[row, k], cy[row, k], np.maximum(gap[row, k], 0.0)
    def solid(i, skip = -1):
        # What the ink of drawing i has to keep clear of
        others = np.flatnonzero(placed & (np.arange(n) != i) & (np.arange(n) != skip))
        return others, np.column_stack([pos[others], pos[others] + P[others, 2:4]])
    def held(i, X, Y):
        # What the ink at (X, Y) would hide, leaving out its own anchor, which it may stand right beside
        gx0 = np.clip(((X - 3.0) / step).astype(int) + 1, 0, grid_w)
        gx1 = np.clip(((X + P[i, 2] + 3.0) / step).astype(int) + 1, 0, grid_w)
        gy0 = np.clip(((Y - 3.0) / step).astype(int) + 1, 0, grid_h)
        gy1 = np.clip(((Y + P[i, 3] + 3.0) / step).astype(int) + 1, 0, grid_h)
        own_x, own_y = int(P[i, 4] / step), int(P[i, 5] / step)
        near = ((gx0 <= own_x) & (own_x < gx1) & (gy0 <= own_y) & (own_y < gy1) if
                0 <= P[i, 4] < (grid_w - 1) * step and 0 <= P[i, 5] < (grid_h - 1) * step else False)
        return cover[gy1, gx1] - cover[gy0, gx1] - cover[gy1, gx0] + cover[gy0, gx0] - 150.0 * near
    def place(i):
        # The best spot for one drawing, given all others placed so far
        x0, y0, w, h, ax, ay = P[i]
        others, boxes = solid(i)
        drawn = others[~np.isnan(ends[others, 0])]
        a0, b0, a1, b1 = P[drawn, 4], P[drawn, 5], ends[drawn, 0], ends[drawn, 1]
        low_x, high_x, low_y, high_y = np.minimum(a0, a1), np.maximum(a0, a1), np.minimum(b0, b1), np.maximum(b0, b1)
        px, py = np.delete(P[:, 4], i), np.delete(P[:, 5], i)
        hi_x, hi_y = max(edge, cw - w - edge), max(edge, ch - h - edge)
        # Its natural spot, if free, right by its point, and hiding next to nothing, needs no search
        at_x, at_y = np.array([x0]), np.array([y0])
        if (edge <= x0 <= hi_x and edge <= y0 <= hi_y and port(i, at_x, at_y)[2][0] < 9.0 and
                held(i, at_x, at_y)[0] <= 6.0 and not ((boxes[:, 0] - pad < x0 + w) & (boxes[:, 2] + pad > x0) &
                                                       (boxes[:, 1] - pad < y0 + h) & (boxes[:, 3] + pad > y0)).any()
                and not (len(drawn) and hits_box(a0, b0, a1, b1, x0 - 1.0, y0 - 1.0, x0 + w + 1.0,
                                                 y0 + h + 1.0).any())):
            return x0, y0
        best, best_cost, inner = None, np.inf, -1.0
        reach = 60.0 + max(w, h)
        while True:
            whole = edge >= ax - w - reach and hi_x <= ax + reach and edge >= ay - h - reach and hi_y <= ay + reach
            # Far from its point a drawing needs no fine placing, so a wide search looks at about as many places as a
            # narrow one
            size = ((min(hi_x, ax + reach) - max(edge, ax - w - reach)) *
                    (min(hi_y, ay + reach) - max(edge, ay - h - reach)) / step ** 2)
            pitch = step * max(1, int(max(size / 12000.0, 0.0) ** 0.5))
            xs = x0 + pitch * np.arange(np.ceil((max(edge, ax - w - reach) - x0) / pitch),
                                        np.floor((min(hi_x, ax + reach) - x0) / pitch) + 1)
            ys = y0 + pitch * np.arange(np.ceil((max(edge, ay - h - reach) - y0) / pitch),
                                        np.floor((min(hi_y, ay + reach) - y0) / pitch) + 1)
            xs = xs if len(xs) else np.array([min(max(x0, edge), hi_x)])
            ys = ys if len(ys) else np.array([min(max(y0, edge), hi_y)])
            free = np.ones((len(ys), len(xs)), dtype = bool)
            lefts = np.searchsorted(xs, boxes[:, 0] - pad - w, 'right')
            rights = np.searchsorted(xs, boxes[:, 2] + pad, 'left')
            tops = np.searchsorted(ys, boxes[:, 1] - pad - h, 'right')
            bottoms = np.searchsorted(ys, boxes[:, 3] + pad, 'left')
            for k in np.flatnonzero((lefts < rights) & (tops < bottoms)):
                free[tops[k]:bottoms[k], lefts[k]:rights[k]] = False
            rows, columns = np.nonzero(free)
            X, Y = xs[columns], ys[rows]
            d = np.maximum(np.hypot(np.maximum(np.maximum(X - ax, ax - X - w), 0.0),
                                    np.maximum(np.maximum(Y - ay, ay - Y - h), 0.0)) - radii[i], 0.0)
            # What the last, narrower search looked at needs no second look
            X, Y, d = X[d > inner], Y[d > inner], d[d > inner]
            if len(X):
                base = held(i, X, Y) + 0.02 * np.hypot(X - x0, Y - y0) + 30.0 * ((d < 3.0) & ((X != x0) | (Y != y0)))
                bound = base + d + d * d / 60.0
                # Beyond the best spots, a thin sample tells as much as all of them, as neighbors cross the same things
                order = np.argsort(bound, kind = 'stable')
                order = np.concatenate([order[:256], order[256::max(1, (len(order) - 256) // 512)]])
                for start in range(0, len(order), 256):
                    chunk = order[start:start + 256]
                    if bound[chunk[0]] >= best_cost:
                        break
                    gx, gy, gap = port(i, X[chunk], Y[chunk])
                    cost = base[chunk] + gap + gap * gap / 60.0
                    # What already costs more than the best spot so far needs no further look
                    keep = cost < best_cost
                    if not keep.any():
                        continue
                    chunk, gx, gy, gap, cost = chunk[keep], gx[keep], gy[keep], gap[keep], cost[keep]
                    hx, hy, lead = X[chunk], Y[chunk], gap >= 9.0
                    # Only what lies within reach of these spots and their leaders can meet them
                    x_lo, x_hi = min(hx.min(), ax) - 4.0, max(hx.max() + w, ax) + 4.0
                    y_lo, y_hi = min(hy.min(), ay) - 4.0, max(hy.max() + h, ay) + 4.0
                    seen = (low_x <= x_hi) & (high_x >= x_lo) & (low_y <= y_hi) & (high_y >= y_lo)
                    if seen.any():
                        u0, v0, u1, v1 = a0[seen][:, None], b0[seen][:, None], a1[seen][:, None], b1[seen][:, None]
                        cost = cost + 1000.0 * (crossed(u0, v0, u1, v1, ax, ay, gx, gy) & lead).sum(0)
                        cost = cost + 2000.0 * hits_box(u0, v0, u1, v1, hx - 1.0, hy - 1.0, hx + w + 1.0,
                                                        hy + h + 1.0).sum(0)
                    seen = (boxes[:, 0] <= x_hi) & (boxes[:, 2] >= x_lo) & (boxes[:, 1] <= y_hi) & (boxes[:, 3] >= y_lo)
                    if seen.any():
                        cost = cost + 2000.0 * (hits_box(ax, ay, gx, gy, boxes[seen, 0][:, None] - 1.0,
                                                         boxes[seen, 1][:, None] - 1.0, boxes[seen, 2][:, None] + 1.0,
                                                         boxes[seen, 3][:, None] + 1.0) & lead).sum(0)
                    seen = (px >= x_lo) & (px <= x_hi) & (py >= y_lo) & (py <= y_hi)
                    if seen.any():
                        qx, qy, vx, vy = px[seen][:, None], py[seen][:, None], gx - ax, gy - ay
                        k = np.clip(((qx - ax) * vx + (qy - ay) * vy) / np.maximum(vx * vx + vy * vy, 1e-9), 0, 1)
                        cost = cost + 40.0 * ((np.hypot(ax + k * vx - qx, ay + k * vy - qy) < 3.5) & lead).sum(0)
                    j = int(np.argmin(cost))
                    if cost[j] < best_cost:
                        best, best_cost = (hx[j], hy[j]), cost[j]
            if best is not None and (whole or best_cost <= reach + reach * reach / 60.0):
                return best
            if whole:
                # Nowhere left within the canvas: a column of its own to its right
                return max([cw] + [box[2] for box in boxes]) + pad, max(edge, min(y0, ch - h - edge))
            inner, reach = reach, reach * 2
    def settle(i, x, y):
        # Puts drawing i at (x, y), with its leader
        pos[i], placed[i] = (x, y), True
        ex, ey, gap = port(i, np.array([x]), np.array([y]))
        ends[i] = (ex[0], ey[0]) if gap[0] >= 9.0 else (np.nan, np.nan)
    def tangled():
        # Pairs of drawings whose leaders cross, leaders running through another drawing by (drawing, drawing it runs
        # through), and drawings whose leader runs long
        have = np.flatnonzero(~np.isnan(ends[:, 0]))
        a0, b0, a1, b1 = P[have, 4], P[have, 5], ends[have, 0], ends[have, 1]
        crossing = crossed(a0[:, None], b0[:, None], a1[:, None], b1[:, None], a0, b0, a1, b1)
        through = hits_box(a0[:, None], b0[:, None], a1[:, None], b1[:, None], pos[:, 0], pos[:, 1],
                           pos[:, 0] + P[:, 2], pos[:, 1] + P[:, 3])
        through[np.arange(len(have)), have] = False
        return ([(have[i], have[j]) for i, j in zip(*np.nonzero(np.triu(crossing, 1)))],
                [(have[i], j) for i, j in zip(*np.nonzero(through))],
                have[np.hypot(a1 - a0, b1 - b0) > 0.25 * ch].tolist())
    def score():
        # Tangles first, with drawings past the canvas's right, then how far the leaders run
        return (sum(map(len, tangled())) + int((pos[:, 0] + P[:, 2] > width + 0.5).sum()),
                float(np.hypot(*(ends - P[:, 4:6]).T)[~np.isnan(ends[:, 0])].sum()))
    for _ in range(3):
        moved = False
        for i in sorted(range(n), key = lambda i: (-P[i, 2] * P[i, 3], i)):
            spot = place(i)
            moved |= not (placed[i] and np.allclose(spot, pos[i]))
            settle(i, *spot)
            cw = max(cw, spot[0] + P[i, 2] + edge)
        # Two crossing leaders uncross when their drawings trade places, if each fits where the other stood
        for i, j in tangled()[0]:
            spots = pos[[j, i]] + (P[[j, i], 2:4] - P[[i, j], 2:4]) / 2
            fits = [edge <= x and x + P[k, 2] + edge <= cw and edge <= y and y + P[k, 3] + edge <= ch and not (
                (boxes[:, 0] - pad < x + P[k, 2]) & (boxes[:, 2] + pad > x) & (boxes[:, 1] - pad < y + P[k, 3]) &
                (boxes[:, 3] + pad > y)).any() for k, (x, y), boxes in ((i, spots[0], solid(i, j)[1]),
                                                                       (j, spots[1], solid(j, i)[1]))]
            (xi, yi), (xj, yj) = spots
            if all(fits) and not (xi < xj + P[j, 2] + pad and xj < xi + P[i, 2] + pad and yi < yj + P[j, 3] + pad and
                                  yj < yi + P[i, 3] + pad):
                settle(i, xi, yi)
                settle(j, xj, yj)
                moved = True
        if not moved:
            break
    best = (score(), pos.copy(), ends.copy())
    # Whatever is still tangled, and what stands around a leader too long, starts over among the others, the farthest
    # from its anchor first
    stale = 0
    for attempt in range(4):
        if best[0][0] == 0 or stale == 2:
            break
        pairs, through, far = tangled()
        again = {int(k) for pair in pairs + through for k in pair} | set(far)
        for i in far:
            again |= {int(k) for k in np.argsort(np.hypot(*(pos + P[:, 2:4] / 2 - P[i, 4:6]).T))[:4]}
        again = sorted(again)
        gaps = np.hypot(*(np.nan_to_num(ends, nan = 0.0) - P[:, 4:6]).T) * ~np.isnan(ends[:, 0])
        placed[again], ends[again] = False, np.nan
        for i in sorted(again, key = lambda i: ((-gaps[i] if attempt % 2 == 0 else gaps[i]), i)):
            spot = place(i)
            settle(i, *spot)
            cw = max(cw, spot[0] + P[i, 2] + edge)
        stale += 1
        if (now := score()) < best[0]:
            best, stale = (now, pos.copy(), ends.copy()), 0
    (tangles, _), pos, ends = best
    leaders = [None if np.isnan(ex) else (ax, ay, ex, ey) for (ax, ay), (ex, ey) in zip(P[:, 4:6].tolist(),
                                                                                       ends.tolist())]
    return pos.tolist(), leaders, tangles


class Axes:
    def __init__(self, figure, pal):
        self.figure, self.pal = figure, pal
        self.artists, self.legend_items, self._legend_opts = [], [], None
        self._legend_size = None
        # The artists each legend entry names, by its index, for naming them directly; where the legend's entries go
        # once that is settled, those it still shows, and how far labels at line ends reach past the plot
        self._sources, self._legend_fallback = {}, 'below'
        # The artists each legend entry stands for, by its index, which a highlight and an interactive SVG find by its
        # label: those it names directly and more, such as a line's markers, or bars that are named in a legend
        self._members = {}
        self._legend_loc, self._legend_keep, self._direct_room, self._direct_spots = None, [], 0.0, (None, {})
        self._xlim, self._ylim = None, None
        self.xlabel, self.ylabel, self.title_text = '', '', ''
        self._title_fixed, self._title_pad = None, None
        self._xticks, self._yticks, self._xtl, self._ytl = None, None, None, None
        self._tick_size = None
        self._xfmt, self._yfmt = None, None
        self._label_size, self._label_pad = {'x': None, 'y': None}, {'x': None, 'y': None}
        self._xemph, self._yemph = {}, {}
        self._tick_opts, self._spine_set = {'x': {}, 'y': {}}, {}
        self._xrot, self._yright = 0.0, False
        # How far glyph labels of each axis shrank to fit between their ticks, and what a legend's entries draw as
        self._gscale, self._legend_glyphs = {'x': 1.0, 'y': 1.0}, None
        # Glyph labels of y in one column, or staggered in two where neighbors would crowd
        self._glyph_spots, self._gcols, self._full = (None, {}), 1, {'x': ([], []), 'y': ([], [])}
        # The last placement of spread glyphs made in full: their notes, their points on the page, and their spots,
        # from which a window's gesture moves them
        self._glyph_settled = ((), [], {})
        # Which points the density fields hide, for the page as laid out (see hidden)
        self._hidden_points = (None, {})
        self._xscale, self._yscale = 'linear', 'linear'
        self._xwarp, self._ywarp = WARPS['linear'], WARPS['linear']
        # Whether an axis's tick labels are numbers, set in tabular figures and lined up on their decimal points
        self._numbers = {'x': False, 'y': False}
        self._xinv, self._yinv = False, False
        self._cycle = 0
        self._cats = {'x': {}, 'y': {}}
        # Where a box, strip, or violin of hue sits, by (category, hue level), for brackets
        self._dodged = {}
        self._hidden = set()
        self._xmargin, self._ymargin = 0.05, 0.05
        self._scale, self._cbar = None, None
        # How far brackets stretch the y axis upward on the page, from its other end, in its own scale
        self._lift = 1.0
        self._over, self._over_plain = [0.0, 0.0], -math.inf
        self._ridge_labels = None
        self._fixed, self._cell, self._span, self._legend_y, self._align = None, None, (1, 1), None, {}
        self.rect = self._room = (0.0, 0.0, 1.0, 1.0)
        # Axes whose limits these share, and whether their tick labels and name stay hidden as an inner panel's
        self._share, self._outer = {'x': [self], 'y': [self]}, {'x': False, 'y': False}
        self.grid_on = pal.grid
        self.show_all = (False, False)
        # What a highlight keeps in full shade, for artists whose own call set none
        self._focus = None

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
        # A DataFrame's column of numbers can arrive as objects
        if arr.dtype == object and all(isinstance(v, numbers.Real) for v in arr.ravel()):
            return arr.astype(float)
        if arr.dtype.kind in 'US' or arr.dtype == object:
            cats = self._cats[axis]
            for v in arr.ravel():
                cats.setdefault(str(v), len(cats))
            return np.array([cats[str(v)] for v in arr.ravel()], dtype = float).reshape(arr.shape)
        return arr.astype(float)

    def _add(self, kind, z, **kw):
        self.artists.append(dict(kind = kind, z = z, order = len(self.artists), **kw))

    def _labeled(self, label, rgb, kind = 'patch', marker = 'o', area = 36.0, source = None, members = None):
        """A legend entry; `marker` holds a point's shape, or a line's dash, `source` the artists it names, where a
        label beside them can name them instead, and `members` all the artists it stands for, by default `source`."""
        if label is not None and rgb is not None:
            self.legend_items.append((str(label), rgb, kind, marker, area))
            if source:
                self._sources[len(self.legend_items) - 1] = source
            if members or source:
                self._members[len(self.legend_items) - 1] = members or source

    # --- plotting ---
    def plot(self, x, y = None, color = None, lw = None, label = None, marker = False, alpha = 1.0, dash = None,
             ms = 42, z = 4, clip = True, y_fraction = False, offset = None):
        """A line, with markers at its points if `marker` is True or a marker name; a NaN breaks it. `y_fraction` reads
        y as fractions of the axes' height and `offset` moves each point by (dx, dy) points, which together tie a line
        to a position beyond the data. With clip = False it may leave the axes, and shows while all its x are within the
        x limits."""
        x, y = (np.arange(len(x)), x) if y is None else (x, y)
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        rgb = self._next_color(color)
        self._add('line', z, x = x, y = y, rgb = rgb, lw = lw, alpha = alpha, dash = dash, clip = clip,
                  frac = y_fraction, offset = None if offset is None else np.asarray(offset, dtype = float))
        line = self.artists[-1]
        if marker:
            marker = 'o' if marker is True else marker
            _check_marker(marker)
            self._add('points', z + 1, x = x, y = y, s = ms, rgb = rgb, alpha = alpha, marker = marker, edge = None,
                      gid = None)
        self._labeled(label, rgb, 'line', dash, source = [line],
                      members = [line] + ([self.artists[-1]] if marker else []))
        return self

    def step(self, x, y, where = 'pre', color = None, lw = None, label = None, alpha = 1.0, dash = None, z = 4,
             clip = True):
        """A line of steps, as in matplotlib, for counts, cumulative distributions, and survival curves: with
        where = 'pre' each y holds from the x before it up to its own, with 'post' from its own x up to the next, and
        with 'mid' between the midpoints around it."""
        if where not in ('pre', 'post', 'mid'):
            raise ValueError(f"A step's where has to be 'pre', 'post', or 'mid', got {where!r}")
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        if where == 'mid' and x.size > 1:
            edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
            xs, ys = np.repeat(edges, 2)[1:-1], np.repeat(y, 2)
        elif where == 'pre':
            # Each corner repeats a point, taking its x from one neighbor and its y from the other
            xs, ys = np.repeat(x, 2)[:-1], np.repeat(y, 2)[1:]
        else:
            xs, ys = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
        return self.plot(xs, ys, color = color, lw = lw, label = label, alpha = alpha, dash = dash, z = z, clip = clip)

    def scatter(self, x, y, s = 30, color = None, c = None, label = None, alpha = 1.0, cmap = 'sequential', vmin = None,
                vmax = None, z = 3, marker = 'o', edge = None, gid = None, hue = None, style = None, size = None,
                sizes = (20.0, 160.0), colors = None, markers = None, hue_label = None, style_label = None,
                size_label = None, legend = 'direct', names = None, rasterized = False, density = True,
                highlight = None):
        """Points, as in seaborn's scatterplot. `hue`, `style`, and `size` hold one value per point and map to color,
        marker shape, and area. Each color's points are named in its color at their middle, where they hold together
        apart from the others, and the rest in a legend at the right, or all of it in a legend at `legend`, a legend's
        place. Numeric `hue` with many values, or `c`, maps through a palette scale;
        call `colorbar` to show it. `colors` and `markers` are lists or {value: name} dicts. `edge` is (color, width),
        which with color = 'none' draws hollow rings. `names` holds one label per point, which draws nothing but tells a
        viewer such as slate what the point under the mouse is. `rasterized` makes all the points one image in PDF and
        SVG, at 300 dpi for True or at the dpi given, which keeps tens of thousands of points light, while axes and text
        stay vector. Where points pile up past telling them apart, a map of their density takes over from them, in bands
        of their color's shades set apart by hairlines of the page, while points stay wherever the data are sparse;
        `density = False` keeps every point, as does a color scale from `c`, whose colors are data. `highlight` keeps
        the hue levels or points it names, by their `names`, in full shade, and the rest recede (see Axes.highlight).
        """
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(self._pos(y, 'y'))
        _check_lengths(x, y)
        if names is not None:
            names = np.atleast_1d(np.asarray(names, dtype = object))
            if names.shape != x.shape:
                raise ValueError(f'names has to hold one label per point, got {names.size} for {x.size} points')
        _check_marker(marker)
        edge = None if edge is None else (self.pal.color(edge[0]), edge[1])
        # The resolution, and the call's first artist, which draws the image for all of them
        raster = (300.0 if rasterized is True else float(rasterized), len(self.artists)) if rasterized else None
        if hue is not None and _numeric(hue) and len(np.unique(hue)) > 8:
            c, hue = hue, None
        # The call's first artist, which lays the fields of where its points pile up under all of them
        crowd = len(self.artists) if density and c is None else None
        if hue is None and style is None and size is None and c is None:
            rgb = self._next_color(color)
            self._add('points', z, x = x, y = y, s = s, rgb = rgb, alpha = alpha, marker = marker, edge = edge,
                      gid = gid, names = names, raster = raster, crowd = crowd, focus = _keyset(highlight))
            self._labeled(label, rgb, 'point', marker, source = [self.artists[-1]])
            return self
        n = len(x)
        if c is not None:
            _check_cmap(cmap)
            c = np.broadcast_to(np.asarray(c, dtype = float), x.shape)
            known = c[np.isfinite(c)]
            lo = vmin if vmin is not None else known.min() if known.size else 0.0
            hi = vmax if vmax is not None else known.max() if known.size else 1.0
            levels = np.round((c - lo) / (hi - lo if hi > lo else 1.0) * 24)
            self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
        hues, styles = (np.array([''] * n if v is None else [str(u) for u in v], dtype = object) for v in (hue, style))
        hue_levels, style_levels = list(dict.fromkeys(hues)), list(dict.fromkeys(styles))
        if isinstance(colors, dict):
            hue_rgb = {k: self.pal.color(colors[k]) for k in hue_levels}
        elif colors is not None:
            hue_rgb = dict(zip(hue_levels, [self.pal.color(k) for k in colors]))
        elif hue is not None:
            hue_rgb = dict(zip(hue_levels, self.pal.contrast(len(hue_levels))))
        else:
            hue_rgb = {'': self.pal.color(self.pal.muted) if c is not None else self._next_color(color)}
        if style is None:
            style_marker = {'': marker}
        elif isinstance(markers, dict):
            style_marker = {k: markers[k] for k in style_levels}
        else:
            order = list(marks.MARKERS) if markers is None else list(markers)
            style_marker = {k: order[i % len(order)] for i, k in enumerate(style_levels)}
        for v in style_marker.values():
            _check_marker(v)
        area = np.broadcast_to(np.asarray(s, dtype = float), x.shape)
        size_key = []
        if size is not None:
            if _numeric(size):
                v = np.asarray(size, dtype = float)
                low, high = np.nanmin(v), np.nanmax(v)
                span = (sizes[1] - sizes[0]) / (high - low if high > low else 1.0)
                scale = lambda u: sizes[0] + (np.asarray(u) - low) * span
                area = scale(v)
                ticks = nice_ticks(float(low), float(high), 4)
                size_key = [(name, float(scale(t))) for name, t in zip(tick_labels(ticks), ticks)]
            else:
                # Not `names`, which holds the points' own labels
                kinds = list(dict.fromkeys(str(u) for u in size))
                steps = np.linspace(sizes[0], sizes[1], max(len(kinds), 2))
                area = np.array([steps[kinds.index(str(u))] for u in size], dtype = float)
                size_key = [(k, float(steps[i])) for i, k in enumerate(kinds)]
        by_hue = {}
        for h in hue_levels:
            for st in style_levels:
                m = (hues == h) & (styles == st)
                # A mapped color makes one group per step of its scale
                groups = [(m, hue_rgb[h])] if c is None else [
                    (m & (levels == level), tuple(getattr(self.pal, cmap)(level / 24)))
                    for level in np.unique(levels[m & np.isfinite(levels)])]
                for chosen, rgb in groups:
                    if chosen.any():
                        self._add('points', z, x = x[chosen], y = y[chosen], s = area[chosen], rgb = rgb, alpha = alpha,
                                  marker = style_marker[st], edge = edge, gid = gid,
                                  names = None if names is None else names[chosen], raster = raster, crowd = crowd,
                                  mapped = c is not None, focus = _keyset(highlight))
                        by_hue.setdefault(h, []).append(self.artists[-1])
        sections = (hue is not None) + (style is not None) + (size is not None)
        plain = self.pal.color(self.pal.muted) if hue is not None else hue_rgb['']
        if hue is not None:
            if sections > 1:
                self.legend_items.append((hue_label or 'Color', None, 'header', 'o', 0.0))
            for h in hue_levels:
                self._labeled(h, hue_rgb[h], 'point', marker, source = by_hue.get(h))
        if style is not None:
            if sections > 1:
                self.legend_items.append((style_label or 'Shape', None, 'header', 'o', 0.0))
            for st in style_levels:
                self._labeled(st, plain, 'point', style_marker[st])
        if size is not None:
            if sections > 1:
                self.legend_items.append((size_label or 'Size', None, 'header', 'o', 0.0))
            for name, a in size_key:
                self._labeled(name, plain, 'point', marker, a)
        if legend and sections:
            self.legend('right' if legend is True else legend)
            self._legend_fallback = 'right'
        return self

    def bar(self, x, height, width = None, bottom = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
            z = 3, highlight = None):
        """Vertical bars; several colors give one bar each, and string positions become category labels. `width` can
        hold one value per bar. `highlight` keeps the categories it names in full shade, or all with the bars' label,
        and the rest recede (see Axes.highlight)."""
        x = np.atleast_1d(self._pos(x, 'x'))
        if np.size(height) not in (1, x.size):
            raise ValueError(f'bar needs one height per bar, got {x.size} bars and {np.size(height)} heights')
        h = np.broadcast_to(np.asarray(height, dtype = float), x.shape)
        b = np.broadcast_to(np.asarray(bottom, dtype = float), x.shape)
        width = np.broadcast_to(np.asarray(self.pal.bar_width if width is None else width, dtype = float), x.shape)
        cols = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(x)
        self._add('bars', z, x = x, h = h, b = b, w = width, rgb = cols, alpha = alpha, horizontal = False,
                  stacked = stacked, focus = _keyset(highlight))
        self._labeled(label, cols[0] if cols else None, source = [self.artists[-1]] if stacked else None,
                      members = [self.artists[-1]])
        return self

    def barh(self, y, width, height = None, left = 0.0, color = None, label = None, alpha = 1.0, stacked = False,
             z = 3, highlight = None):
        """Horizontal bars; `height` can hold one value per bar, and `highlight` is as in bar."""
        y = np.atleast_1d(self._pos(y, 'y'))
        if np.size(width) not in (1, y.size):
            raise ValueError(f'barh needs one width per bar, got {y.size} bars and {np.size(width)} widths')
        w = np.broadcast_to(np.asarray(width, dtype = float), y.shape)
        b = np.broadcast_to(np.asarray(left, dtype = float), y.shape)
        height = np.broadcast_to(np.asarray(self.pal.bar_width if height is None else height, dtype = float), y.shape)
        cols = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(y)
        self._add('bars', z, x = y, h = w, b = b, w = height, rgb = cols, alpha = alpha, horizontal = True,
                  stacked = stacked, focus = _keyset(highlight))
        self._labeled(label, cols[0] if cols else None, members = [self.artists[-1]])
        return self

    def stacked_bar(self, x, values, labels = None, colors = None, width = None, highlight = None):
        """Bars stacked from a (bars, segments) array. Pass colors = pal.groups(3, 2) to group segments by hue.
        `highlight` keeps the bars it names by category, or the segments it names by label, in full shade (see
        Axes.highlight)."""
        values = np.asarray(values, dtype = float)
        if values.ndim != 2 or values.shape[0] != np.size(x):
            raise ValueError(f'stacked_bar needs a (bars, segments) array with a row per bar, got {np.size(x)} bars '
                             f'and an array of shape {values.shape}')
        colors = self.pal.colors(values.shape[1]) if colors is None else [self.pal.color(k) for k in colors]
        # A segment needs more room than a needle to show its color
        width = (0.42 if self.pal.bar == 'needle' else self.pal.bar_width) if width is None else width
        base = np.zeros(values.shape[0])
        for j in range(values.shape[1]):
            self.bar(x, values[:, j], bottom = base, color = colors[j], width = width,
                     label = labels[j] if labels is not None else None, stacked = True, highlight = highlight)
            base = base + values[:, j]
        return self

    def hist(self, x, bins = 20, color = None, label = None, density = False, alpha = 1.0, frequency = False):
        """A histogram, with its bars touching except for the palette's gap. `density` scales it to an area of one, and
        `frequency` gives counts per unit of x."""
        x = np.asarray(x, dtype = float)
        counts, edges = np.histogram(x[np.isfinite(x)], bins = bins, density = density)
        n = len(self.legend_items)
        self.bar(edges[:-1] + np.diff(edges) / 2, counts / np.diff(edges) if frequency else counts,
                 width = np.diff(edges) * 0.94, color = color, label = label, alpha = alpha)
        # Named over its highest bar
        if len(self.legend_items) > n:
            self._sources[n] = self._members[n] = [self.artists[-1]]
        return self

    def fill_between(self, x, y1, y2 = 0.0, color = None, alpha = 0.22, label = None, z = 2):
        """A band between two curves; the band behind a line takes the line's color."""
        x = self._pos(x, 'x')
        y1 = np.broadcast_to(np.asarray(y1, dtype = float), x.shape)
        y2 = np.broadcast_to(np.asarray(y2, dtype = float), x.shape)
        rgb = self._next_color(color)
        self._add('band', z, x = x, y1 = y1, y2 = y2, rgb = rgb, alpha = alpha)
        self._labeled(label, rgb, source = [self.artists[-1]])
        return self

    def contour(self, x, y, values, levels = 5, color = 'muted', lw = 0.7, alpha = 1.0, dash = None, label = None,
                z = 2):
        """Lines of equal value over a grid, where `values` has one row per y and one column per x. `levels` is how many
        evenly spaced levels to draw between the extremes, or a list of levels."""
        x, y, m = np.asarray(x, dtype = float), np.asarray(y, dtype = float), np.asarray(values, dtype = float)
        if m.shape != (len(y), len(x)):
            raise ValueError(f'values have to have one row per y and one column per x, got {m.shape} for {len(y)} y '
                             f'and {len(x)} x')
        known = np.isfinite(m)
        if np.ndim(levels) == 0:
            levels = np.linspace(m[known].min(), m[known].max(), int(levels) + 2)[1:-1] if known.any() else []
        # Grid units to data along each axis, where a straight line between grid points says
        lines = [(np.interp(u, np.arange(len(x)), x), np.interp(v, np.arange(len(y)), y)) for level in levels
                 for u, v in marks.isolines(m, level)]
        rgb = self.pal.color(color)
        self._add('contour', z, x = x, y = y, lines = lines, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        self._labeled(label, rgb, 'line', dash)
        return self

    def errorbar(self, x, y, yerr, color = 'ink', lw = 1.2, capsize = 3.0, z = 5):
        """Error bars around y; `yerr` is one value, one per point, or two rows of those, for below and above."""
        x, y = np.atleast_1d(self._pos(x, 'x')), np.atleast_1d(np.asarray(y, dtype = float))
        _check_lengths(x, y)
        below, above = yerr if np.ndim(yerr) == 2 else (yerr, yerr)
        self._add('errors', z, x = x, lo = y - np.broadcast_to(np.asarray(below, dtype = float), y.shape),
                  hi = y + np.broadcast_to(np.asarray(above, dtype = float), y.shape), rgb = self.pal.color(color),
                  lw = lw, cap = capsize)
        return self

    def _groups(self, data, x, y, hue, names, order, hue_order, orient, positions, labels, colors, whis, width, kind,
                legend):
        """Groups of values for box, strip, and violin plots, as a dict of their positions, widths, colors, values,
        names, and stats, each group's (category, hue level), and whether they lie across the page. Data comes as a
        list of value lists, a {label: values} dict, a {label: {hue: values}} dict, or in long form, where `x` and
        `y` name columns of a table such as a DataFrame, or hold one entry per value themselves, and so do `hue` and
        `names`. Values run along the numeric one of x and y, unless `orient` says otherwise. Categories and hue
        levels follow `order` and `hue_order`, else a categorical's own order, else numbers by size, else first
        appearance; a category takes its place on the axis as any category does, so that panels sharing the axis
        place it alike. With hue, a category's groups stand side by side across 0.8 of it, each as wide as 0.8 of its
        share or as `width`, whichever is less, with a legend of the levels. Stats per group are its lower whisker,
        quartile, median, quartile, upper whisker, and the values beyond the whiskers, or None for a group without
        values."""
        if orient not in (None, 'v', 'h'):
            raise ValueError(f"orient has to be 'v' or 'h', got {orient!r}")
        if x is not None or y is not None:
            column = lambda v: data[v] if isinstance(v, str) and data is not None else v
            xs, ys = (None if v is None else column(v) for v in (x, y))
            if orient is None:
                orient = 'h' if ys is None or xs is not None and _numeric(xs) and not _numeric(ys) else 'v'
            cats, vals = (ys, xs) if orient == 'h' else (xs, ys)
            if vals is None:
                raise ValueError(f"Long-form data needs its values as {'x' if orient == 'h' else 'y'}")
            vals = _floats(vals)
            given = dict(categories = cats, hue = None if hue is None else column(hue),
                         names = None if names is None else column(names))
            for what, v in given.items():
                if v is not None and np.size(v) != len(vals):
                    raise ValueError(f'{what} has to hold one entry per value, got {np.size(v)} for {len(vals)} values')
            cat_of, cat_levels = _levels(cats, order) if cats is not None else (np.full(len(vals), '', object), [''])
            hue_of, hue_levels = (_levels(given['hue'], hue_order) if hue is not None else
                                  (np.full(len(vals), '', object), ['']))
            tags = None if names is None else np.asarray(given['names'], dtype = object).ravel()
            parts = [(vals[m], None if tags is None else tags[m]) for c in cat_levels for h in hue_levels
                     for m in [(cat_of == c) & (hue_of == h)]]
            labeled, nested = cats is not None, hue is not None
        else:
            if hue is not None:
                raise ValueError('hue takes long-form data, with x and y, or a {category: {hue: values}} dict')
            if isinstance(data, dict):
                by = {str(k): v for k, v in data.items()}
                keys = [str(k) for k in (order if order is not None else data)]
                cat_levels = keys if labels is None else [str(k) for k in labels]
                groups = [by[k] for k in keys]
                named = {str(k): v for k, v in names.items()} if isinstance(names, dict) else None
                tags = names if named is None else [named.get(k) for k in keys]
            else:
                groups = list(data)
                cat_levels = list(range(len(groups))) if labels is None else [str(k) for k in labels]
                tags = names
            tags = [None] * len(groups) if tags is None else list(tags)
            if len(tags) != len(groups):
                raise ValueError(f'names has to hold one list per group, got {len(tags)} for {len(groups)} groups')
            labeled, nested = isinstance(data, dict) or labels is not None, any(isinstance(g, dict) for g in groups)
            hue_levels = (([str(k) for k in hue_order] if hue_order is not None else
                           list(dict.fromkeys(str(k) for g in groups if isinstance(g, dict) for k in g)))
                          if nested else [''])
            parts = []
            for g, t in zip(groups, tags):
                g, t = ({str(k): v for k, v in g.items()}, {str(k): v for k, v in (t or {}).items()}) if nested else (
                    {'': g}, {'': t})
                parts += [(g.get(h, []), t.get(h)) for h in hue_levels]
        horizontal, axis = orient == 'h', 'y' if orient == 'h' else 'x'
        values, tags, stats = [], [], []
        for v, t in parts:
            v = _floats(v)
            if t is not None and np.size(t) != len(v):
                raise ValueError(f'names has to hold one label per value, got {np.size(t)} for {len(v)} values')
            ok = np.isfinite(v)
            v = v[ok]
            values.append(v)
            tags.append(None if t is None else np.asarray(t, dtype = object).ravel()[ok])
            if not v.size:
                stats.append(None)
                continue
            q1, median, q3 = np.percentile(v, (25, 50, 75))
            # With short whiskers a few values can leave none within reach, and then they end at the box
            within = v[(v >= q1 - whis * (q3 - q1)) & (v <= q3 + whis * (q3 - q1))]
            low, high = (within.min(), within.max()) if within.size else (q1, q3)
            stats.append((low, q1, median, q3, high, v[(v < low) | (v > high)]))
        if positions is not None:
            base = np.asarray(positions, dtype = float)
            if labeled:
                (self.set_yticks if horizontal else self.set_xticks)(base, cat_levels)
        elif labeled:
            base = self._pos(np.array(cat_levels, dtype = object), axis)
        else:
            base = np.arange(len(cat_levels), dtype = float)
        n = len(hue_levels)
        slot = 0.8 / max(n, 1)
        spots = np.array([b + (j - (n - 1) / 2) * slot * nested for b in base for j in range(n)], dtype = float)
        if nested and labeled:
            self._dodged.update({(c, h): float(b + (j - (n - 1) / 2) * slot) for c, b in zip(cat_levels, base)
                                 for j, h in enumerate(hue_levels)})
        keys = hue_levels if nested else cat_levels
        if isinstance(colors, dict):
            colors = {str(k): v for k, v in colors.items()}
            shades = [self.pal.color(colors[str(k)]) for k in keys]
        else:
            shades = self.pal.contrast(len(keys)) if colors is None else [self.pal.color(k) for k in colors]
        rgb = [shades[(j if nested else i) % len(shades)] for i in range(len(cat_levels)) for j in range(n)]
        if nested and legend:
            entries = [(hue, None, 'header', 'o', 0.0)] if isinstance(hue, str) else []
            entries += [(h, shades[j % len(shades)], kind, 'o', 36.0) for j, h in enumerate(hue_levels)]
            # A plot drawn over another of the same groups, such as points over boxes, adds no entries twice
            self.legend_items += [e for e in entries if e not in self.legend_items]
            self.legend('right' if legend is True else legend)
        if horizontal and positions is None:
            # The first group on top, where reading starts
            self._yinv = True
        return dict(x = spots, w = np.full(len(spots), min(width, 0.8 * slot) if nested else float(width)), rgb = rgb,
                    values = values, names = None if all(t is None for t in tags) else tags, stats = stats,
                    horizontal = horizontal, groups = [(str(c), h) for c in cat_levels for h in hue_levels])

    def _strip(self, values, spread, size):
        """Each group's points spread across a share of its width, `spread` per value, by a golden-ratio sequence over
        the values in order, so that values close in size lie apart, and the same on every draw."""
        offsets = []
        for v, share in zip(values, spread):
            u = np.empty(len(v))
            u[np.argsort(v, kind = 'stable')] = (np.arange(len(v)) * 0.6180339887498949 + 0.5) % 1.0 - 0.5
            offsets.append(u * share)
        # Many points get smaller, so that a crowd still reads as points
        sizes = [float(size) if size is not None else 16.0 * min(1.0, math.sqrt(150.0 / max(len(v), 1))) + 2.0
                 for v in values]
        return offsets, sizes

    def boxplot(self, data = None, positions = None, labels = None, width = 0.5, colors = None, whis = 1.5,
                fliers = True, alpha = 1.0, z = 3, points = True, jitter = 0.7, point_size = None, x = None, y = None,
                hue = None, names = None, order = None, hue_order = None, orient = None, swarm = False,
                legend = 'right', highlight = None):
        """Box plots of several groups, in the palette's manner, as in seaborn: `data` is a list of value lists, a
        {label: values} dict, a {label: {hue: values}} dict, or a table such as a DataFrame whose columns `x`, `y`, and
        `hue` name, as in boxplot(df, x = 'stage', y = 'abundance', hue = 'condition'); x and y can also hold the
        values themselves. With hue, each category's boxes stand side by side, with a legend of the levels at `legend`
        (False for none). orient = 'h' lays the boxes across the page with the categories down the y axis, first on
        top, which is what long names read best in, and comes by itself when x holds the numbers and y the labels.
        A box spans the quartiles with its median marked, and whiskers reach the last value within `whis` box heights.
        With `points`, every value shows as a point over its box, spread across `jitter` of its width, or with `swarm`
        packed side by side so that none overlap, and the box takes a paler shade; without, `fliers` shows the values
        beyond the whiskers. `names` holds a label per value, in the shape of the data or as a column, which draws
        nothing but tells a viewer such as slate which sample a point is. Groups sit at 0, 1, 2, and so on, unless
        `positions` says otherwise; `order` and `hue_order` choose and order categories and levels. `colors` are per
        group, or per hue level with hue, as a list or a {label: color} dict. `highlight` keeps the categories, hue
        levels, or values it names, by `names`, in full shade, and the rest recede (see Axes.highlight)."""
        g = self._groups(data, x, y, hue, names, order, hue_order, orient, positions, labels, colors, whis, width,
                         'patch', legend)
        offsets, sizes = self._strip(g['values'], g['w'] * jitter, point_size)
        self._add('box', z, **g, alpha = alpha, fliers = fliers, offsets = offsets if points else None, sizes = sizes,
                  box = True, swarm = bool(swarm and points), focus = _keyset(highlight))
        return self

    def stripplot(self, data = None, positions = None, labels = None, width = None, colors = None, alpha = 1.0, z = 3,
                  jitter = 0.7, point_size = None, x = None, y = None, hue = None, names = None, order = None,
                  hue_order = None, orient = None, swarm = False, legend = 'right', highlight = None):
        """Every value of several groups as a point, spread across `jitter` of the group's `width` so that equal values
        stay apart, as in seaborn's stripplot, or with `swarm` packed side by side within the width so that none
        overlap, as in its swarmplot, which reads best for 20 to 100 values a group. The width is 0.5, or 0.8 for a
        swarm, which needs the room, unless given. Data, hue, orientation, names, colors, and highlight are as in
        boxplot."""
        width = (0.8 if swarm else 0.5) if width is None else width
        g = self._groups(data, x, y, hue, names, order, hue_order, orient, positions, labels, colors, 1.5, width,
                         'point', legend)
        offsets, sizes = self._strip(g['values'], g['w'] * jitter, point_size)
        self._add('box', z, **g, alpha = alpha, fliers = False, offsets = offsets, sizes = sizes, box = False,
                  swarm = bool(swarm), focus = _keyset(highlight))
        return self

    def violinplot(self, data = None, positions = None, labels = None, width = 0.8, colors = None, alpha = 1.0, z = 3,
                   inner = 'box', bw_adjust = 1.0, cut = 2.0, point_size = None, x = None, y = None, hue = None,
                   names = None, order = None, hue_order = None, orient = None, legend = 'right',
                   density_norm = 'area', highlight = None):
        """Violin plots of several groups, in the palette's manner: each group's density, by a Gaussian kernel of
        Scott's bandwidth times `bw_adjust`, mirrored around its position and running `cut` bandwidths past its
        extremes, as in seaborn. `density_norm` scales them as seaborn does: 'area' gives all one scale, so that each
        holds the same area, 'count' an area in proportion to its number of values, and 'width' each the same width,
        which keeps every violin readable beside a narrow one. The widest is `width` across. `inner` is 'box' for the
        quartiles and median inside, 'points' for every value spread across the violin's width at its height, 'swarm'
        for every value packed side by side without overlap, or None. Data, hue, orientation, names, colors, and
        highlight are as in boxplot."""
        if inner not in ('box', 'points', 'swarm', None):
            raise ValueError(f"A violin's inner has to be 'box', 'points', 'swarm', or None, got {inner!r}")
        if density_norm not in ('area', 'count', 'width'):
            raise ValueError(f"density_norm has to be 'area', 'count', or 'width', got {density_norm!r}")
        g = self._groups(data, x, y, hue, names, order, hue_order, orient, positions, labels, colors, 1.5, width,
                         'patch', legend)
        shapes = []
        for v in g['values']:
            # A single value or equal ones have no spread to draw
            bw = np.std(v, ddof = 1) * len(v) ** -0.2 * bw_adjust if len(v) > 1 else 0.0
            if not bw > 0:
                shapes.append(None)
                continue
            grid = np.linspace(v.min() - cut * bw, v.max() + cut * bw, 128)
            spots, weights = np.unique(v, return_counts = True)
            if len(spots) > 5000:
                # Many values go through a histogram first, far finer than the kernel
                weights, edges = np.histogram(v, 2000)
                spots = (edges[:-1] + edges[1:]) / 2
            density = np.zeros(len(grid))
            for start in range(0, len(spots), 2000):
                d = (grid[:, None] - spots[None, start:start + 2000]) / bw
                density += (np.exp(-0.5 * d * d) * weights[start:start + 2000]).sum(axis = 1)
            shapes.append((grid, density / (len(v) * bw * math.sqrt(2 * math.pi))))
        heights = [None if s is None else s[1] * (len(v) if density_norm == 'count' else
                                                  1.0 / s[1].max() if density_norm == 'width' else 1.0)
                   for s, v in zip(shapes, g['values'])]
        peak = max([u.max() for u in heights if u is not None], default = 1.0)
        shapes = [None if s is None else (s[0], u / peak * w / 2) for s, u, w in zip(shapes, heights, g['w'])]
        unit, sizes = self._strip(g['values'], [1.0] * len(g['values']), point_size)
        # Spread as wide as the violin is at each value, within it, and as a strip's where it has no shape
        offsets = [u * 0.7 * w if s is None else u * 1.6 * np.interp(v, s[0], s[1]) if inner == 'points' else u
                   for u, v, s, w in zip(unit, g['values'], shapes, g['w'])]
        self._add('violin', z, **g, shapes = shapes, alpha = alpha, inner = inner, offsets = offsets, sizes = sizes,
                  swarm = inner == 'swarm', focus = _keyset(highlight))
        return self

    def stem(self, x, y, color = None, lw = 3.0, label = None, cap = False, z = 4, names = None, highlight = None):
        """Vertical sticks from zero, as in a mass spectrum. `names` labels each stick, which draws nothing but names it
        under the mouse, and `highlight` keeps the sticks it names in full shade while the rest recede (see
        Axes.highlight)."""
        x, y = np.atleast_1d(np.asarray(x, dtype = float)), np.atleast_1d(np.asarray(y, dtype = float))
        _check_lengths(x, y)
        _check_names(names, x.size)
        rgb = self._next_color(color)
        self._add('stems', z, x = x, y = y, rgb = rgb, lw = lw, cap = cap, names = names, focus = _keyset(highlight))
        self._labeled(label, rgb, members = [self.artists[-1]])
        return self

    def hlines(self, y, x0, x1, color = None, lw = 1.5, alpha = 1.0, dash = None, label = None, z = 2):
        """Horizontal segments, under the points, such as the intervals of a forest plot."""
        y = np.atleast_1d(np.asarray(y, dtype = float))
        x0 = np.broadcast_to(np.asarray(x0, dtype = float), y.shape)
        x1 = np.broadcast_to(np.asarray(x1, dtype = float), y.shape)
        rgb = self._next_color(color)
        self._add('hsegs', z, y = y, x0 = x0, x1 = x1, rgb = rgb, lw = lw, alpha = alpha, dash = dash)
        self._labeled(label, rgb, 'line', dash)
        return self

    def vlines(self, x, y0, y1, color = None, lw = 1.0, alpha = 1.0, dash = None, label = None, z = 3, names = None,
               highlight = None):
        """Vertical segments of exactly this width, as many sticks need; `stem` draws them in the palette's manner. A
        list of colors gives each its own. `names` and `highlight` are as in stem, such as for a spectrum's peaks."""
        x = np.atleast_1d(np.asarray(x, dtype = float))
        y0 = np.broadcast_to(np.asarray(y0, dtype = float), x.shape)
        y1 = np.broadcast_to(np.asarray(y1, dtype = float), x.shape)
        _check_names(names, x.size)
        rgb = [self.pal.color(k) for k in color] if isinstance(color, list) else [self._next_color(color)] * len(x)
        self._add('vsegs', z, x = x, y0 = y0, y1 = y1, rgb = rgb, lw = lw, alpha = alpha, dash = dash, names = names,
                  focus = _keyset(highlight))
        self._labeled(label, rgb[0] if rgb else None, 'line', dash)
        return self

    def axvspan(self, x0, x1, color = 'a2', alpha = 0.25, label = None, z = 1):
        """A band across the axes' height between two x values."""
        rgb = self.pal.color(color)
        self._add('vspan', z, x0 = float(x0), x1 = float(x1), rgb = rgb, alpha = alpha)
        self._labeled(label, rgb)
        return self

    def axhspan(self, y0, y1, color = 'a2', alpha = 0.25, label = None, z = 1):
        """A band across the axes' width between two y values."""
        rgb = self.pal.color(color)
        self._add('hspan', z, y0 = float(y0), y1 = float(y1), rgb = rgb, alpha = alpha)
        self._labeled(label, rgb)
        return self

    def image(self, pixels, x, y, width, height, offset = (0.0, 0.0), y_fraction = False, z = 6):
        """An array of rows and columns, gray or with RGB or RGBA channels, first row at the top, in integers up to 255
        or floats up to 1, with its center at a data position, moved by `offset` points. `width` and `height` are in
        points. `y_fraction` reads y as a fraction of the axes' height. It may leave the axes, and shows while its
        position is within the limits."""
        px = np.asarray(pixels)
        px = (np.clip(px, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if px.dtype.kind == 'f' else px.astype(np.uint8)
        px = np.dstack([px] * 3) if px.ndim == 2 else px
        alpha = px[..., 3].tobytes() if px.shape[2] == 4 else None
        self._add('image', z, x = float(x), y = float(y), w = float(width), h = float(height), dx = offset[0],
                  dy = offset[1], frac = y_fraction, nx = px.shape[1], ny = px.shape[0],
                  rgb = px[..., :3].tobytes(), alpha = alpha)
        return self

    def arrow(self, x0, y0, x1, y1, color = 'muted', lw = 1.0, alpha = 1.0, gid = None, z = 2, head = None,
              shrink = (0.0, 0.0), bend = 0.0, name = None):
        """An arrow between two data points; `gid` names it in SVG output, for hover text. `head` is the head's length
        in points, or 0 for a plain leader line, `shrink` how many points it stays short of its start and its end, as
        beside a marker or a label, and `bend` turns it into an arc whose middle stands that share of its length to its
        left. `name` is what it stands for, such as a reaction, which an interactive SVG shows under the mouse, lighting
        every arrow of that name."""
        self._add('arrow', z, x0 = float(x0), y0 = float(y0), x1 = float(x1), y1 = float(y1),
                  rgb = self.pal.color(color), lw = lw, alpha = alpha, gid = gid, head = head,
                  shrink = tuple(float(v) for v in shrink), bend = float(bend), name = name)
        return self

    def axline(self, xy, slope = None, xy2 = None, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        """A straight line through a data point at a slope, or through a second point, across the whole axes, such as
        the diagonal of observed against predicted values; it leaves the limits to the data."""
        if (slope is None) == (xy2 is None):
            raise ValueError('axline takes a slope or a second point, and only one of them')
        x2, y2 = (xy[0] + 1.0, xy[1] + slope) if xy2 is None else xy2
        self._add('aline', z, x0 = float(xy[0]), y0 = float(xy[1]), x1 = float(x2), y1 = float(y2),
                  rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash)
        return self

    def rectangle(self, x, y, width, height, color = 'faint', edge = None, radius = 0.0, alpha = 1.0, z = 1,
                  axes_fraction = False):
        """A rectangle from its lower left corner, in data units or with axes_fraction = True in fractions of the
        axes, such as a box in a schematic or a card behind text. `edge` is (color, width), `radius` rounds its corners
        by that many points, and color = 'none' leaves only the edge."""
        self._add('rect', z, x = float(x), y = float(y), w = float(width), h = float(height),
                  rgb = self.pal.color(color), edge = None if edge is None else (self.pal.color(edge[0]), edge[1]),
                  radius = float(radius), alpha = alpha, frac = axes_fraction)
        return self

    def axhline(self, y = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        self._add('hline', z, v = float(y), rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash, source = [self.artists[-1]])
        return self

    def axvline(self, x = 0.0, color = 'muted', lw = 0.8, dash = (1, 3), label = None, z = 1):
        self._add('vline', z, v = float(x), rgb = self.pal.color(color), lw = lw, dash = dash)
        self._labeled(label, self.pal.color(color), 'line', dash, source = [self.artists[-1]])
        return self

    def heatmap(self, data, xticklabels = None, yticklabels = None, cmap = 'diverging', vmin = None, vmax = None,
                center = 0.0, labels_right = True, show_all = False, colorbar = False, colorbar_label = '',
                annot = False, fmt = '.2f', annot_size = None, names = None, highlight = None):
        """A matrix of cells, first row at the top, where NaN is left empty. Labels are given per column and row, True
        numbers them, and False or None leaves them out. `cmap` is diverging, sequential, or sequential_r, which puts
        the darkest color at the lowest value. Too many labels are thinned unless show_all,
        which can also be a pair for the x and y axes; the page grows for those that must all show. `annot` writes each
        cell's value into it, formatted by `fmt`, a format spec or a function of the value, in whichever of ink or white
        reads over the cell. `names` holds the
        rows' and the columns' labels as a pair, by default the tick labels, which draws nothing but tells a viewer such
        as slate what the cell under the mouse is. `highlight` keeps the rows and columns it names, by those names, in
        full shade, and the other cells recede (see Axes.highlight)."""
        _check_cmap(cmap)
        m = np.asarray(data, dtype = float)
        if m.ndim != 2 or not m.size:
            raise ValueError(f'A heatmap needs rows and columns of at least one cell, got an array of shape {m.shape}')
        # As seaborn takes them: True numbers the rows or columns, False leaves them unlabeled
        xticklabels, yticklabels = [(list(range(n)) if v is True else None if v is False else v)
                                    for v, n in ((xticklabels, m.shape[1]), (yticklabels, m.shape[0]))]
        rows, columns = (yticklabels, xticklabels) if names is None else names
        for given, n, what in ((rows, m.shape[0], 'row'), (columns, m.shape[1], 'column')):
            if names is not None and given is not None and len(given) != n:
                raise ValueError(f'names has to hold one label per {what}, got {len(given)} for {n} {what}s')
        self.show_all = show_all if isinstance(show_all, tuple) else (show_all, show_all)
        known = m[np.isfinite(m)]
        if cmap == 'diverging':
            d = np.max(np.abs(known - center), initial = 0.0) if vmin is None and vmax is None else max(
                abs((vmin if vmin is not None else center) - center),
                abs((vmax if vmax is not None else center) - center))
            d = float(d) if d else 1.0
            t = 0.5 + (m - center) / (2 * d)
            lo, hi = center - d, center + d
        else:
            lo = vmin if vmin is not None else known.min() if known.size else 0.0
            hi = vmax if vmax is not None else known.max() if known.size else 1.0
            t = (m - lo) / (hi - lo if hi > lo else 1.0)
        self._scale = dict(cmap = cmap, lo = float(lo), hi = float(hi))
        rgb = getattr(self.pal, cmap)(t)
        self._add('heat', 2, m = m, rgb = rgb, rows = None if rows is None else [str(v) for v in rows],
                  columns = None if columns is None else [str(v) for v in columns], focus = _keyset(highlight))
        if annot:
            for r, j in np.argwhere(np.isfinite(m)):
                self.annotate(fmt(m[r, j]) if callable(fmt) else format(m[r, j], fmt), (j + 0.5, m.shape[0] - r - 0.5),
                              size = annot_size or 8.0 * self.pal.spacing, halo = 0.0, color = self.pal.on(rgb[r, j]),
                              ha = 'center', clip = True)
        self.set_xlim(0, m.shape[1])
        self.set_ylim(0, m.shape[0])
        # An empty list of labels means none, as None does
        named = xticklabels is not None and len(xticklabels) > 0
        turned = 90.0 if named and max(len(str(v)) for v in xticklabels) > 2 else 0.0
        self.set_xticks(list(np.arange(m.shape[1]) + 0.5), xticklabels, turned) if named else self.set_xticks([])
        if yticklabels is not None and len(yticklabels) > 0:
            self.set_yticks(list(np.arange(m.shape[0]) + 0.5), [str(v) for v in yticklabels][::-1])
        else:
            self.set_yticks([])
        self._yright = labels_right
        self._hidden |= {'spines', 'grid', 'ticks'}
        self.grid_on = False
        if colorbar:
            self.colorbar(colorbar_label)
        return self

    def colorbar(self, label = '', loc = 'right', source = None, integer = False):
        """Show the color scale of this axes, or of `source`: to its right, or filling it with loc = 'fill'. `integer`
        keeps whole numbers only on its ticks, for counts."""
        if loc not in ('right', 'fill'):
            raise ValueError(f"A colorbar's place has to be right or fill, got {loc!r}")
        self._cbar = dict(label = label, loc = loc, source = source or self, integer = integer)
        return self

    def ridgeline(self, x, curves, labels = None, overlap = 2.05):
        """Distributions stacked with overlap, from the back (first) to the front (last)."""
        x = np.asarray(x, dtype = float)
        heights = [np.nan_to_num(np.asarray(y, dtype = float), posinf = 0.0, neginf = 0.0) for y in curves]
        for y in heights:
            _check_lengths(x, y)
        # A curve of zeros stays flat
        curves = [y / (y.max(initial = 0.0) or 1.0) for y in heights]
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

    def dendrogram(self, icoord, dcoord, orientation = 'top', n = None, labels = None, scale = False, colors = None,
                   show_all = False):
        """Link lines as scipy gives them; heights run away from the matrix that sits on the given side. `labels` name
        the leaves, which too many of are thinned unless show_all, `scale` keeps ticks on the height axis, and `colors`
        gives each link its own color, to mark clusters."""
        self._add('dendro', 3, ic = icoord, dc = dcoord, orient = orientation,
                  rgb = None if colors is None else [self.pal.color(k) for k in colors])
        self.show_all = (show_all, show_all)
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
             halo = 0.0, z = 20, axes_fraction = False, linespacing = 1.25):
        """Text at a data position, or at a fraction of the axes with axes_fraction = True; several lines are
        `linespacing` sizes apart."""
        self._add('text', z, x = float(x), y = float(y), s = str(s), size = size, rgb = self.pal.color(color),
                  bold = bold, ha = ha, va = va, rotation = rotation, halo = halo, dx = 0.0, dy = 0.0,
                  frac = axes_fraction, tag = False, clip = False, spread = False, linespacing = linespacing,
                  placed = True, glyph = None)
        return self

    def annotate(self, s, xy, offset = (0.0, 0.0), size = None, color = 'ink', bold = None, ha = 'left', va = 'center',
                 rotation = 0.0, halo = None, z = 20, svg_label = False, clip = False, spread = False,
                 linespacing = 1.25, glyph = None, y_fraction = False):
        """Text next to a data point, offset in points, with a halo of the page's color that keeps it readable over the
        marks. With svg_label = True, SVG output writes it the way matplotlib writes text, so that glycowork's
        annotate_figure can replace it with a glycan drawing. With clip = True it shows only while its point is within
        the limits. Labels with spread = True on one side of their points, such as names at the ends of lines, move
        apart vertically as little as it takes for none to overlap, within the axes' height, each joined to its point
        by a hairline where it had to move. Without a size it takes 8.6 points, and its halo 2.6, both scaled by the
        palette's spacing. `glyph` draws the label instead, such as 'glycan' for its SNFG drawing (see tick_params), its
        root symbol level with the point, its box's edge there with va 'top' or 'bottom', over a die-cut halo of the
        page's color, or white as a sticker on a dark page. `y_fraction` reads y as a share of the axes' height, as in
        plot and image, which keeps a label, such as a fragment's drawing over its peak, in its band through zooming;
        such a label takes no part in the limits and shows while its x is within them."""
        if spread and y_fraction:
            raise ValueError('A label spread among the points needs its point in data, not y_fraction')
        k = self.pal.spacing
        self._add('text', z, x = float(xy[0]), y = float(xy[1]), s = str(s), size = 8.6 * k if size is None else size,
                  rgb = self.pal.color(color), bold = bold, ha = ha, va = va, rotation = rotation,
                  halo = 2.6 * k if halo is None else halo, dx = offset[0], dy = offset[1], frac = False,
                  tag = svg_label, clip = clip, spread = spread, linespacing = linespacing,
                  glyph = text.check_glyphs(glyph), grow = size is None, yfrac = bool(y_fraction))
        return self

    def _anchor(self, a):
        """Where a text artist's point lies on the page: in data, in fractions of the axes, or with y_fraction at its x
        in data and a share of the axes' height."""
        x0, y0, w, h = self.rect
        return (x0 + a['x'] * w if a['frac'] else float(self._tx(a['x'])),
                y0 + a['y'] * h if a['frac'] or a.get('yfrac') else float(self._ty(a['y'])))

    def bracket(self, x0, x1, text = '', color = None, lw = None, size = None, tip = None, z = 6):
        """A bracket from the group at x0 to the one at x1 with text above it, such as 'p = 0.003' or stars, or a
        p-value given as a number, which it writes as 'p = 0.003', or with a power of ten below 0.001. It sits above the
        data from x0 to x1, and brackets that would meet stack, the narrower lower, while the y limits grow to hold them
        all, unless set by hand. Positions are numbers, categories, a group's label, or (category, hue level) for a
        group of a box, strip, or violin plot with hue. Line and type follow the
        palette's whiskers and text; `tip` is how far its ends reach down, in points."""
        pal = self.pal
        ends = []
        for v in (x0, x1):
            if isinstance(v, tuple) and tuple(map(str, v)) in self._dodged:
                v = self._dodged[tuple(map(str, v))]
            elif isinstance(v, str) and v in self._cats['x']:
                v = self._cats['x'][v]
            elif isinstance(v, str) and self._xtl is not None and v in self._xtl:
                v = self._xticks[self._xtl.index(v)]
            elif isinstance(v, str):
                raise ValueError(f'A bracket names a group by a category or a tick label of x, and {v!r} is neither')
            elif isinstance(v, tuple):
                raise ValueError(f'A bracket names a group of hue by (category, level), and {v!r} is not one of x')
            ends.append(float(v))
        if not math.isfinite(ends[0] + ends[1]) or ends[0] == ends[1]:
            raise ValueError(f'A bracket runs between two different finite positions, got {x0!r} and {x1!r}')
        if isinstance(text, numbers.Real) and not isinstance(text, bool):
            p = float(text)
            text = 'p = ' + _minus(pal, ['%.2g' % p if p >= 1e-3 or p <= 0 else fmt(p, True, 2)])[0]
        self._add('bracket', z, x0 = min(ends), x1 = max(ends), s = str(text),
                  rgb = pal.color(color or pal.spine_color or pal.ink),
                  lw = max(pal.line_width * 0.5, 0.9) if lw is None else lw,
                  size = pal.tick_size if size is None else size, tip = 4.0 * pal.spacing if tip is None else tip,
                  bold = pal.text_bold, clip = False)
        return self

    # --- settings ---
    def set_xlabel(self, s, size = None, pad = None):
        """The x axis's name, at `size` points or the palette's size for axis names, `pad` points below the tick labels
        if given, as matplotlib's labelpad. An inner panel of axes sharing x keeps it hidden, as matplotlib does."""
        self.xlabel, self._label_size['x'], self._label_pad['x'] = '' if self._outer['x'] else s, size, pad
        return self

    def set_ylabel(self, s, size = None, pad = None):
        """The y axis's name, at `size` points or the palette's size for axis names, `pad` points beside the tick labels
        if given, as matplotlib's labelpad. An inner panel of axes sharing y keeps it hidden, as matplotlib does."""
        self.ylabel, self._label_size['y'], self._label_pad['y'] = '' if self._outer['y'] else s, size, pad
        return self

    def set_title(self, s, size = None, pad = None):
        """The axes' title; without a size it takes 15 points, less where it would not fit. `pad` is the points from the
        axes to its baseline, eight scaled by the palette's spacing unless given."""
        self.title_text, self._title_fixed, self._title_pad = s, size, pad
        return self

    def set_xlim(self, lo, hi):
        """The x limits; None for an end keeps it automatic, as for an axis that starts at zero and leaves its top
        to the data and to what brackets need above them."""
        self._xlim = _check_limits(lo, hi)
        return self

    def set_ylim(self, lo, hi):
        """The y limits, as set_xlim takes them."""
        self._ylim = _check_limits(lo, hi)
        return self

    def set_xticks(self, ticks = None, labels = None, rotation = 0.0, fmt = None, emphasis = None):
        """Ticks at these positions, or automatic ones with ticks = None, with these labels or their values, whose
        labels turn by `rotation` degrees. `fmt` writes the values, as a format string such as '{:.0%}' or a
        function of the value. `emphasis` maps labels to a color they are set in, bold, such as the method a figure is
        about."""
        _check_tick_labels(ticks, labels)
        self._xticks = None if ticks is None else list(ticks)
        self._xtl = None if labels is None else [str(v) for v in labels]
        self._xrot, self._xfmt, self._xemph = rotation, fmt, dict(emphasis or {})
        return self

    def set_yticks(self, ticks = None, labels = None, fmt = None, emphasis = None):
        """Ticks at these positions, or automatic ones with ticks = None, with these labels or their values. `fmt`
        writes the values, as a format string such as '{:.0%}' or a function of the value. `emphasis` maps labels to a
        color they are set in, bold."""
        _check_tick_labels(ticks, labels)
        self._yticks = None if ticks is None else list(ticks)
        self._ytl = None if labels is None else [str(v) for v in labels]
        self._yfmt, self._yemph = fmt, dict(emphasis or {})
        return self

    def set_tick_size(self, size):
        """The size of the tick labels in points, which otherwise the palette decides."""
        self._tick_size = size
        return self

    def tick_params(self, axis = 'both', length = None, labelsize = None, pad = None, labels = None, top = None,
                    fit = None, glyphs = None):
        """Ticks of the x axis, the y axis, or both, as in matplotlib: their `length` and the `pad` between them and
        their labels in points, the labels' size, and labels = False to leave the labels out, as beside a panel
        that shares them. top = True writes the x labels above the axes, as over a table of scores, where a positive
        `rotation` turns them up and to the right. fit = False keeps every label as set, level or turned as asked,
        where viscose would otherwise turn or thin labels that crowd. `glyphs` draws the labels instead, where it
        can: 'glycan' as glycowork's SNFG drawings, or a name given to register_glyphs, or a function of (label,
        vertical) returning SVG; glyphs = False turns it off. A glyph's root, such as a glycan's reducing end, sits
        at its tick, its symbols as tall as the labels' type, shrinking to fit between ticks, while x labels that still
        crowd run up the page, as text would turn. What is not given stays as it was."""
        if axis not in ('x', 'y', 'both'):
            raise ValueError(f"tick_params takes axis = 'x', 'y', or 'both', got {axis!r}")
        if top is not None and axis != 'x':
            raise ValueError("Only the x labels can go on top, with axis = 'x'")
        for name in ('x', 'y') if axis == 'both' else (axis, ):
            given = dict(length = length, labelsize = labelsize, pad = pad, labels = labels, top = top, fit = fit,
                         glyphs = text.check_glyphs(glyphs))
            self._tick_opts[name].update({k: v for k, v in given.items() if v is not None})
        return self

    def set_spines(self, left = None, bottom = None):
        """Show or hide the left and the bottom axis line, with their ticks; None keeps the palette's choice."""
        self._spine_set.update({k: bool(v) for k, v in (('left', left), ('bottom', bottom)) if v is not None})
        return self

    def set_xscale(self, s, functions = None):
        """'linear', 'log', 'sqrt' (signed, so zero and below keep their place), or 'function' with
        functions = (forward, inverse), two maps over arrays from data to even spacing along the axis and back. Axes
        sharing x share it."""
        warp = _warp(s, functions)
        for ax in self._share['x']:
            ax._xwarp, ax._xscale = warp, s
        return self

    def set_yscale(self, s, functions = None):
        """'linear', 'log', 'sqrt' (signed, so zero and below keep their place), or 'function' with
        functions = (forward, inverse), two maps over arrays from data to even spacing along the axis and back. Axes
        sharing y share it."""
        warp = _warp(s, functions)
        for ax in self._share['y']:
            ax._ywarp, ax._yscale = warp, s
        return self

    def set(self, **settings):
        """Several settings at once, as in matplotlib: xlabel, ylabel, title, xlim, ylim, xscale, yscale, tick_size."""
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

    def legend(self, loc = 'direct', glyphs = None):
        """Name what the plot shows. By default directly, each label in its series' color beside what it names: at a
        line's or a band's right end, moving apart as little as it takes, with a hairline to each one that moved; at
        the middle of a group of points, wherever the groups hold together, on the free spot nearest it; over a
        histogram's highest bar; at the right of stacked bars; and along reference lines. Whatever has no such place
        goes into a legend below the axes. A place, below or right of the axes, or inside at upper left, upper right,
        lower left, or lower right, puts everything into one legend there. `glyphs` draws the entries' names as glyphs
        where it can, such as 'glycan' (see tick_params), in a legend."""
        # Any other name would quietly land at the lower left
        if loc not in LEGEND_SPOTS:
            raise ValueError(f"A legend's place has to be one of {', '.join(LEGEND_SPOTS)}, got {loc!r}")
        self._legend_opts, self._legend_glyphs = loc, text.check_glyphs(glyphs)
        self._legend_fallback = 'below'
        return self

    def highlight(self, keys):
        """Keep what `keys` name in full shade and let everything else on the axes recede toward the page, each in its
        own hue: a series by its label, such as a line's or a hue level's, the categories and hue levels of bars,
        boxes, and violins, points and values by their `names`, and a heatmap's rows and columns by theirs. The focus
        draws over the context, which casts no shadow, and the legend and the labels that name series directly follow.
        A plot call's own `highlight` takes over for its marks, and None ends it."""
        self._focus = _keyset(keys)
        return self

    def off(self):
        """Hide axis lines, ticks and grid, for figures whose position axes carry no meaning."""
        self._hidden |= {'spines', 'ticks', 'grid'}
        self._xticks, self._yticks = [], []
        self.grid_on = False
        return self

    # --- positions ---
    def boxes(self):
        """What the axes draw, as rows of (left, bottom, right, top) in page points once `Figure.layout` has placed
        them: one per point, per piece of a line or a bracket about a point long, per bar and rectangle, per line of
        text at its unspread place, and per glyph where it is drawn, for placing labels, keys, and cards clear of it
        all. Bands and spans are left out, as the ground under the data."""
        return self._boxes(set())

    def _boxes(self, skip):
        """boxes(), without the artists whose ids are in `skip`."""
        # Single boxes as rows, and those of many points or line pieces as arrays, which stay arrays, all in order
        pal, out = self.pal, []
        x0, y0, w, h = self.rect
        lo_x, hi_x = sorted(self.xlim)
        lo_y, hi_y = sorted(self.ylim)
        pieces = []
        for a in self.artists:
            k = a['kind']
            if id(a) in skip:
                continue
            if k == 'points':
                x, y = np.atleast_1d(a['x']), np.atleast_1d(a['y'])
                keep = np.isfinite(x) & np.isfinite(y) & (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
                area = np.broadcast_to(np.asarray(a['s'], dtype = float), x.shape)
                r = np.sqrt(np.maximum(area, 0) * pal.point_scale) / 2
                r = r[keep] + (a['edge'][1] / 2 if a['edge'] else 0.0)
                px, py = self._tx(x[keep]), self._ty(y[keep])
                out.append(np.column_stack([px - r, py - r, px + r, py + r]))
            elif k == 'line':
                px, py = self._tx(a['x']), y0 + a['y'] * h if a['frac'] else self._ty(a['y'])
                if a['offset'] is not None:
                    px, py = px + a['offset'][:, 0], py + a['offset'][:, 1]
                pieces.append((px, py, pal.line_width if a['lw'] is None else a['lw']))
            elif k in ('hline', 'vline'):
                v = float(self._ty(a['v']) if k == 'hline' else self._tx(a['v']))
                ends = ([x0, x0 + w], [v, v]) if k == 'hline' else ([v, v], [y0, y0 + h])
                pieces.append((*map(np.array, ends), a['lw']))
            elif k == 'hsegs':
                for y, xa, xb in zip(self._ty(a['y']), self._tx(a['x0']), self._tx(a['x1'])):
                    pieces.append((np.array([xa, xb]), np.array([y, y]), a['lw']))
            elif k == 'vsegs':
                for x, ya, yb in zip(self._tx(a['x']), self._ty(a['y0']), self._ty(a['y1'])):
                    pieces.append((np.array([x, x]), np.array([ya, yb]), a['lw']))
            elif k == 'bars':
                for x, b, v, bw in zip(a['x'], a['b'], a['h'], a['w']):
                    xs, ys = (self._tx([b, b + v]), self._ty([x - bw / 2, x + bw / 2])) if a['horizontal'] else (
                        self._tx([x - bw / 2, x + bw / 2]), self._ty([self._base(b, self._yscale, self.ylim), b + v]))
                    out.append([min(xs), min(ys), max(xs), max(ys)])
            elif k in ('box', 'violin'):
                # Each group as one box around what it draws
                for x, w, stats, v, shape in zip(a['x'], a['w'], a['stats'], a['values'],
                                                 a.get('shapes', [None] * len(a['x']))):
                    if stats is None:
                        continue
                    half = w / 2 if shape is None else float(shape[1].max())
                    shown = v if k == 'violin' or a['offsets'] is not None or a['fliers'] else []
                    ys = [*([stats[0], stats[4]] if k == 'box' and a['box'] else []), *shown,
                          *(shape[0][[0, -1]] if shape is not None else [])]
                    across, along = [x - half, x + half], [min(ys), max(ys)]
                    xs, ys = (self._tx(along), self._ty(across)) if a['horizontal'] else (self._tx(across),
                                                                                          self._ty(along))
                    out.append([min(xs), min(ys), max(xs), max(ys)])
            elif k == 'bracket':
                # Its line, as pieces, and its text as one box per line
                with text.family(pal.font):
                    placed = self._bracket_plan().get(id(a))
                    if placed is None:
                        continue
                    left, right, line, top = placed
                    pieces.append((np.array([left, left, right, right]),
                                   np.array([line - a['tip'], line, line, line - a['tip']]), a['lw']))
                    lines = str(a['s']).split('\n') if a['s'].strip() else []
                    for i, s in enumerate(lines):
                        tw, base = text.width(s, a['size'], a['bold']), line + 2.0 * pal.spacing + 0.22 * a['size']
                        base += (len(lines) - 1 - i) * 1.25 * a['size']
                        out.append([(left + right - tw) / 2, base - 0.22 * a['size'], (left + right + tw) / 2,
                                    base + text.CAP_HEIGHT * a['size']])
            elif k == 'rect':
                xs = [x0 + a['x'] * w, x0 + (a['x'] + a['w']) * w] if a['frac'] else self._tx([a['x'], a['x'] + a['w']])
                ys = [y0 + a['y'] * h, y0 + (a['y'] + a['h']) * h] if a['frac'] else self._ty([a['y'], a['y'] + a['h']])
                out.append([min(xs), min(ys), max(xs), max(ys)])
            elif (k == 'text' and a.get('glyph') and math.isfinite(a['x'] + a['y'])
                  and (box := self._note_size(a)[1]) is not None):
                # A glyph as drawn: where spreading put it, or beside its point
                left, bottom, right, top = box
                if a['spread']:
                    px, py, _, pts = self._place_glyphs()[id(a)]
                    g, _, cut = self._note(a)
                    left, bottom, right, top = g.extent(pts, cut)
                else:
                    px, py = self._anchor(a)
                    px, py = px + a.get('_dx', a['dx']), py + a['dy']
                    px -= {'left': left, 'center': (left + right) / 2, 'right': right}[a.get('_ha', a['ha'])]
                    py -= {'top': top, 'bottom': bottom, 'baseline': bottom}.get(a['va'], 0.0)
                out.append([px + left, py + bottom, px + right, py + top])
            elif k == 'text' and not a['rotation'] and math.isfinite(a['x'] + a['y']):
                px, py = self._anchor(a)
                if a.get('direct') in ('cloud', 'peak'):
                    with text.family(pal.font):
                        px, py = self._place_direct().get(id(a), (px, py))
                size, bold = a['size'] or pal.tick_size, pal.text_bold if a['bold'] is None else a['bold']
                lines = str(a['s']).split('\n')
                # As text.draw sets them: the block's lift, then each line's baseline and its glyphs' reach
                lift = {'center': 0.5, 'bottom': 1.0}.get(a['va'], 0.0) * (len(lines) - 1)
                shift = {'baseline': 0.0, 'center': -text.CAP_HEIGHT * size / 2, 'top': -text.CAP_HEIGHT * size,
                         'bottom': 0.22 * size}[a['va']]
                with text.family(pal.font):
                    for i, line in enumerate(lines):
                        tw = text.width(line, size, bold, a.get('tracking', 0.0))
                        ha = a.get('_ha', a['ha'])
                        left = px + a.get('_dx', a['dx']) - {'left': 0.0, 'center': tw / 2, 'right': tw}[ha]
                        base = py + a['dy'] - (i - lift) * size * a['linespacing'] + shift
                        out.append([left, base - 0.22 * size, left + tw, base + text.CAP_HEIGHT * size])
        for px, py, lw in pieces:
            for xa, ya, xb, yb in zip(px[:-1], py[:-1], px[1:], py[1:]):
                if np.isfinite([xa, ya, xb, yb]).all():
                    t = np.linspace(0.0, 1.0, max(2, int(math.hypot(xb - xa, yb - ya)) + 1))
                    sx, sy = xa + t * (xb - xa), ya + t * (yb - ya)
                    out.append(np.column_stack([sx - lw / 2, sy - lw / 2, sx + lw / 2, sy + lw / 2]))
        return np.concatenate([np.reshape(np.asarray(v, dtype = float), (-1, 4)) for v in out] + [np.zeros((0, 4))])

    def get_xlim(self):
        """The x limits as drawn, whether set or automatic."""
        self._range()
        return self.xlim

    def get_ylim(self):
        """The y limits as drawn, whether set or automatic."""
        self._range()
        return self.ylim

    def page(self, x, y):
        """Data positions in page points, from the page's lower left, once `Figure.layout` has placed the axes."""
        return self._tx(x), self._ty(y)

    def data(self, px, py):
        """Page points as data positions, the reverse of `page`."""
        x0, y0, w, h = self.rect
        out = []
        for v, start, size, (lo, hi), (forward, inverse) in ((px, x0, w, self.xlim, self._xwarp),
                                                              (py, y0, h, self.ylim, self._ywarp)):
            t = (np.asarray(v, dtype = float) - start) / size
            lo, hi = forward(np.float64(lo)), forward(np.float64(hi))
            out.append(inverse(lo + t * (hi - lo)))
        return tuple(out)

    # --- limits and ticks ---
    def _extent(self, axis):
        lo, hi = [], []
        sticky = []
        log = (self._xscale if axis == 'x' else self._yscale) == 'log'
        for a in self.artists:
            k = a['kind']
            if axis == 'x' and k in ('line', 'points', 'stems', 'band', 'errors', 'vsegs'):
                vals = a['x']
            elif axis == 'y' and k == 'line' and a['frac']:
                continue
            elif axis == 'y' and k in ('line', 'points', 'hsegs'):
                vals = a['y']
            elif axis == 'y' and k == 'vsegs':
                vals = np.r_[a['y0'], a['y1']]
            elif axis == 'x' and k == 'vspan':
                vals = np.array([a['x0'], a['x1']])
            elif axis == 'y' and k == 'hspan':
                vals = np.array([a['y0'], a['y1']])
            elif axis == 'y' and k == 'band':
                vals = np.r_[a['y1'], a['y2']]
            elif axis == 'y' and k == 'stems':
                vals = np.r_[a['y'], 0.0]
                sticky.append(0.0)
            elif axis == 'y' and k == 'errors':
                vals = np.r_[a['lo'], a['hi']]
            elif axis == 'x' and k == 'hsegs':
                vals = np.r_[a['x0'], a['x1']]
            elif k == 'arrow':
                vals = np.array([a[axis + '0'], a[axis + '1']])
            elif k == 'rect' and not a['frac']:
                vals = np.array([a[axis], a[axis] + a['w' if axis == 'x' else 'h']])
            elif k == 'contour':
                vals = a[axis]
            elif k in ('box', 'violin') and (axis == 'y') == a['horizontal']:
                vals = np.r_[a['x'] - a['w'] * 0.66, a['x'] + a['w'] * 0.66]
            elif k == 'box' and a['offsets'] is not None:
                vals = np.concatenate([np.zeros(0)] + a['values'])
            elif k == 'box':
                # Fliers the plot does not show take up no room in it
                vals = np.array([u for v in a['stats'] if v for u in (v[0], v[4], *(v[5] if a['fliers'] else ()))])
            elif k == 'violin':
                vals = np.concatenate([np.zeros(0)] + [v if s is None else s[0][[0, -1]]
                                                       for v, s in zip(a['values'], a['shapes'])])
            elif k == 'bars' and (axis == 'x') != a['horizontal']:
                # A little room beside the outer bars, in their own scale, as a histogram's can be of any
                vals = np.r_[a['x'] - a['w'] * 0.6, a['x'] + a['w'] * 0.6]
            elif k == 'bars':
                vals = np.r_[a['b'], a['b'] + a['h']]
                sticky.extend(np.unique(a['b']).tolist())
            elif k == 'hline' and axis == 'y' or k == 'vline' and axis == 'x':
                vals = np.array([a['v']])
            elif k == 'text' and not a['frac'] and not a.get('direct') and not a.get('yfrac'):
                vals = np.array([a[axis]])
            elif k == 'bracket' and axis == 'x':
                vals = np.array([a['x0'], a['x1']])
            else:
                continue
            vals = np.asarray(vals, dtype = float)
            # A log axis has no place for zero or below, where the base of a bar usually is
            vals = vals[np.isfinite(vals) & (vals > 0)] if log else vals[np.isfinite(vals)]
            if vals.size:
                lo.append(vals.min())
                hi.append(vals.max())
        return (min(lo), max(hi), sticky) if lo else None

    def _range(self):
        """The limits of both axes, as set or from the data, of all the axes they share an axis with."""
        for axis, scale, margin in (('x', self._xscale, self._xmargin), ('y', self._yscale, self._ymargin)):
            group = self._share[axis]
            found = [e for e in (ax._extent(axis) for ax in group) if e is not None]
            lo, hi, sticky = ((min(e[0] for e in found), max(e[1] for e in found), [v for e in found for v in e[2]])
                              if found else (0.0, 1.0, []))
            # Limits set on one of them, or turned around, hold for all
            fixed = next((v for v in [getattr(ax, '_' + axis + 'lim') for ax in [self] + group] if v is not None),
                         None)
            inverted = any(getattr(ax, '_' + axis + 'inv') for ax in group)
            if scale == 'log':
                # Nothing above zero leaves one decade to show
                lo = lo if lo > 0 else 1.0
                hi = hi if hi > lo else lo * 10
                span = 0.03 * math.log10(hi / lo)
                lo, hi = 10 ** (math.log10(lo) - span), 10 ** (math.log10(hi) + span)
            else:
                # The margin is a share of the axis as drawn, so measured after the scale's map
                forward, inverse = self._xwarp if axis == 'x' else self._ywarp
                ends = forward(np.array([lo, hi, *sticky], dtype = float))
                lo, hi, sticky = float(ends[0]), float(ends[1]), ends[2:]
                stuck = any(abs(lo - s) < 1e-12 for s in sticky)
                # Data without a range spans a tenth of its value, or one, and unless it is a bar's base the margin
                # holds around that too, so that no tick lands on a corner, where the x and y labels meet
                pad = (hi - lo) * margin if hi > lo else (abs(lo) * 0.1 or 1.0) * (1.0 + 2.0 * margin * (not stuck))
                # Bars of no height stick to both ends, and the axis still needs a range
                lo, hi = (lo if stuck else lo - pad,
                          hi if hi > lo and any(abs(hi - s) < 1e-12 for s in sticky) else hi + pad)
                lo, hi = (float(v) for v in inverse(np.array([lo, hi])))
            if fixed is not None:
                # An end left None stays automatic, and so does a log axis's zero or negative end, which it has none of
                lo, hi = [auto if v is None or (v <= 0 and scale == 'log') else v for v, auto in zip(fixed, (lo, hi))]
                pad = abs(lo) * 0.1 or 1.0
                lo, hi = (lo - pad, hi + pad) if lo == hi else (lo, hi)
            # Brackets lift only an end that is automatic
            free = fixed is None or fixed[0 if inverted else 1] is None
            lift = max(ax._lift for ax in group) if axis == 'y' and free else 1.0
            if lift != 1.0:
                # Brackets stretch the end at the page's top away from the other one, the low end on a turned axis
                forward, inverse = self._ywarp
                f0, f1 = forward(np.array([hi, lo] if inverted else [lo, hi], dtype = float))
                top = float(inverse(np.array([f0 + lift * (f1 - f0)]))[0])
                lo, hi = (top, hi) if inverted else (lo, top)
            setattr(self, axis + 'lim', (hi, lo) if inverted else (lo, hi))

    def _limits(self):
        self._over = [0.0, 0.0]
        self._legend_size = None
        if self._cbar and self._cbar['source']._scale is None:
            raise ValueError('A colorbar shows the color scale of a heatmap or of scatter(c = ...), and these axes '
                             'have neither')
        # What fitting the labels turned or moved last time starts over, as limits or a size may have changed
        self.xrot = self._xrot
        self._gscale, self._gcols = {'x': 1.0, 'y': 1.0}, 1
        for a in self.artists:
            if a['kind'] == 'text':
                a['_ha'], a['_dx'] = a['ha'], a['dx']
        self._range()
        self._ticks('x')
        self._ticks('y')
        self._direct()

    def _direct(self):
        """Labels that name what the legend's entries name, beside it, for the limits as they stand, as text artists
        made anew at each layout, and the entries a legend still shows: those with no such place, and the headers of
        their sections."""
        self.artists = [a for a in self.artists if not a.get('direct')]
        items, self._direct_room, self._direct_spots = self.legend_items, 0.0, (None, {})
        self._legend_keep, self._legend_loc = list(range(len(items))), self._legend_opts
        if self._legend_opts != 'direct':
            return
        pal, k = self.pal, self.pal.spacing
        size, track, _ = self._legend_entries(())
        fx, fy = self._xwarp[0], self._ywarp[0]
        # Where on the axes a value lies, from 0 at the left or bottom to 1 at the right or top, however turned
        across = lambda v, f, lim: (f(np.asarray(v, dtype = float)) - f(np.float64(lim[0]))) / (
            f(np.float64(lim[1])) - f(np.float64(lim[0])))
        seen = lambda x, y: (np.isfinite(across(x, fx, self.xlim)) & np.isfinite(across(y, fy, self.ylim)) &
                             (across(x, fx, self.xlim) >= 0) & (across(x, fx, self.xlim) <= 1) &
                             (across(y, fy, self.ylim) >= 0) & (across(y, fy, self.ylim) <= 1))
        clouds, made = {}, []
        for i, (s, rgb, kind, marker, area) in enumerate(items):
            source = self._sources.get(i)
            if not source or self._legend_glyphs or self._yright:
                continue
            a = source[0]
            spot = None
            if a['kind'] == 'line' and a['clip'] and not a['frac'] and a['offset'] is None:
                ok = seen(a['x'], a['y']) & (a['x'] > 0 if self._xscale == 'log' else True) & (
                    a['y'] > 0 if self._yscale == 'log' else True)
                if ok.any():
                    j = np.flatnonzero(ok)[np.argmax(across(a['x'][ok], fx, self.xlim))]
                    # Clear of the line's own width and of a marker at its end
                    lw = pal.line_width if a['lw'] is None else a['lw']
                    spot = ('end', a['x'][j], a['y'][j], lw / 2 + 4.0 * k)
            elif a['kind'] == 'band':
                mid = (np.asarray(a['y1'], dtype = float) + np.asarray(a['y2'], dtype = float)) / 2
                ok = seen(a['x'], mid)
                if ok.any():
                    j = np.flatnonzero(ok)[np.argmax(across(np.asarray(a['x'])[ok], fx, self.xlim))]
                    spot = ('end', a['x'][j], mid[j], 4.0 * k)
            elif a['kind'] == 'hline' and seen(self.xlim[1], a['v']):
                spot = ('end', self.xlim[1], a['v'], 4.0 * k)
            elif a['kind'] == 'vline' and seen(a['v'], self.ylim[1]):
                spot = ('top', a['v'], self.ylim[1], a['lw'] / 2 + 3.0 * k)
            elif a['kind'] == 'bars' and a['stacked'] and not a['horizontal']:
                top = np.asarray(a['b']) + np.asarray(a['h']) / 2
                ok = seen(a['x'], top) & (np.asarray(a['h']) != 0)
                if ok.any():
                    j = np.flatnonzero(ok)[np.argmax(across(np.asarray(a['x'])[ok], fx, self.xlim))]
                    spot = ('end', a['x'][j] + a['w'][j] / 2, top[j], 4.0 * k)
            elif a['kind'] == 'bars' and not a['horizontal']:
                tops = np.asarray(a['b']) + np.asarray(a['h'])
                ok = seen(a['x'], tops)
                if ok.any():
                    j = np.flatnonzero(ok)[np.argmax(across(tops[ok], fy, self.ylim))]
                    spot = ('peak', a['x'][j], tops[j], 0.0)
            elif a['kind'] == 'points' and not a.get('mapped'):
                xs = np.concatenate([np.atleast_1d(b['x']) for b in source])
                ys = np.concatenate([np.atleast_1d(b['y']) for b in source])
                ok = seen(xs, ys)
                if ok.any():
                    u, v = across(xs[ok], fx, self.xlim), across(ys[ok], fy, self.ylim)
                    clouds[i] = np.column_stack([u, v])
                    mx, my = float(np.median(u)), float(np.median(v))
                    # The middle on the axes as drawn, back in data
                    back = lambda t, warp, lim: float(warp[1](np.array([warp[0](np.float64(lim[0])) + t * (
                        warp[0](np.float64(lim[1])) - warp[0](np.float64(lim[0])))]))[0])
                    spot = ('cloud', back(mx, self._xwarp, self.xlim), back(my, self._ywarp, self.ylim), 0.0)
            if spot is not None:
                made.append((i, spot))
        # Groups of points that mingle have no middle of their own, so all of them keep the legend
        if len(clouds) > 1:
            middles = np.array([np.median(c, axis = 0) for c in clouds.values()])
            for n, c in enumerate(clouds.values()):
                near = np.argmin(((c[:, None, :] - middles[None]) ** 2).sum(-1), axis = 1)
                if (near == n).mean() < DIRECT_OWN:
                    made = [(i, spot) for i, spot in made if i not in clouds]
                    break
        for i, (where, x, y, gap) in made:
            # Groups' names stand centered where _place_direct puts them, and the others start beside their mark
            ha, gap = ('center', 0.0) if where in ('cloud', 'peak') else ('left', gap)
            self.artists.append(dict(
                kind = 'text', z = 20, order = 10 ** 9 + i, x = float(x), y = float(y),
                s = text.spaced(pal, items[i][0]), size = size, rgb = items[i][1], bold = pal.text_bold, ha = ha,
                va = 'top' if where == 'top' else 'center', rotation = 0.0, halo = 2.6 * k, dx = gap,
                dy = -2.0 * k if where == 'top' else 0.0, frac = False, tag = False, clip = False,
                spread = where == 'end', linespacing = 1.25, placed = True, glyph = None, tracking = track,
                direct = where, _ha = ha, _dx = gap))
        named = {i for i, _ in made}
        keep = [i for i in range(len(items)) if i not in named]
        # A header stays with the entries of its section that the legend still shows
        heads = [i for i in keep if items[i][2] == 'header']
        self._legend_keep = [i for i in keep if items[i][2] != 'header' or any(
            j > i and items[j][2] != 'header' and not any(i < h < j for h in heads) for j in keep)]
        self._legend_loc = self._legend_fallback if self._legend_keep else None

    def _ticks(self, axis):
        """An axis's ticks and their labels, for its limits as they stand."""
        ticks, labels, form = ((self._xticks, self._xtl, self._xfmt) if axis == 'x' else
                               (self._yticks, self._ytl, self._yfmt))
        cats = self._cats[axis]
        lo, hi = sorted(getattr(self, axis + 'lim'))
        if ticks is None and cats:
            ticks, labels = list(cats.values()), list(cats.keys())
        elif ticks is None and (self._xscale if axis == 'x' else self._yscale) == 'log':
            powers = range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)
            ticks = [10.0 ** e for e in powers if lo <= 10.0 ** e <= hi]
            # Less than a decade holds too few powers of ten to read the axis by
            if len(ticks) < 2:
                ticks = [m * 10.0 ** e for e in powers for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
            if len(ticks) < 2:
                ticks = nice_ticks(lo, hi)
        elif ticks is None:
            ticks = nice_ticks(lo, hi)
        else:
            # Ticks set by hand can lie beyond limits that zooming narrowed
            shown = [i for i, v in enumerate(ticks) if lo - 1e-9 * (hi - lo) <= v <= hi + 1e-9 * (hi - lo)]
            ticks, labels = [ticks[i] for i in shown], None if labels is None else [labels[i] for i in shown]
        # Only numbers take the palette's minus sign, not labels given or categories
        numbers = labels is None
        if labels is None and form is not None:
            labels = [form(v) if callable(form) else form.format(v) for v in ticks]
        setattr(self, axis + 'ticks', ticks)
        labels = labels if labels is not None else tick_labels(ticks)
        # Ticks without labels, as beside a panel that shares them
        if self._tick_opts[axis].get('labels', True) is False or self._outer[axis]:
            labels = []
        setattr(self, axis + 'labels', _minus(self.pal, labels) if numbers else labels)
        self._numbers[axis] = numbers
        # All of them, as fitting may leave some out, while the page grows for all
        self._full[axis] = (getattr(self, axis + 'ticks'), getattr(self, axis + 'labels'))

    def _tx(self, v):
        x0, y0, w, h = self.rect
        forward = self._xwarp[0]
        lo, hi = forward(np.float64(self.xlim[0])), forward(np.float64(self.xlim[1]))
        return x0 + (forward(np.asarray(v, dtype = float)) - lo) / (hi - lo) * w

    def _ty(self, v):
        x0, y0, w, h = self.rect
        forward = self._ywarp[0]
        lo, hi = forward(np.float64(self.ylim[0])), forward(np.float64(self.ylim[1]))
        return y0 + (forward(np.asarray(v, dtype = float)) - lo) / (hi - lo) * h

    # --- layout ---
    def _tick_sizes(self):
        """Tick label sizes for the x and y axes; a heatmap's labels are many, so they are smaller."""
        if self._tick_size is not None:
            sizes = self._tick_size, self._tick_size
        elif any(a['kind'] == 'heat' for a in self.artists):
            sizes = 8.2 * self.pal.spacing, 8.0 * self.pal.spacing
        else:
            sizes = self.pal.tick_size, self.pal.tick_size
        return tuple(self._tick_opts[axis].get('labelsize', v) for axis, v in zip('xy', sizes))

    def _spines(self):
        """The axis lines drawn, by the palette's choice unless set_spines changed it."""
        if 'spines' in self._hidden:
            return ()
        return tuple(s for s in ('left', 'bottom')
                     if self._spine_set.get(s, s in self.pal.spines and s not in self._hidden))

    def _tick_room(self, axis):
        """The spine's offset, the tick's length, and how far from the plot an axis's tick labels start, in points."""
        pal, spines = self.pal, self._spines()
        off = pal.spine_offset if spines else 0.0
        # Ticks ride their axis line, so a hidden line takes its ticks along
        drawn = {'x': 'bottom', 'y': 'left'}[axis] in spines and 'ticks' not in self._hidden
        tl = self._tick_opts[axis].get('length', pal.tick_length) if drawn else 0.0
        return off, tl, off + tl + self._tick_opts[axis].get('pad', 5.0 * pal.spacing)

    def _title_size(self):
        """A long title shrinks to fit the figure's width, down to a readable size; past that the page grows."""
        if self._title_fixed is not None:
            return self._title_fixed
        k = self.pal.spacing
        wide = text.width(self.title_text, 15.0 * k, self.pal.title_bold) or 1.0
        return max(9.0 * k, min(15.0 * k, 15.0 * k * (self.figure._base[0] - self.rect[0] - 3.0 * k) / wide))

    def _axis_name_widths(self, size):
        """How long the x and y axis names run at a size."""
        pal, track = self.pal, 0.9 if self.pal.caps else 0.0
        return [text.width(text.spaced(pal, s), size, pal.label_bold, track) for s in (self.xlabel, self.ylabel)]

    def _tick_glyph(self, axis, s):
        """The drawing a tick label shows, or None where it stays text; x labels run up the page once turned."""
        maker = self._tick_opts[axis].get('glyphs')
        return text.glyph(maker, s, axis == 'x' and bool(self.xrot)) if maker else None

    @property
    def _cut(self):
        """Whether glyphs take the room of a sticker's cut: on a dark page, and on a light one laid out with its dark
        version, as an SVG of both themes has one page for both."""
        return self.pal.is_dark or getattr(self.figure, '_stickers', False)

    def _glyph_extent(self, axis, g):
        """A tick label's glyph's (left, bottom, right, top) around its tick on the page, as drawn: its symbols as tall
        as the labels' type, by how far they shrank, and cut out as a sticker on a dark page."""
        size = self._tick_sizes()['xy'.index(axis)]
        return g.extent(size * text.GLYPH_SYMBOL * self._gscale[axis], self._cut)

    def _note(self, a):
        """A text artist's glyph, its points per symbol, and whether it takes the room of a cut, a halo or a sticker
        on a dark page; no glyph where it stays text."""
        g = text.glyph(a['glyph'], a['s']) if a.get('glyph') else None
        return g, (a['size'] or self.pal.tick_size) * text.GLYPH_SYMBOL, self._cut or a['halo'] > 0

    def _note_size(self, a):
        """A text artist's width on the page, and a glyph's (left, bottom, right, top) around its anchor, or None."""
        g, pts, cut = self._note(a)
        if g is None:
            return text.width(a['s'], a['size'] or self.pal.tick_size,
                              self.pal.text_bold if a['bold'] is None else a['bold'], a.get('tracking', 0.0)), None
        box = g.extent(pts, cut)
        return box[2] - box[0], box

    def _place_glyphs(self):
        """Where spread glyph annotations go, as {id: (x, y, leader, scale)}, with (x, y) the glyph's root on the page,
        `leader` the segment drawn from the point to a drawing that stands apart, or None, and `scale` its points per
        symbol. `_spread_glyphs` places them inside the axes, glycowork's way: each at the free spot nearest its point,
        its root beside it to the left where that is free, clear of the other drawings and of what the axes draw, with
        short leaders ending on a symbol that cross nothing. Drawings take the largest of `GLYPH_SIZES` at which no
        leader is long or tangled, as annotate_figure's did, and never cover more than a fifth of the axes. Kept for
        the page as laid out. While a window's gesture runs (Figure._draft), each drawing and its leader move with
        their point from the last placement made in full, which takes microseconds where placing takes up to half a
        second, and the window places them in full once the gesture pauses."""
        x0, y0, w, h = self.rect
        lo, hi = sorted(self.ylim)
        notes = [a for a in self.artists if a['kind'] == 'text' and a['spread'] and a.get('glyph') and not a['frac']
                 and math.isfinite(a['x'] + a['y']) and not (a['clip'] and not (self._in_x(a['x']) and lo <= a['y'] <=
                                                                                  hi)) and self._note(a)[0] is not None]
        key = (self.rect, tuple(self.xlim), tuple(self.ylim), tuple(id(a) for a in notes))
        if self._glyph_spots[0] == key:
            return self._glyph_spots[1]
        points = [(float(self._tx(a['x'])), float(self._ty(a['y']))) for a in notes]
        settled = self._glyph_settled
        if getattr(self.figure, '_draft', False) and notes and settled[0] == key[3]:
            moved = {}
            for a, (px, py), (qx, qy) in zip(notes, points, settled[1]):
                gx, gy, leader, pts = settled[2][id(a)]
                dx, dy = px - qx, py - qy
                moved[id(a)] = (gx + dx, gy + dy, None if leader is None else (
                    leader[0] + dx, leader[1] + dy, leader[2] + dx, leader[3] + dy), pts)
            return moved
        spots = {}
        if notes:
            k, top = self.pal.spacing, y0 + h
            gap = 4.0 * k
            # Spreading works from the axes' top left with y running down, and what the axes draw weighs as
            # annotate_figure weighed it: a point 12, a line 3 every 6 points
            marks, drawn = [], self._boxes({id(a) for a in notes})
            for l, b, r, t in drawn:
                if r - l > 3.0 and t - b > 3.0 and max(r - l, t - b) <= 24.0:
                    marks.append(((l + r) / 2 - x0, top - (b + t) / 2, 12.0))
                elif max(r - l, t - b) <= 24.0:
                    marks.append(((l + r) / 2 - x0, top - (b + t) / 2, 0.25))
                else:
                    marks += [(u - x0, top - v, 3.0) for u in np.arange(l, r, 6.0) for v in np.arange(b, t, 6.0)]
            # Unlike annotate_figure's anchors, which sat at a label beside its point, these are the points: each weighs
            # enough that no drawing hides one, its own included, and a drawing's natural place starts past its marker
            marks += [(px - x0, top - py, 60.0) for px, py in points]
            reach = [max([(r - l) / 2 for l, b, r, t in drawn if l <= px <= r and b <= py <= t and r - l <= 24.0],
                         default = 0.0) for px, py in points]
            sizes = [v for v in GLYPH_SIZES if v <= 1.0 or all(a.get('grow') for a in notes)]
            best = None
            for share in sizes:
                placements, ports, shapes = [], [], []
                for a, (px, py), radius in zip(notes, points, reach):
                    g, pts, cut = self._note(a)
                    pts *= share
                    left, bottom, right, top_ = g.extent(pts, cut)
                    # Naturally beside its point on the left, its root level with it, as a reducing end faces it there
                    gx = px - radius - gap - right
                    placements.append((gx + left - x0, top - (py + top_), right - left, top_ - bottom, px - x0,
                                       top - py))
                    ports.append([(u * pts - left, top_ - v * pts, r * pts) for u, v, r in g.symbols])
                    shapes.append((left, top_, pts))
                # More ink than a fifth of the axes leaves no way through for leaders
                if share != sizes[-1] and sum(p[2] * p[3] for p in placements) > 0.2 * w * h:
                    continue
                corners, leaders, tangles = _spread_glyphs(placements, (w, h), marks, ports, pad = gap, radii = reach)
                if best is None or tangles < best[0]:
                    best = (tangles, corners, leaders, shapes)
                # A stray crossing among many drawings is not worth smaller ones
                if tangles <= len(notes) // 25:
                    break
            _, corners, leaders, shapes = best
            for a, (px, py), (cx, cy), line, (left, top_, pts) in zip(notes, points, corners, leaders, shapes):
                leader = None
                if line is not None:
                    ex, ey = line[2] + x0, top - line[3]
                    far = math.hypot(ex - px, ey - py)
                    leader = (px + 2.0 * k * (ex - px) / far, py + 2.0 * k * (ey - py) / far, ex, ey)
                spots[id(a)] = (cx + x0 - left, top - cy - top_, leader, pts)
        self._glyph_spots = (key, spots)
        self._glyph_settled = (key[3], points, spots)
        return spots

    def _place_direct(self):
        """Where labels that name groups of points or a histogram go, as {id: (x, y)}, the middle of their line on the
        page: each at the free spot nearest the middle of its group, or over its highest bar, by `_spread_glyphs`, clear
        of each other, of the other groups' middles, and of as many marks as it can. Kept for the page as laid out."""
        x0, y0, w, h = self.rect
        key = (self.rect, tuple(self.xlim), tuple(self.ylim))
        if self._direct_spots[0] == key:
            return self._direct_spots[1]
        notes = [a for a in self.artists if a.get('direct') in ('cloud', 'peak')]
        spots, top, k = {}, y0 + h, self.pal.spacing
        if notes:
            own = {id(a) for a in notes}
            points = {id(a) for a in self.artists if a['kind'] == 'points'}
            # A label had rather move off its group than hide its points, so a point weighs more than for a glyph,
            # however small it is drawn
            dots = self._boxes({id(a) for a in self.artists if id(a) not in points})
            dots = np.column_stack([(dots[:, 0] + dots[:, 2]) / 2 - x0, top - (dots[:, 1] + dots[:, 3]) / 2,
                                    np.full(len(dots), 40.0)])
            marks = []
            for l, b, r, t in self._boxes(own | points):
                if r - l > 3.0 and t - b > 3.0 and max(r - l, t - b) <= 24.0:
                    marks.append(((l + r) / 2 - x0, top - (b + t) / 2, 40.0))
                elif max(r - l, t - b) <= 24.0:
                    marks.append(((l + r) / 2 - x0, top - (b + t) / 2, 0.25))
                else:
                    marks += [(u - x0, top - v, 3.0) for u in np.arange(l, r, 6.0) for v in np.arange(b, t, 6.0)]
            # Text already on the plot, such as a reference line's label, is no place for a label at all, nor right
            # beside it, where the two would read as one
            for l, b, r, t in self._boxes({id(a) for a in self.artists if a['kind'] != 'text' or id(a) in own}):
                em = notes[0]['size']
                marks += [(u - x0, top - v, 150.0) for u in np.arange(l - 2.0 * em, r + 2.0 * em + 3.0, 3.0)
                          for v in np.arange(b - 0.3 * em, t + 0.3 * em + 3.0, 3.0)]
            placements = []
            for a in notes:
                tw, size = self._note_size(a)[0], a['size']
                bh = (text.CAP_HEIGHT + 0.22) * size
                px, py = float(self._tx(a['x'])), float(self._ty(a['y']))
                # Over its middle, or standing on its bar
                lift = bh / 2 + 3.0 * k if a['direct'] == 'peak' else 0.0
                placements.append((px - tw / 2 - x0, top - (py + lift + bh / 2), tw, bh, px - x0, top - py))
            corners, _, _ = _spread_glyphs(placements, (w, h), np.concatenate([dots, np.reshape(marks, (-1, 3))]),
                                           pad = 3.0 * k)
            for a, (cx, cy), (_, _, tw, bh, _, _) in zip(notes, corners, placements):
                # The line's middle, as va = 'center' sets it, a little above the box's middle for the descenders
                spots[id(a)] = (cx + x0 + tw / 2, top - cy - bh + 0.22 * a['size'] + text.CAP_HEIGHT * a['size'] / 2)
        self._direct_spots = (key, spots)
        return spots

    def _ytick_width(self, s, size):
        g = self._tick_glyph('y', s)
        if g is None:
            return text.width(s, size, self.pal.tick_bold, tabular = self._numbers['y'])
        left, _, right, _ = self._glyph_extent('y', g)
        return right - left

    def _ytick_columns(self, size):
        """How wide the y tick labels run from their pad, and where each starts past it, in label order: all at once
        in one column, or, staggered, every other label down the axis a column farther out, so that tall glyphs need
        clear only the next label but one."""
        split = self._decimals(self.ylabels, size)
        if split:
            return max(a for a, _ in split) + max(b for _, b in split), [0.0] * len(split)
        widths = [self._ytick_width(s, size) for s in self.ylabels]
        if self._gcols == 1 or len(widths) < 2:
            return max(widths, default = 0.0), [0.0] * len(widths)
        rank = np.argsort(np.argsort(self._ty(self.yticks), kind = 'stable'), kind = 'stable') % 2
        near, far = (max((w for w, r in zip(widths, rank) if r == side), default = 0.0) for side in (0, 1))
        return near + 0.5 * size + far, [near + 0.5 * size if r else 0.0 for r in rank]

    def _decimals(self, labels, size, numbers = None):
        """For y tick labels, or with `numbers` True any column of numbers, that line up on their decimal points: each
        one's width before that point and from it on, in tabular figures, else None."""
        split = text.decimal(labels) if (self._numbers['y'] if numbers is None else numbers) else None
        if not split or self._tick_opts['y'].get('glyphs'):
            return None
        bold = [s in self._yemph or self.pal.tick_bold for s in labels]
        return [(text.width(a, size, b, tabular = True), text.width(rest, size, b, tabular = True))
                for (a, rest), b in zip(split, bold)]

    def _tick_need(self, axis):
        """How long an axis must be for all its tick labels to show, glyphs at their least size: the plot's length as it
        stands, stretched until its closest ticks hold the most that any label reaches toward its neighbor and that
        neighbor reaches back, as categories lie evenly apart. Pairs are what count, as a tall glycan beside a short one
        needs less than two tall ones would."""
        size = self._tick_sizes()['xy'.index(axis)]
        ticks, labels = (self.xticks, self.xlabels) if axis == 'x' else (self.yticks, self.ylabels)
        if not self._tick_opts[axis].get('glyphs') or len(labels) != len(ticks):
            return len(ticks) * size * 1.05
        # Fitting may have left labels out, but the page grows for all, or each pass would leave out more
        ticks, labels = self._full[axis]
        if len(labels) != len(ticks):
            return len(ticks) * size * 1.05
        pos = np.asarray(self._tx(ticks) if axis == 'x' else self._ty(ticks), dtype = float)
        reach = []
        for s in labels:
            g = self._tick_glyph(axis, s)
            if g is None:
                reach.append((0.6 * size, 0.6 * size))
                continue
            left, bottom, right, top = g.extent(size * text.GLYPH_SYMBOL * text.GLYPH_MIN, self._cut)
            low, high = (bottom, top) if axis == 'y' else (left, right) if self.xrot else ((left - right) / 2,
                                                                                          (right - left) / 2)
            reach.append((-low, high))
        # Staggered labels meet the next but one in their column, twice as far off
        order, step = np.argsort(pos, kind = 'stable'), self._gcols if axis == 'y' else 1
        need = max((reach[a][1] + reach[b][0] for a, b in zip(order[:-step], order[step:])), default = 1.2 * size)
        closest, length = _gap(pos) * step, self.rect[2 if axis == 'x' else 3]
        return (need + 0.35 * size) * (length / closest if math.isfinite(closest) else 1.0)

    def _glyph_room(self, axis):
        """How far an axis's tick labels may reach across from their pad on a page that cannot grow for them: from axes
        at a fixed place to within the page's margin, and on a page held to a window's size, as far as margins may take
        of their cell. Infinite where the page grows."""
        if self._fixed is None and not self.figure._held:
            return math.inf
        (left, right, bottom, top), (x, y, w, h) = self._margins(), self.rect
        if axis == 'y':
            labels = self._ytick_columns(self._tick_sizes()[1])[0]
            if self._fixed is None:
                return PLOT_SHARE_MARGINS * self._room[2] - left - right + labels
            return (self.figure.width - x - w - right if self._yright else x - left) + labels
        labels = self._xtick_height()
        if self._fixed is None:
            return PLOT_SHARE_MARGINS * self._room[3] - bottom - top + labels
        return (self.figure.height - y - h - top if self._tick_opts['x'].get('top', False) else y - bottom) + labels

    def _fit_glyphs(self, axis):
        """Shrinks an axis's glyph labels to fit between their ticks, turns x's to run up the page and staggers y's in
        two columns where they still crowd at their least size, and past that leaves out labels as text would, unless
        all must show. On a page that cannot grow, labels also keep within the room it has across: y's stagger only
        where both columns fit there, and labels shrink past their least size, down to half of it, before any is left
        out. True if anything changed."""
        pal, i = self.pal, 'xy'.index(axis)
        ticks, labels = (self.xticks, self.xlabels) if axis == 'x' else (self.yticks, self.ylabels)
        if len(ticks) < 2 or not labels or not self._tick_opts[axis].get('fit', True):
            return False
        size, maker = self._tick_sizes()[i], self._tick_opts[axis]['glyphs']
        pos = np.asarray(self._tx(ticks) if axis == 'x' else self._ty(ticks), dtype = float)
        order = np.argsort(pos, kind = 'stable')
        before = (self._gscale[axis], self.xrot, len(ticks), self._gcols)
        room = self._glyph_room(axis)
        def fits(turned, step):
            # Each label's reach before and after its tick along the axis, the part that scales with the glyph and the
            # part that does not, then the least share of their full size that fits between all neighbors, each pair's
            # need, and the closest gap; on a page that cannot grow, also the largest share whose labels fit across
            reach, across = [], []
            for s in labels:
                g = text.glyph(maker, s, turned)
                if g is None:
                    half = text.width(s, size, pal.tick_bold) / 2 if axis == 'x' and not turned else 0.6 * size
                    reach.append((0.0, 0.0, half, half))
                    if math.isfinite(room):
                        wide = text.width(s, size, pal.tick_bold, tabular = self._numbers[axis])
                        across.append((0.0, wide if axis == 'y' else wide + 0.4 * size if turned else
                                       size * (1.0 + 1.25 * _extra_lines(s))))
                    continue
                left, bottom, right, top = g.extent(size * text.GLYPH_SYMBOL, self._cut)
                reach.append((-bottom, top, 0.0, 0.0) if axis == 'y' else (-left, right, 0.0, 0.0) if turned else
                             ((right - left) / 2, (right - left) / 2, 0.0, 0.0))
                across.append((right - left if axis == 'y' else top - bottom, 0.0))
            share, pairs, least = math.inf, [], math.inf
            for a, b in zip(order[:-step], order[step:]):
                gap = pos[b] - pos[a]
                if gap <= 0:
                    continue
                scaled, fixed = reach[a][1] + reach[b][0], reach[a][3] + reach[b][2] + 0.35 * size
                if scaled:
                    share = min(share, (gap - fixed) / scaled)
                pairs.append((scaled, fixed))
                least = min(least, gap)
            if not math.isfinite(room):
                return share, math.inf, pairs, least
            if step == 1:
                # Text that runs off by itself is no glyph's to make room for
                most = max(v for v, _ in across)
                return share, room / most if most else math.inf, pairs, least
            # Staggered, every other label down the axis lies a column farther out, half the type's size beyond
            rank = np.empty(len(labels), dtype = int)
            rank[order] = np.arange(len(labels)) % 2
            deep = lambda f: size / 2 + sum(max((max(v * f, t) for (v, t), r in zip(across, rank) if r == side),
                                                default = 0.0) for side in (0, 1))
            if deep(1.0) <= room:
                return share, math.inf, pairs, least
            low, high = 0.0, 1.0
            for _ in range(50):
                low, high = ((low + high) / 2, high) if deep((low + high) / 2) <= room else (low, (low + high) / 2)
            return share, low if deep(low) <= room else 0.0, pairs, least
        # A page grown to exactly what the labels need gives back their least size within rounding
        floor = text.GLYPH_MIN * (1 - 1e-9)
        if math.isfinite(room):
            # A page that cannot grow takes the first of one column and two, or of x's lying and turned, that holds the
            # labels at their least size both along the axis and across it, else whichever holds them largest, which
            # may be below that size, down to half of it, before any is left out, as this page has no other way to
            # show them all
            ways = ([(True, 1)] if self._xrot else [(False, 1), (True, 1)]) if axis == 'x' else [(False, 1)] + (
                [(False, 2)] if len(ticks) > 2 else [])
            best = None
            for turned, step in ways:
                found = fits(turned, step)
                if best is None or min(found[:2]) > min(best[2][:2]):
                    best = (turned, step, found)
                if min(found[:2]) >= floor:
                    break
            turned, step, (share, wide, pairs, least) = best
            if axis == 'y':
                self._gcols = step
            elif not self._xrot:
                self.xrot = 90.0 if turned else 0.0
            self._gscale[axis] = min(1.0, max(min(share, wide), text.GLYPH_MIN / 2))
            thin, need = share < self._gscale[axis] * (1 - 1e-9), self._gscale[axis]
        else:
            share, _, pairs, least = fits(axis == 'x' and bool(self.xrot), self._gcols if axis == 'y' else 1)
            turned = share < floor and axis == 'x' and not self.xrot
            if turned:
                self.xrot = 90.0
                share, _, pairs, least = fits(True, 1)
            staggered = share < floor and axis == 'y' and self._gcols == 1 and len(ticks) > 2
            if staggered:
                self._gcols = 2
                share, _, pairs, least = fits(False, 2)
            self._gscale[axis] = min(1.0, max(share, text.GLYPH_MIN))
            # Labels go only once the page has grown all it may, which it does for turned or staggered ones next pass
            thin, need = share < floor and not turned and not staggered, text.GLYPH_MIN
        if thin and not self.show_all[i]:
            k = max(2, int(math.ceil(max((scaled * need + fixed for scaled, fixed in pairs), default = 0.0) / least)))
            setattr(self, axis + 'ticks', ticks[::k])
            setattr(self, axis + 'labels', labels[::k])
        return (abs(self._gscale[axis] - before[0]) > 0.01 * before[0] or self.xrot != before[1]
                or len(getattr(self, axis + 'ticks')) != before[2] or self._gcols != before[3])

    def _xtick_height(self):
        """How far the x tick labels reach below their start."""
        sx = self._tick_sizes()[0]
        labels, tall = self.xlabels, 0.0
        if self._tick_opts['x'].get('glyphs'):
            glyphs = [self._tick_glyph('x', s) for s in labels]
            tall = max((top - bottom for _, bottom, _, top in (self._glyph_extent('x', g) for g in glyphs if g)),
                       default = 0.0)
            labels = [s for s, g in zip(labels, glyphs) if g is None]
            if not labels:
                return tall
        if self.xrot:
            sin = abs(math.sin(math.radians(self.xrot)))
            return max(tall, max((text.width(s, sx, self.pal.tick_bold, tabular = self._numbers['x']) for s in labels),
                                 default = 0.0) * sin
                       + sx * 0.4)
        return max(tall, sx * (1.0 + 1.25 * max(map(_extra_lines, labels), default = 0)))

    def _ridge_width(self):
        """Room the ridges' names take left of the axes, or nothing without any."""
        pal = self.pal
        if not self._ridge_labels:
            return 0.0
        k = pal.spacing
        return 8.0 * k + max(text.width(text.spaced(pal, s), 7.8 * k if pal.caps else pal.tick_size + k,
                                        pal.text_bold, 0.9 if pal.caps else 0.0) for s in self._ridge_labels)

    def _ylabel_room(self):
        """Room the y axis name takes beside the tick labels."""
        size, pad = self._label_size['y'] or self.pal.label_size, self._label_pad['y']
        if not self.ylabel:
            return 0.0
        lines = 1.25 * _extra_lines(self.ylabel)
        # The name's baseline sits `pad` beyond the tick labels, and its glyphs a size beyond that
        # `pad` runs to the name's lowest glyph, which faces the tick labels: a descender, or deeper, a subscript
        low = 0.37 if any(rise < 0 for _, rise, _ in text.runs(self.ylabel)) else 0.22
        return size * (1.7 + lines) if pad is None else pad + size * (1.0 + lines + low)

    def _colorbar_width(self):
        """Room a colorbar at the right takes, past the tick labels: bar, its ticks, and its label."""
        source = self._cbar['source']
        ticks = nice_ticks(source._scale['lo'], source._scale['hi'], 5)
        label = self.pal.label_size * 1.6 if self._cbar['label'] else 0.0
        k = self.pal.spacing
        labels = _minus(self.pal, tick_labels(ticks))
        split = self._decimals(labels, 8.4 * k, True)
        widest = (max(a for a, _ in split) + max(b for _, b in split) if split else
                  max(text.width(s, 8.4 * k, tabular = True) for s in labels))
        return 12.0 * k + COLORBAR_WIDTH * k + 5.0 * k + widest + label

    def _margins(self):
        """Room the decorations need around the plot area, in points: left, right, bottom, top."""
        pal = self.pal
        off, xtl, xpad = self._tick_room('x')
        _, ytl, ypad = self._tick_room('y')
        sy = self._tick_sizes()[1]
        yw = self._ytick_columns(sy)[0]
        k = pal.spacing
        left, right, bottom, top = 3.0 * k, 3.0 * k, 3.0 * k, 3.0 * k
        # Ticks left without labels still reach out from their axis line
        yside = ypad + yw if self.ylabels else off + ytl
        if self.yticks is not None and len(self.yticks):
            if self._yright:
                right += yside
            else:
                left += yside
        above = self._tick_opts['x'].get('top', False)
        if len(self.xticks) and above:
            bottom += off + xtl
            top += self._tick_opts['x'].get('pad', 5.0 * k) + self._xtick_height() if self.xlabels else 0.0
        elif len(self.xticks):
            bottom += xpad + self._xtick_height() if self.xlabels else off + xtl
        if self.ylabel:
            # A spine held away from the data takes room even without tick labels
            room = self._ylabel_room() + (0.0 if len(self.yticks) else off)
            if self._yright:
                right += room
            else:
                left += room
        if self.xlabel:
            size, gap = self._label_size['x'] or pal.label_size, self._label_pad['x']
            lines = 1.25 * _extra_lines(self.xlabel)
            bottom += size * (1.7 + lines) if gap is None else gap + size * (1.15 + lines)
            bottom += 0.0 if len(self.xticks) else off
        left += self._ridge_width()
        if self.title_text:
            pad = 8.0 * k if self._title_pad is None else self._title_pad
            top += 13.0 * k + pad + 1.25 * self._title_size() * _extra_lines(self.title_text)
        if self._legend_keep and self._legend_loc in ('below', 'right'):
            tw, th = self._legend_size or self._legend_natural()
            if self._legend_loc == 'below':
                bottom += th + 8.0 * k
            else:
                right += tw + 18.0 * k
        # Labels at line ends reach past the plot on its right, ahead of a legend or a color bar there
        right += self._direct_room
        if self._cbar and self._cbar['loc'] == 'right':
            right += self._colorbar_width()
        return left, right, bottom, top

    def _legend_entries(self, items = None):
        """Each item's text size, tracking, shown text, width, glyph or None, and how much taller than a row it is, in
        legend order: of the items the legend shows, or of `items`."""
        pal = self.pal
        size = 7.4 * pal.spacing if pal.caps else pal.tick_size - 0.5 * pal.spacing
        track = 0.7 if pal.caps else 0.0
        out = []
        for s, rgb, kind, marker, area in [self.legend_items[i] for i in self._legend_keep] if items is None else items:
            # Its rows are of one height, but where a glyph needs more
            shown = text.spaced(pal, s).replace('\n', ' ')
            header = kind == 'header'
            g = None if header or not self._legend_glyphs else text.glyph(self._legend_glyphs, s)
            if g is None:
                out.append((shown, rgb, kind, marker, area, text.width(shown, size, pal.text_bold or header, track)
                            + (0.0 if header else 18.0 * pal.spacing), None, 0.0))
                continue
            left, bottom, right, top = g.extent(size * text.GLYPH_SYMBOL, self._legend_cut() is not None or self._cut)
            out.append((shown, rgb, kind, marker, area, right - left + 18.0 * pal.spacing, g,
                        max(0.0, top - bottom + 4.0 * pal.spacing - LEGEND_ROW * pal.spacing)))
        return size, track, out

    def _legend_cut(self):
        """The color of the cut under a legend's glyphs: a sticker's white on a dark page, the page's own color as a
        halo inside the axes, or None."""
        if self.pal.is_dark:
            return 1.0, 1.0, 1.0
        return None if self._legend_loc in ('right', 'below') else self.pal.color('paper')

    def _legend_natural(self):
        """Legend size before the axes are placed: one row or one column."""
        _, _, entries = self._legend_entries()
        widths, row, extra = [e[5] for e in entries], LEGEND_ROW * self.pal.spacing, [e[7] for e in entries]
        return ((max(widths), row * len(widths) + sum(extra)) if self._legend_loc == 'right' else
                (sum(widths), row + max(extra)))

    def _legend_grid(self):
        """Offsets of the legend's items from its top left to the middle of their rows, less half a row, wrapped to the
        axes, and the legend's total size."""
        _, _, w, h = self.rect
        _, _, entries = self._legend_entries()
        widths, k, extra = [e[5] for e in entries], self.pal.spacing, [e[7] for e in entries]
        row = LEGEND_ROW * k
        if self._legend_loc == 'below':
            spots, lines, x, y = [], [0.0], 0.0, 0.0
            for v, more in zip(widths, extra):
                if x and x + v > w:
                    x, y = 0.0, y + row
                    lines.append(0.0)
                lines[-1] = max(lines[-1], more)
                spots.append((x, y, len(lines) - 1))
                x += v + 14.0 * k
            # A line takes what its tallest glyph adds, and those above it push it down
            spots = [(x, y + sum(lines[:n]) + lines[n] / 2) for x, y, n in spots]
            return spots, (w if y else sum(widths) + 14.0 * k * (len(widths) - 1), y + row + sum(lines))
        per = max(1, int(h // row)) if self._legend_loc == 'right' else len(widths)
        tall = any(extra) and self._legend_loc == 'right'
        spots, offset, total_h, start = [], 0.0, 0.0, 0
        while start < len(widths):
            end = start + 1
            # A column holds as many rows as the axes' height does, fewer where glyphs make them taller
            while end < len(widths) and ((end + 1 - start) * row + sum(extra[start:end + 1]) <= h if tall else
                                         end - start < per):
                end += 1
            for i in range(start, end):
                spots.append((offset, (i - start) * row + sum(extra[start:i]) + extra[i] / 2))
            total_h = max(total_h, (end - start) * row + sum(extra[start:end]))
            offset += max(widths[start:end]) + 12.0 * k
            start = end
        return spots, (offset - 12.0 * k, total_h)

    def _fit_labels(self):
        """Turn tick labels that would collide, thin those that would still collide, wrap the legend, and move or make
        room for annotations that would leave the page. True if anything changed."""
        pal = self.pal
        changed = False
        sx, sy = self._tick_sizes()
        glyphs = {axis: bool(self._tick_opts[axis].get('glyphs')) for axis in 'xy'}
        for axis in (axis for axis in 'xy' if glyphs[axis]):
            changed = self._fit_glyphs(axis) or changed
        if len(self.xticks) > 1 and self.xlabels and self._tick_opts['x'].get('fit', True) and not glyphs['x']:
            gap = _gap(self._tx(self.xticks))
            widest = max(text.width(s, sx, pal.tick_bold, tabular = self._numbers['x']) for s in self.xlabels)
            if not self.xrot and widest + 6.0 * pal.spacing > gap:
                self.xrot = 90.0
                changed = True
            if self.xrot and gap < sx * 1.05 and not self.show_all[0]:
                k = int(math.ceil(sx * 1.05 / gap))
                self.xticks, self.xlabels = self.xticks[::k], self.xlabels[::k]
                changed = True
        if (len(self.yticks) > 1 and self.ylabels and not self.show_all[1] and self._tick_opts['y'].get('fit', True)
                and not glyphs['y']):
            gap = _gap(self._ty(self.yticks))
            if gap < sy * 1.05:
                k = int(math.ceil(sy * 1.05 / gap))
                self.yticks, self.ylabels = self.yticks[::k], self.ylabels[::k]
                changed = True
        if self._legend_keep and self._legend_loc:
            size = self._legend_grid()[1]
            if self._legend_size is None or max(abs(u - v) for u, v in zip(size, self._legend_size)) > 0.5:
                self._legend_size = size
                changed = True
        # Labels at line ends take the room they reach past the plot's right
        room = max([float(self._tx(a['x'])) + a['dx'] + self._note_size(a)[0] - self.rect[0] - self.rect[2]
                    for a in self.artists if a.get('direct') == 'end'] + [0.0])
        if abs(room - self._direct_room) > 0.5:
            self._direct_room = room
            changed = True
        page = self.figure.width
        over = [0.0, 0.0]
        # What the right side lacks for all but glycan labels, whose drawings replace them, for Figure._width_unlabeled
        self._over_plain = -math.inf
        if self.title_text:
            over[1] = self.rect[0] + text.width(self.title_text, self._title_size(), pal.title_bold) - (page - 3.0)
            self._over_plain = over[1]
        for a in self.artists:
            if (a['kind'] != 'text' or a['rotation'] or a['frac'] or a['ha'] == 'center' or not math.isfinite(a['x'])
                    or a['spread'] and a.get('glyph') or a.get('direct')):
                continue
            tw = self._note_size(a)[0]
            px, gap = float(self._tx(a['x'])), abs(a['dx']) + 3.0
            natural_right = a['ha'] == 'left'
            # On its own side a label reaches as far as its offset takes it, which may point back across its point (a
            # part of a label laid out beside another, such as a loss written right of a drawing): measured from the
            # offset's size, such a label read as running past the page and was turned over onto what lies left of it
            over_right = px + (a['dx'] + 3.0 if natural_right else gap) + tw - (page - 3.0)
            over_left = tw + (3.0 - a['dx'] if not natural_right else gap) - (px - 3.0)
            # The label's own side if it fits, else the other side if that fits, else whichever side overflows less. A
            # glycan drawing always grows rightward from its label's anchor, so such a label stays and the page grows,
            # as does text placed with `text`, where its caller put it.
            if a['tag'] or a.get('placed') or (over_right if natural_right else over_left) <= 0:
                use_right = natural_right
            elif (over_left if natural_right else over_right) <= 0:
                use_right = not natural_right
            else:
                use_right = over_right <= over_left
            a['_ha'] = 'left' if use_right else 'right'
            a['_dx'] = a['dx'] if use_right == natural_right else (gap if use_right else -gap)
            over[1 if use_right else 0] = max(over[1 if use_right else 0], over_right if use_right else over_left)
            if use_right and not a['tag']:
                self._over_plain = max(self._over_plain, over_right)
        # A page held to a window's size has nowhere to grow
        for side in () if self.figure._held else (0, 1):
            if over[side] > 0.5:
                self._over[side] += over[side]
                changed = True
        return changed

    def _bracket_floors(self):
        """Per bracket, how high the data from its start to its end reaches above the plot's bottom, in points, as two
        arrays: the data's height on the page at the limits as they stand, and what a mark adds above it, such as a
        point's radius."""
        pal, rows = self.pal, []
        log = self._yscale == 'log'
        for a in self.artists:
            k = a['kind']
            if k == 'points':
                x = np.atleast_1d(a['x'])
                area = np.maximum(np.broadcast_to(np.asarray(a['s'], dtype = float), x.shape), 0)
                rows.append((x, x, np.atleast_1d(a['y']),
                             np.sqrt(area * pal.point_scale) / 2 + (a['edge'][1] / 2 if a['edge'] else 0.0)))
            elif k == 'line' and not a['frac'] and a['offset'] is None:
                rows.append((a['x'], a['x'], a['y'], (pal.line_width if a['lw'] is None else a['lw']) / 2))
            elif k == 'bars' and a['horizontal']:
                ends = np.r_[a['b'], a['b'] + a['h']].reshape(2, -1)
                for side in (-0.5, 0.5):
                    rows.append((ends.min(0), ends.max(0), a['x'] + side * a['w'], 0.0))
            elif k == 'bars':
                for v in (a['b'], a['b'] + a['h']):
                    rows.append((a['x'] - a['w'] / 2, a['x'] + a['w'] / 2, v, 0.0))
            elif k in ('box', 'violin'):
                for i, (x, w, stats, v) in enumerate(zip(a['x'], a['w'], a['stats'], a['values'])):
                    shape = a['shapes'][i] if k == 'violin' else None
                    if stats is None:
                        continue
                    shown = v if k == 'violin' or a['offsets'] is not None else stats[5] if a['fliers'] else []
                    ys = np.r_[stats[0], stats[4], shown, [] if shape is None else shape[0][[0, -1]]]
                    reach = math.sqrt(a['sizes'][i] * pal.point_scale) / 2
                    # Across the page, a group spans its values along x and its width up y
                    rows.append((ys.min(), ys.max(), np.array([x - w / 2, x + w / 2]), reach) if a['horizontal'] else
                                (np.full(len(ys), x - w / 2), np.full(len(ys), x + w / 2), ys, reach))
            elif k in ('errors', 'vsegs', 'band', 'stems'):
                names = {'errors': ('lo', 'hi'), 'vsegs': ('y0', 'y1'), 'band': ('y1', 'y2'), 'stems': ('y', 'y')}[k]
                for name in names:
                    rows.append((a['x'], a['x'], a[name], a.get('lw', 0.0) / 2))
            elif k == 'hsegs':
                rows.append((np.minimum(a['x0'], a['x1']), np.maximum(a['x0'], a['x1']), a['y'], a['lw'] / 2))
            elif k == 'hline':
                rows.append((-np.inf, np.inf, a['v'], a['lw'] / 2))
            elif k == 'rect' and not a['frac']:
                for v in (a['y'], a['y'] + a['h']):
                    rows.append((min(a['x'], a['x'] + a['w']), max(a['x'], a['x'] + a['w']), v, 0.0))
            elif k == 'arrow':
                for end in '01':
                    rows.append((a['x' + end], a['x' + end], a['y' + end], 0.0))
        rows = [[np.broadcast_to(np.asarray(v, dtype = float), np.shape(np.atleast_1d(r[2]))) for v in r] for r in rows]
        xl, xr, ys, extra = (np.concatenate([np.zeros(0)] + [r[i] for r in rows]) for i in range(4))
        known = np.isfinite(ys) & ~np.isnan(xl) & ~np.isnan(xr) & ((ys > 0) if log else True)
        xl, xr, extra = xl[known], xr[known], extra[known]
        height = self._ty(ys[known]) - self.rect[1]
        out = []
        for a in self.artists:
            if a['kind'] == 'bracket':
                within = (xr >= a['x0']) & (xl <= a['x1'])
                out.append((height[within], extra[within]))
        return out

    def _bracket_plan(self, lift = 1.0, floors = None, every = False):
        """Where each bracket goes, as {id: (left, right, line, top)} in page points, with the data squeezed toward the
        plot's bottom by `lift`: each sits above the data from its start to its end, and above any placed bracket whose
        span or text it would meet, narrower brackets first. Like what else may leave the axes, a bracket shows while
        its ends are within the x limits, and, unless `every` asks for all of those, while it fits under the axes' top,
        which limits set by hand may not leave room for."""
        k = self.pal.spacing
        y0, h = self.rect[1], self.rect[3]
        brackets = [a for a in self.artists if a['kind'] == 'bracket']
        floors = self._bracket_floors() if floors is None else floors
        placed, out = [], {}
        spans = []
        for a, (height, extra) in zip(brackets, floors):
            if not self._in_x([a['x0'], a['x1']]):
                continue
            px0, px1 = sorted(float(v) for v in self._tx([a['x0'], a['x1']]))
            tw = text.width(a['s'], a['size'], a['bold'])
            mid = (px0 + px1) / 2
            spans.append((round(px1 - px0, 2), a['order'], a, px0, px1, min(px0, mid - tw / 2), max(px1, mid + tw / 2),
                          max(float(np.max(height / lift + extra, initial = 0.0)), 0.0)))
        for _, _, a, px0, px1, left, right, floor in sorted(spans, key = lambda v: v[:2]):
            base = max([floor] + [top for l, r, top in placed if l < right + 3.0 * k and left < r + 3.0 * k])
            line = base + 5.0 * k + a['tip']
            lines = _extra_lines(a['s'])
            # The text's descenders sit just above the line, and its capitals, or its superscripts, reach this far
            rise = max([text.CAP_HEIGHT] + [r + s * text.CAP_HEIGHT for t, r, s in text.runs(a['s'])])
            top = line + (2.0 * k + (0.22 + rise + 1.25 * lines) * a['size'] if a['s'].strip() else a['lw'] / 2)
            if every or top <= h + 0.5:
                placed.append((left, right, top))
                out[id(a)] = (px0, px1, y0 + line, y0 + top)
        return out

    def _fit_brackets(self):
        """Stretch the y axis, and all that share it, until the brackets stack within the plot's height; True if that
        changed the limits."""
        group = self._share['y']
        held = [ax for ax in group if any(a['kind'] == 'bracket' for a in ax.artists)]
        inverted = any(ax._yinv for ax in group)
        if not held or any(ax._ylim is not None and ax._ylim[0 if inverted else 1] is not None for ax in group):
            return False
        need = 1.0
        for ax in held:
            floors, h, pad = ax._bracket_floors(), ax.rect[3], 3.0 * ax.pal.spacing
            # How much more the axis stretches than it does now, from undoing the stretch to the most allowed
            fits = lambda lift: max([v[3] for v in ax._bracket_plan(lift, floors, True).values()],
                                    default = -math.inf) - ax.rect[1] + pad <= h
            lo, hi = 1.0 / ax._lift, BRACKET_LIFT_MAX / ax._lift
            if fits(lo):
                hi = lo
            elif fits(hi):
                for _ in range(48):
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if fits(mid) else (mid, hi)
            need = max(need, hi * ax._lift)
        if abs(need - self._lift) <= 1e-9 * need:
            return False
        for ax in group:
            ax._lift = need
        for ax in group:
            ax._range()
            ax._ticks('y')
        return True

    # --- drawing ---
    def _series(self):
        """The label of the legend entry that names each artist, by the artist's id."""
        return {id(s): self.legend_items[i][0] for i, found in self._members.items() for s in found or ()}

    def _own(self, a):
        """What each of an artist's elements is called, by which a highlight or a hover finds it: a point's name, a
        bar's category, or a box's or violin's group, or None where its elements have no names of their own."""
        k = a['kind']
        if k in ('points', 'vsegs', 'stems') and a.get('names') is not None:
            return [[str(v)] for v in a['names']]
        if k == 'bars':
            cats = {v: s for s, v in self._cats['y' if a['horizontal'] else 'x'].items()}
            return [[cats[x]] if x in cats else [] for x in a['x'].tolist()]
        if k in ('box', 'violin'):
            return [self._group_names(a, i) for i in range(len(a['groups']))]
        return None

    def _focused(self, a, n, own = False):
        """Which of an artist's `n` elements a highlight keeps in full shade: all where none is set or where it names
        the artist's series, else those it names by their own names, from `own` or else _own."""
        focus = a['focus'] if a.get('focus') is not None else self._focus
        if focus is None or self._series().get(id(a)) in focus:
            return np.ones(n, dtype = bool)
        own = self._own(a) if own is False else own
        if own is None:
            return np.zeros(n, dtype = bool)
        return np.array([any(k in focus for k in keys) for keys in own], dtype = bool)

    def _chosen(self, c, a, n, own = False):
        """Which of an artist's `n` elements this pass of drawing takes, and whether they recede: all where nothing is
        highlighted, else the context in the first pass and the focus over it in the second, as the canvas notes: its
        own, and not the axes', as a figure's shadow is drawn on a thread beside its page."""
        if c.passing is None:
            return np.ones(n, dtype = bool), False
        kept = self._focused(a, n, own)
        return (~kept, True) if c.passing == 'context' else (kept, False)

    def _standing(self):
        """Whether a highlight leaves anything of the axes as context, beside which a pale color in focus steps up to
        a stronger shade of its hue (Palette.stand); a highlight of everything leaves the figure as drawn without
        one."""
        if self._focus is None and not any(a.get('focus') is not None for a in self.artists):
            return False
        return any(not self._focused(a, len(self._own(a) or [None])).all() for a in self.artists
                   if a['kind'] in FOCUS_KINDS and a['kind'] != 'heat')

    def _paint(self, c, a, rgb, back):
        """A mark's color as this pass draws it: receded as context, stepped up where it stands in focus beside
        context, or as it is; colors from a scale are data and stay."""
        if back:
            return self.pal.recede(rgb)
        if getattr(c, 'standing', False) and c.passing == 'focus' and not a.get('mapped'):
            return self.pal.stand(rgb)
        return rgb

    def _receding(self, i):
        """Whether a legend entry names only what a highlight leaves as context."""
        label, _, kind, _, _ = self.legend_items[i]
        focus = [f for f in [self._focus] + [a.get('focus') for a in self.artists] if f is not None]
        if not focus or kind == 'header':
            return False
        # What it stands for recedes, unless it is a guide, which a highlight leaves as it is
        members = self._members.get(i)
        if members:
            return not any(a['kind'] not in FOCUS_KINDS or self._focused(
                a, len(np.atleast_1d(a['x'])) if 'x' in a else 1).any() for a in members)
        if any(label in f for f in focus):
            return False
        # A hue level's entry stays while any group in focus holds that level
        for a in self.artists:
            own = self._own(a) if a['kind'] in FOCUS_KINDS else None
            if own and any(label in names for names, kept in zip(own, self._focused(a, len(own), own)) if kept):
                return False
        return True

    def _number(self, v):
        """A number as an interactive SVG's hover text gives it."""
        return _minus(self.pal, ['%.4g' % v])[0]

    def _says(self, axis, v):
        """A value along an axis as an interactive SVG's hover text gives it: its category, or the axis's name, or else
        its letter, and the number."""
        cats = {p: s for s, p in self._cats[axis].items()}
        if float(v) in cats:
            return cats[float(v)]
        name = next((getattr(ax, axis + 'label') for ax in self._share[axis] if getattr(ax, axis + 'label')), '')
        return '%s %s' % (' '.join(''.join(t for t, _, _ in text.runs(name)).split()) or axis, self._number(v))

    def _group_names(self, a, i):
        """A box's, violin's, or strip's group by its category and hue level, without a level where it has none."""
        return [str(g) for g in a['groups'][i] if g is not None and g != '']

    def _group_hover(self, a, i):
        """A box's or a violin's names and hover text: its group, median, quartiles, and number of values."""
        group = tuple(self._group_names(a, i))
        low, q1, median, q3, high = a['stats'][i][:5]
        return group, (' · '.join(group) or self._says('y' if a['horizontal'] else 'x', a['x'][i]),
                       'median ' + self._number(median),
                       'quartiles %s to %s' % (self._number(q1), self._number(q3)), 'n %d' % len(a['values'][i]))

    def _draw_marks(self, c):
        x0, y0, w, h = self.rect
        c.saveState()
        ordered = sorted(self.artists, key = lambda r: (r['z'], r['order']))
        # In PowerPoint, each artist's shapes take its legend entry's name, or its kind and its place among its kind,
        # and in an interactive SVG, its series links them to what else that label names
        names, series, kinds = {}, self._series() if getattr(c, 'shapes', False) else {}, {}
        if getattr(c, 'shapes', False):
            for a in ordered:
                if a['kind'] != 'text':
                    kinds[a['kind']] = kinds.get(a['kind'], -1) + 1
                    names[id(a)] = '%s %s' % (a['kind'], series.get(id(a), kinds[a['kind']]))
        # With a highlight, all that recedes draws first, and what it keeps in full shade over it
        focusing = self._focus is not None or any(a.get('focus') is not None for a in self.artists)
        # What never recedes, such as guides and error bars, keeps its order with the marks in focus: under all of them
        # in the first pass, else among them in the second, so that a highlight of everything draws what none does
        floor = min((a['z'] for a in ordered if a['kind'] in FOCUS_KINDS), default = math.inf)
        c.standing = focusing and self._standing()
        for done in ('context', 'focus') if focusing else (None, ):
            c.passing = done
            for a in ordered:
                k = a['kind']
                # Color 'none' draws nothing but an edge, while a dendrogram's None means the palette's choice
                if k == 'text' or (k != 'dendro' and a.get('rgb', k) is None and not a.get('edge')):
                    continue
                if k not in FOCUS_KINDS and done is not None and (done == 'focus') != (a['z'] >= floor):
                    continue
                n = len(c.ops)
                pad = 9.0 if k in ('points', 'box', 'violin') else 0.0 if k == 'heat' else 1.0
                # A named element stays a bare path, for the code that looks for it in the SVG
                gid = a.get('gid')
                free = gid or k == 'image' or not a.get('clip', True)
                c.state['clip'] = None if free else [((x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad),
                                                      c.state['ctm'])]
                getattr(self, '_d_' + k)(c, a)
                for op in c.ops[n:]:
                    op['gid'] = gid
                    # What recedes casts no shadow
                    if (k not in ('hline', 'vline', 'vspan', 'hspan', 'image', 'contour', 'aline', 'rect', 'bracket')
                            and not (done == 'context' and k in FOCUS_KINDS) and not op.get('grain')):
                        op['mark'] = True
                marks.tag(c, n, artist = names.get(id(a)), **({'series': series[id(a)]} if id(a) in series else {}))
        c.passing = None
        c.restoreState()

    def _in_x(self, v):
        """Whether x values lie within the x limits, as what may leave the axes must once zooming has left it behind."""
        lo, hi = sorted(self.xlim)
        return bool(np.all((np.asarray(v) >= lo) & (np.asarray(v) <= hi)))

    def _d_line(self, c, a):
        pick, back = self._chosen(c, a, 1)
        if not pick[0]:
            return
        rgb = self._paint(c, a, a['rgb'], back)
        mask = np.isfinite(a['x']) & np.isfinite(a['y'])
        # A log axis has no place for zero or below, so those points break the line too
        if self._xscale == 'log':
            mask &= a['x'] > 0
        if self._yscale == 'log' and not a['frac']:
            mask &= a['y'] > 0
        if not mask.any() or (not a['clip'] and not self._in_x(a['x'][mask])):
            return
        _, y0, _, h = self.rect
        xs, ys = self._tx(a['x']), y0 + a['y'] * h if a['frac'] else self._ty(a['y'])
        if a['offset'] is not None:
            xs, ys = xs + a['offset'][:, 0], ys + a['offset'][:, 1]
        for run in _runs(mask):
            if run.stop - run.start > 1:
                marks.line(c, self.pal, xs[run], ys[run], rgb, a['lw'], a['alpha'], a['dash'])

    def _d_vsegs(self, c, a):
        pick, back = self._chosen(c, a, len(a['x']))
        by_color, sticks = {}, []
        for i, (x, lo, hi, rgb) in enumerate(zip(self._tx(a['x']), self._ty(a['y0']), self._ty(a['y1']), a['rgb'])):
            if pick[i] and rgb is not None and np.isfinite((x, lo, hi)).all():
                rgb = self._paint(c, a, rgb, back)
                by_color.setdefault(rgb, []).append(((x, x), (lo, hi)))
                sticks.append((i, x, lo, hi, rgb))
        n = len(c.ops)
        for rgb, lines in by_color.items():
            marks.polylines(c, lines, rgb, a['lw'], a['alpha'], cap = 0, dash = a['dash'])
        if getattr(c, 'hover', False) and sticks:
            # As for points drawn in groups, the sticks fade with their panel, and an unpainted line over each, which
            # the pointer finds, fills in while it is lit
            marks.tag(c, n, hover = ((), ()))
            for i, x, lo, hi, rgb in sticks:
                at, own = len(c.ops), () if a['names'] is None else (str(a['names'][i]), )
                marks.polyline(c, [x, x], [lo, hi], rgb, a['lw'], 0.0, cap = 0)
                marks.tag(c, at, part = i, hit = True, hover = (own, own + (self._says('x', a['x'][i]),
                                                                            self._says('y', a['y1'][i]))))

    def _d_vspan(self, c, a):
        _, y0, _, h = self.rect
        x0, x1 = float(self._tx(a['x0'])), float(self._tx(a['x1']))
        if math.isfinite(x0 + x1):
            marks.polygon(c, [x0, x1, x1, x0], [y0, y0, y0 + h, y0 + h], a['rgb'], a['alpha'])

    def _d_hspan(self, c, a):
        x0, _, w, _ = self.rect
        y0, y1 = float(self._ty(a['y0'])), float(self._ty(a['y1']))
        if math.isfinite(y0 + y1):
            marks.polygon(c, [x0, x0 + w, x0 + w, x0], [y0, y0, y1, y1], a['rgb'], a['alpha'])

    def _d_image(self, c, a):
        _, y0, _, h = self.rect
        lo, hi = sorted(self.ylim)
        if not self._in_x(a['x']) or (not a['frac'] and not lo <= a['y'] <= hi):
            return
        cy = y0 + a['y'] * h if a['frac'] else float(self._ty(a['y']))
        op = dict(c.state)
        op.update(kind = 'image', w = a['nx'], h = a['ny'], rgb = a['rgb'], alpha = a['alpha'],
                  ctm = (a['w'], 0.0, 0.0, a['h'], float(self._tx(a['x'])) + a['dx'] - a['w'] / 2,
                         cy + a['dy'] - a['h'] / 2))
        c.ops.append(op)

    def _visible(self, b):
        """Which of a scatter artist's points are drawn: a point beyond the limits is not, while one at the edge is
        drawn whole, which the clip's margin allows."""
        mask = np.isfinite(b['x']) & np.isfinite(b['y'])
        for v, (lo, hi) in ((b['x'], sorted(self.xlim)), (b['y'], sorted(self.ylim))):
            slack = 1e-9 * (hi - lo)
            mask &= (v >= lo - slack) & (v <= hi + slack)
        return mask

    def hidden(self, a):
        """Which of a scatter artist's points a density field hides on the page as laid out, as a boolean array over
        its points, for whatever answers the mouse, such as slate's dot map: a hidden point is part of the field, not a
        dot of its own. Kept for the page as laid out."""
        crowd = a.get('crowd') if a['kind'] == 'points' else None
        empty = np.zeros(len(np.atleast_1d(a.get('x', ()))), dtype = bool)
        if crowd is None:
            return empty
        key = (self.rect, tuple(self.xlim), tuple(self.ylim), crowd)
        if self._hidden_points[0] != key:
            # The field as the call's first artist lays it, on the clip that the marks are drawn in
            call = [b for b in self.artists if b['kind'] == 'points' and b.get('crowd') == crowd
                    and (b['rgb'] is not None or b['edge'])]
            masks = [self._visible(b) for b in call]
            parts = [(self._tx(b['x'][m]), self._ty(b['y'][m]), np.sqrt(np.maximum(np.asarray(
                b['s'] if np.ndim(b['s']) == 0 else np.asarray(b['s'])[m], dtype = float), 0.0) * self.pal.point_scale)
                      / 2.0, b['rgb'] if b['rgb'] is not None else b['edge'][0]) for b, m in zip(call, masks)]
            x0, y0, w, h = self.rect
            found = {}
            for b, m, (_, keep, _) in zip(call, masks, marks.crowd(self.pal, parts, (x0 - 9.0, y0 - 9.0, w + 18.0,
                                                                                  h + 18.0))):
                out = np.zeros(len(b['x']), dtype = bool)
                out[np.flatnonzero(m)[~np.asarray(keep, dtype = bool)]] = True
                found[id(b)] = out
            self._hidden_points = (key, found)
        return self._hidden_points[1].get(id(a), empty)

    def _d_points(self, c, a):
        # A rasterized scatter is one image in vector output, which its first artist draws for all of them
        spec = a.get('raster') if not getattr(c, 'stamp_dpi', None) else None
        group = [b for b in self.artists if b['kind'] == 'points' and b.get('raster') == spec] if spec else [a]
        if group[0] is not a:
            return
        # Where a call's points pile up, its first artist lays a field for each color under all of them, once, under
        # the context and the focus alike where a highlight draws them in turn
        crowd = a.get('crowd')
        call = ([b for b in self.artists if b['kind'] == 'points' and b.get('crowd') == crowd] if crowd is not None
                else [])
        lays = call and call[0] is a and c.passing != 'focus'
        drawn, kept = [], {}
        for b in call if lays else group:
            mask = self._visible(b)
            drawn.append((self._tx(b['x'][mask]), self._ty(b['y'][mask]),
                          b['s'] if np.ndim(b['s']) == 0 else np.asarray(b['s'])[mask], b, mask))
        layer, clip = c, c.state['clip'][0][0] if c.state['clip'] else (0.0, 0.0, c.width, c.height)
        if spec:
            found = self._layer(drawn, spec[0], clip)
            if found is None:
                return
            layer, drawn, place = found
        if lays:
            # Hollow rings pile up in their edge's color, and what a highlight leaves as context recedes
            colored = [(d, d[3]['rgb'] if d[3]['rgb'] is not None else d[3]['edge'][0]) for d in drawn
                       if d[3]['rgb'] is not None or d[3]['edge']]
            colored = [(d, rgb, c.passing == 'context' and not self._focused(d[3], len(d[3]['x'])).all())
                       for d, rgb in colored]
            parts = [(px, py, np.sqrt(np.maximum(np.asarray(s, dtype = float), 0.0) * self.pal.point_scale) / 2.0,
                      self.pal.recede(rgb) if back else rgb) for (px, py, s, _, _), rgb, back in colored]
            fields = marks.crowd(self.pal, parts, layer.state['clip'][0][0] if spec else clip)
            # The points a field holds lie under it, where its paler bands let them show through, drawn as pixels: on
            # the page in raster output, else as one image of their own. Flat in their field's color, they build up to
            # a solid ground where they are many, which costs nothing to store, and to a grain where they thin out
            # Under the veil they cast no shadow, and a window leaves them out while a gesture runs
            under = [] if getattr(c, 'shadow', False) or getattr(self.figure, '_draft', False) else [
                (px[veiled], py[veiled], s if np.ndim(s) == 0 else np.asarray(s)[veiled], b, rgb)
                for ((px, py, s, b, _), _, _), (_, _, _, rgb), (_, _, veiled) in zip(colored, parts, fields)
                if veiled.any()]
            beneath = (self._layer(under, marks.CROWD_DPI, clip) if under and not spec
                       and not getattr(c, 'stamp_dpi', None) else (layer, under, None))
            flat = self.pal
            n = len(c.ops)
            for px, py, s, b, rgb in beneath[1] if beneath else []:
                marks.points(beneath[0], flat, px, py, s, rgb, b['alpha'], b['marker'])
            if beneath and beneath[2] is not None:
                self._image(c, *beneath[::2])
            for op in c.ops[n:]:
                op['grain'] = True
            for (bands, keep, _), ((_, _, _, b, _), _, _) in zip(fields, colored):
                kept[id(b)] = keep
                if bands is not None:
                    marks.crowd_op(layer, self.pal, bands)
            c.crowd = getattr(c, 'crowd', {})
            c.crowd.update(kept)
            drawn = [d for d in drawn if d[3] is a or spec]
        elif call:
            kept = getattr(c, 'crowd', {})
        for px, py, s, b, mask in drawn:
            keep = kept.get(id(b))
            # With a highlight, the context recedes and the focus draws over it
            pick, back = self._chosen(c, b, len(b['x']))
            # Points a highlight names one by one draw over the field, which recedes under them, as they may lie in it
            single = c.passing == 'focus' and not self._focused(b, len(b['x'])).all()
            keep = pick[mask] if keep is None or single else keep & pick[mask]
            # Each point's own name or place in the call's data, which PowerPoint names its shape by, and in an
            # interactive SVG, its name, series, and place, which its hover text gives
            ids = hovers = None
            if getattr(c, 'shapes', False):
                ids = np.flatnonzero(mask) if b.get('names') is None else b['names'][mask]
            if getattr(c, 'hover', False):
                series, named = self._series().get(id(b)), b.get('names') is not None
                # Axes switched off carry no meaning in their positions, as in a network
                where = 'ticks' not in self._hidden
                hovers = [((str(t), ) if named else (), tuple(v for v in (str(t) if named else None, series) if v) + (
                    (self._says('x', u), self._says('y', w)) if where else ()))
                    for t, u, w in zip(ids, b['x'][mask], b['y'][mask])]
            if not keep.all():
                px, py, s = px[keep], py[keep], s if np.ndim(s) == 0 else s[keep]
                ids = None if ids is None else ids[keep]
                hovers = None if hovers is None else [h for h, k in zip(hovers, keep) if k]
            rgb, edge = b['rgb'], b['edge']
            rgb = None if rgb is None else self._paint(c, b, rgb, back)
            edge = None if edge is None else (self._paint(c, b, edge[0], back), edge[1])
            marks.points(layer, self.pal, px, py, s, rgb, b['alpha'], b['marker'], edge, ids, hovers)
        if spec:
            self._image(c, layer, place)

    def _layer(self, drawn, dpi, clip):
        """A canvas of its own for points that become one image, on a grid of `dpi` over the pixels that they can reach
        within the clip, as (canvas, `drawn` with each (x, y, area, ...) moved onto it, its place), or None where they
        reach no pixel."""
        xs, ys = np.concatenate([d[0] for d in drawn]), np.concatenate([d[1] for d in drawn])
        if not len(xs):
            return None
        bx, by, bw, bh = clip
        sc = dpi / 72.0
        reach = 1.4 * math.sqrt(max(float(np.max(d[2], initial = 0.0)) for d in drawn) * self.pal.point_scale) / 2 + 9.0
        px0, py0 = math.floor(max(bx, xs.min() - reach) * sc), math.floor(max(by, ys.min() - reach) * sc)
        nx = math.ceil(min(bx + bw, xs.max() + reach) * sc) - px0
        ny = math.ceil(min(by + bh, ys.max() + reach) * sc) - py0
        if nx <= 0 or ny <= 0:
            return None
        ox, oy = px0 / sc, py0 / sc
        layer = pdfwrite.Canvas(BytesIO(), pagesize = (nx / sc, ny / sc))
        layer.stamp_dpi = dpi
        layer.state['clip'] = [((bx - ox, by - oy, bw, bh), layer.state['ctm'])]
        return layer, [(px - ox, py - oy) + tuple(rest) for px, py, *rest in drawn], (nx, ny, sc, ox, oy)

    def _image(self, c, layer, place):
        """What a canvas from _layer drew, as one image op on the page."""
        nx, ny, sc, ox, oy = place
        if len(layer.ops) == 1 and layer.ops[0]['kind'] == 'image':
            # Stamped, it is an image already
            op = layer.ops[0]
            w, h, rgb, alpha, (cw, _, _, ch, cx, cy) = op['w'], op['h'], op['rgb'], op['alpha'], op['ctm']
        elif layer.ops:
            px = raster.render(layer.ops, nx / sc, ny / sc, sc, sc, None)
            w, h, rgb, alpha = nx, ny, px[..., :3].tobytes(), px[..., 3].tobytes()
            cw, ch, cx, cy = nx / sc, ny / sc, 0.0, 0.0
        else:
            return
        op = dict(c.state)
        op.update(kind = 'image', w = w, h = h, rgb = rgb, alpha = alpha, fill_alpha = 1.0,
                  ctm = (cw, 0.0, 0.0, ch, cx + ox, cy + oy))
        c.ops.append(op)

    def _d_band(self, c, a):
        pick, back = self._chosen(c, a, 1)
        if not pick[0]:
            return
        for run in _runs(np.isfinite(a['x']) & np.isfinite(a['y1']) & np.isfinite(a['y2'])):
            x, y1, y2 = a['x'][run], a['y1'][run], a['y2'][run]
            if len(x) > 1:
                marks.polygon(c, np.r_[self._tx(x), self._tx(x[::-1])], np.r_[self._ty(y1), self._ty(y2[::-1])],
                              self._paint(c, a, a['rgb'], back), a['alpha'])

    def _base(self, v, scale, limits):
        """Where a bar or a stem starts: a log axis has no place for zero or below, so the start of the axis."""
        return v if scale != 'log' or v > 0 else min(limits)

    def _d_bars(self, c, a):
        room = np.full(a['x'].shape, np.inf)
        if self.pal.bar == 'needle' and a['x'].size > 1:
            # A needle's wide top stays clear of the bars beside it, as those of a histogram touch
            order = np.argsort(a['x'])
            free = (np.diff(a['x'][order]) - (a['w'][order][:-1] + a['w'][order][1:]) / 2) / 2
            room[order[:-1]] = np.maximum(free, 0.0)
            room[order[1:]] = np.fmin(room[order[1:]], np.maximum(free, 0.0))
        pick, back = self._chosen(c, a, len(a['x']))
        hover = getattr(c, 'hover', False)
        own, series = (self._own(a), self._series().get(id(a))) if hover else (None, None)
        across, along = ('y', 'x') if a['horizontal'] else ('x', 'y')
        for i, (x, b, v, w, rgb, free) in enumerate(zip(a['x'], a['b'], a['h'], a['w'], a['rgb'], room)):
            if rgb is None or not pick[i] or not np.isfinite((x, b, v, w)).all():
                continue
            rgb = self._paint(c, a, rgb, back)
            n = len(c.ops)
            if a['horizontal']:
                x1, x2 = self._tx(self._base(b, self._xscale, self.xlim)), self._tx(b + v)
                y1, y2 = self._ty(x - w / 2), self._ty(x + w / 2)
            else:
                x1, x2 = self._tx(x - w / 2), self._tx(x + w / 2)
                y1, y2 = self._ty(self._base(b, self._yscale, self.ylim)), self._ty(b + v)
            lx, ly, bw, bh = float(min(x1, x2)), float(min(y1, y2)), float(abs(x2 - x1)), float(abs(y2 - y1))
            if a['stacked']:
                marks.segment(c, self.pal, lx, ly, bw, bh, rgb, 0.0 if b == 0 else max(self.pal.heat_gap, 0.8),
                              a['alpha'])
            else:
                # Its base lies at the far end if it runs down or left on the page
                marks.bar(c, self.pal, lx, ly, bw, bh, rgb, a['horizontal'], a['alpha'],
                          x2 < x1 if a['horizontal'] else y2 < y1, min(0.7, free / w) if w else 0.0)
            marks.tag(c, n, part = i)
            if hover:
                marks.tag(c, n, hover = (tuple(own[i]), tuple(t for t in (
                    own[i][0] if own[i] else self._says(across, x), series, self._says(along, v)) if t is not None)))

    def _d_errors(self, c, a):
        for x, lo, hi in zip(self._tx(a['x']), self._ty(a['lo']), self._ty(a['hi'])):
            if np.isfinite((x, lo, hi)).all():
                marks.polyline(c, [x, x], [lo, hi], a['rgb'], a['lw'], cap = 0)
                for v in (lo, hi):
                    marks.polyline(c, [x - a['cap'], x + a['cap']], [v, v], a['rgb'], a['lw'], cap = 0)

    def _d_contour(self, c, a):
        if a['lines']:
            marks.polylines(c, [(self._tx(xs), self._ty(ys)) for xs, ys in a['lines']], a['rgb'], a['lw'], a['alpha'],
                            dash = a['dash'])

    def _paired(self, rgb):
        """A group's mark in a paler shade under points in its own color, or, where its color is the palest already, in
        that color under points a shade darker."""
        dark, _, light = self.pal.shades(rgb)
        return (light, rgb) if light != rgb else (rgb, dark)

    def _across(self, a):
        """For a box, strip, or violin plot, the maps to the page across its groups and along its values, and the limits
        across and along, which a plot across the page swaps."""
        if a['horizontal']:
            return self._ty, self._tx, sorted(self.ylim), sorted(self.xlim)
        return self._tx, self._ty, sorted(self.xlim), sorted(self.ylim)

    def _spread(self, a):
        """The offsets of a box, strip, or violin plot's points from their group's place, in data units, and their
        sizes: as spread when plotted, or packed as a swarm, for the page as it stands. A swarm places the values in
        order, each as near its group's middle as it can lie without meeting a point placed before it. Where a row would
        spread past the group's width, as many equal values do, the points within a diameter of it squeeze into the
        width, and the rest of the group keeps its places."""
        if not a['swarm']:
            return a['offsets'], a['sizes']
        key = (self.rect, tuple(self.xlim), tuple(self.ylim))
        if a.get('_swarm', (None, ))[0] == key:
            return a['_swarm'][1], a['sizes']
        across, along, _, _ = self._across(a)
        out = []
        for x, v, w, size in zip(a['x'], a['values'], a['w'], a['sizes']):
            d = math.sqrt(size * self.pal.point_scale) + 0.5 * self.pal.spacing
            order = np.argsort(along(v), kind = 'stable')
            ys, xs, start = np.asarray(along(v), dtype = float)[order], np.zeros(len(v)), 0
            for i in range(len(ys)):
                # Only points less than a diameter lower can meet this one
                while ys[i] - ys[start] >= d:
                    start += 1
                dy, near = ys[i] - ys[start:i], xs[start:i]
                if not len(near):
                    continue
                reach = np.sqrt(d * d - dy * dy)
                spots = np.r_[0.0, near - reach, near + reach]
                spots = spots[np.argsort(np.abs(spots), kind = 'stable')]
                # Only the few points less than a diameter across from a spot can meet it, as no two placed ones are
                # closer, which keeps many equal values from costing the cube of their number
                k = np.argsort(near)
                nx, ny = near[k], dy[k]
                lo, hi = np.searchsorted(nx, spots - d, 'right'), np.searchsorted(nx, spots + d, 'left')
                seen = np.minimum(lo[:, None] + np.arange(max(int((hi - lo).max()), 1)), len(nx) - 1)
                meets = (seen < hi[:, None]) & ((spots[:, None] - nx[seen]) ** 2 + ny[seen] ** 2 < d * d * (1 - 1e-9))
                # The farthest spot out is always free
                xs[i] = spots[np.argmax(~meets.any(axis = 1))]
            unit = abs(float(across(x + 0.5) - across(x - 0.5)))
            reach = np.abs(xs)
            lo, hi = np.searchsorted(ys, ys - d, 'right'), np.searchsorted(ys, ys + d, 'left')
            widest = np.array([reach[a:b].max() for a, b in zip(lo, hi)])
            xs *= np.minimum(1.0, w / 2 * unit / np.maximum(widest, 1e-12))
            u = np.empty(len(v))
            u[order] = xs / unit
            out.append(u)
        a['_swarm'] = (key, out)
        return out, a['sizes']

    def _group_points(self, c, a, i, offsets, size, rgb):
        """A group's values as points at their spread places, leaving out those beyond the limits."""
        across, along, (plo, phi), (vlo, vhi) = self._across(a)
        v, px = a['values'][i], a['x'][i] + offsets
        keep = (v >= vlo) & (v <= vhi) & (px >= plo) & (px <= phi)
        # Named in PowerPoint by their group and their own names or places in it, and found by a highlight or a hover
        # by their own names and their group's
        named = a['names'] is not None and a['names'][i] is not None
        names = a['names'][i] if named else range(len(v))
        group = self._group_names(a, i)
        pick, back = self._chosen(c, a, len(v), [[str(t)] + group for t in names] if named else [group] * len(v))
        keep &= pick
        ids = ['%s %s' % (' '.join(group), t) for t, k in zip(names, keep) if k]
        hovers = None
        if getattr(c, 'hover', False):
            axis, title = 'x' if a['horizontal'] else 'y', ' · '.join(group)
            hovers = [((str(t), ) if named else (), (str(t), title, self._says(axis, u)) if named else
                       (title, self._says(axis, u))) for t, u, k in zip(names, v, keep) if k]
        n = len(c.ops)
        marks.points(c, self.pal, *marks.turn(a['horizontal'], across(px[keep]), along(v[keep])), size,
                     self._paint(c, a, rgb, back), a['alpha'], ids = ids, hovers = hovers)
        # In an interactive SVG, they light up with their group, which they belong to as a series
        marks.tag(c, n, series = tuple(group))

    def _d_box(self, c, a):
        dots = a['offsets'] is not None
        offsets, sizes = self._spread(a)
        across, along, _, _ = self._across(a)
        pick, back = self._chosen(c, a, len(a['x']))
        for i, (x, w, stats, rgb) in enumerate(zip(a['x'], a['w'], a['stats'], a['rgb'])):
            if stats is None:
                continue
            n = len(c.ops)
            x0, x1 = float(across(x - w / 2)), float(across(x + w / 2))
            fill, ink = self._paired(rgb) if dots and a['box'] else (rgb, rgb)
            if a['box'] and pick[i]:
                # In order up or across the page, which a turned axis reverses
                marks.box(c, self.pal, min(x0, x1), abs(x1 - x0), sorted(float(along(v)) for v in stats[:5]),
                          self._paint(c, a, fill, back), a['alpha'], a['horizontal'])
            if dots:
                self._group_points(c, a, i, offsets[i], sizes[i], ink)
            elif a['fliers'] and len(stats[5]) and pick[i]:
                marks.points(c, self.pal, *marks.turn(a['horizontal'], np.full(len(stats[5]), (x0 + x1) / 2),
                                                      along(stats[5])), 12.0, self._paint(c, a, rgb, back),
                             a['alpha'])
            marks.tag(c, n, part = ' '.join(self._group_names(a, i)))
            if getattr(c, 'hover', False):
                marks.tag(c, n, hover = self._group_hover(a, i))

    def _d_violin(self, c, a):
        pal = self.pal
        offsets, sizes = self._spread(a)
        across, along, _, _ = self._across(a)
        pick, back = self._chosen(c, a, len(a['x']))
        for i, (x, w, stats, shape, rgb) in enumerate(zip(a['x'], a['w'], a['stats'], a['shapes'], a['rgb'])):
            if stats is None:
                continue
            dots = a['inner'] in ('points', 'swarm')
            fill, ink = self._paired(rgb) if dots else (rgb, rgb)
            cx = float(across(x))
            n = len(c.ops)
            if shape is not None and pick[i]:
                grid, half = shape
                ys = along(grid)
                left, right = across(x - half), across(x + half)
                marks.violin(c, pal, np.r_[left, right[::-1]], np.r_[ys, ys[::-1]], cx,
                             float(np.max(np.abs(right - left))) / 2, self._paint(c, a, fill, back), a['alpha'],
                             a['horizontal'])
            if dots or shape is None:
                self._group_points(c, a, i, offsets[i], sizes[i], ink)
            elif a['inner'] == 'box' and pick[i]:
                low, q1, median, q3, high = sorted(float(along(v)) for v in stats[:5])
                span = abs(float(across(x + w / 2) - across(x - w / 2)))
                marks.inner_box(c, pal, cx, min(max(span * 0.08, 2.5), 7.0), (low, q1, median, q3, high),
                                self._paint(c, a, rgb, back), a['alpha'], a['horizontal'])
            marks.tag(c, n, part = ' '.join(self._group_names(a, i)))
            if getattr(c, 'hover', False):
                marks.tag(c, n, hover = self._group_hover(a, i))

    def _d_stems(self, c, a):
        pal = self.pal
        pick, back = self._chosen(c, a, len(a['x']))
        rgb = self._paint(c, a, a['rgb'], back)
        keep = np.flatnonzero(np.isfinite(a['x']) & np.isfinite(a['y']) & pick)
        if not len(keep):
            return
        xs, ys = self._tx(a['x'][keep]), self._ty(a['y'][keep])
        zero = float(self._ty(self._base(0.0, self._yscale, self.ylim)))
        scale = {'needle': 0.35, 'flat': 1.6}.get(pal.bar, 1.0)
        for i, x, y, u, v in zip(keep.tolist(), xs, ys, a['x'][keep], a['y'][keep]):
            n = len(c.ops)
            if pal.bar in ('deep', 'soft'):
                dark, base, light = pal.shades(rgb)
                hw = a['lw'] / 2.0
                grad = ((x, zero, x, y), (dark, light) if pal.bar == 'deep' else (light, base))
                marks.polygon(c, [x - hw, x + hw, x + hw, x - hw], [zero, zero, y, y], None, 1.0, grad)
            elif pal.bar == 'sear':
                dark, base, light = pal.shades(rgb)
                hw = a['lw'] / 2.0
                # As a bar: one scale across the plot, on either side of zero, so that a tall stem browns at its tip
                (_, by, _, bh), _ = c.state['clip'][0] if c.state['clip'] else ((0.0, min(zero, y), 0.0, abs(y - zero)),
                                                                                None)
                grad = ((x, zero, x, by if y < zero else by + bh), (light, base, dark))
                marks.polygon(c, [x - hw, x + hw, x + hw, x - hw], [zero, zero, y, y], None, 1.0, grad)
            else:
                marks.polyline(c, [x, x], [zero, y], rgb, a['lw'] * scale, cap = 0)
            marks.tag(c, n, part = i)
            if getattr(c, 'hover', False):
                own = () if a.get('names') is None else (str(a['names'][i]), )
                marks.tag(c, n, hover = (own, own + (self._says('x', u), self._says('y', v))))
        if a['cap']:
            marks.points(c, pal, xs, ys, 26.0, rgb)

    def _d_hsegs(self, c, a):
        for y, x0, x1 in zip(self._ty(a['y']), self._tx(a['x0']), self._tx(a['x1'])):
            if np.isfinite((y, x0, x1)).all():
                marks.line(c, self.pal, [x0, x1], [y, y], a['rgb'], a['lw'], a['alpha'], a['dash'])

    def _d_arrow(self, c, a):
        x0, y0 = float(self._tx(a['x0'])), float(self._ty(a['y0']))
        x1, y1 = float(self._tx(a['x1'])), float(self._ty(a['y1']))
        n = math.hypot(x1 - x0, y1 - y0)
        if not math.isfinite(x0 + y0 + x1 + y1) or n <= sum(a['shrink']):
            return
        (s0, s1), ux, uy = a['shrink'], (x1 - x0) / n, (y1 - y0) / n
        at = len(c.ops)
        marks.arrow(c, self.pal, x0 + ux * s0, y0 + uy * s0, x1 - ux * s1, y1 - uy * s1, a['rgb'], a['lw'], a['alpha'],
                    a['head'], a['bend'])
        if getattr(c, 'hover', False) and a.get('name') is not None:
            # A thin arrow is hard to find, so an unpainted line along it, part of the same element, widens its grip
            if not a['bend']:
                marks.polyline(c, [x0 + ux * s0, x1 - ux * s1], [y0 + uy * s0, y1 - uy * s1], a['rgb'],
                               max(a['lw'], ARROW_GRIP), 0.0, cap = 0)
            marks.tag(c, at, hover = ((str(a['name']), ), (str(a['name']), )))

    def _d_aline(self, c, a):
        x0, y0, w, h = self.rect
        px0, py0 = float(self._tx(a['x0'])), float(self._ty(a['y0']))
        dx, dy = float(self._tx(a['x1'])) - px0, float(self._ty(a['y1'])) - py0
        if not math.isfinite(px0 + py0 + dx + dy) or not (dx or dy):
            return
        # Far enough each way to cross the whole axes, which the clip then trims
        reach = 2.0 * math.hypot(w, h) / math.hypot(dx, dy)
        marks.polyline(c, [px0 - reach * dx, px0 + reach * dx], [py0 - reach * dy, py0 + reach * dy], a['rgb'],
                       a['lw'], cap = 0, dash = a['dash'])

    def _d_rect(self, c, a):
        x0, y0, w, h = self.rect
        if a['frac']:
            xs, ys = [x0 + a['x'] * w, x0 + (a['x'] + a['w']) * w], [y0 + a['y'] * h, y0 + (a['y'] + a['h']) * h]
        else:
            xs = [float(self._tx(a['x'])), float(self._tx(a['x'] + a['w']))]
            ys = [float(self._ty(a['y'])), float(self._ty(a['y'] + a['h']))]
        if not math.isfinite(sum(xs) + sum(ys)):
            return
        (lx, hx), (ly, hy) = sorted(xs), sorted(ys)
        r = max(0.0, min(a['radius'], (hx - lx) / 2, (hy - ly) / 2))
        q = r * (1 - marks.KAPPA)
        p = c.beginPath()
        p.moveTo(lx + r, ly)
        # Each side, then the quarter circle into the next one
        for start, c1, c2, end in (((hx - r, ly), (hx - q, ly), (hx, ly + q), (hx, ly + r)),
                                   ((hx, hy - r), (hx, hy - q), (hx - q, hy), (hx - r, hy)),
                                   ((lx + r, hy), (lx + q, hy), (lx, hy - q), (lx, hy - r)),
                                   ((lx, ly + r), (lx, ly + q), (lx + q, ly), (lx + r, ly))):
            p.lineTo(*start)
            if r:
                p.curveTo(*c1, *c2, *end)
        p.close()
        edge = a['edge'] or (None, None)
        c.saveState()
        marks._style(c, a['rgb'], a['alpha'], edge[0], edge[1], a['alpha'], join = 1)
        c.drawPath(p, fill = a['rgb'] is not None, stroke = a['edge'] is not None)
        c.restoreState()

    def _d_bracket(self, c, a):
        placed = self._bracket_plan().get(id(a))
        if placed is None:
            return
        px0, px1, line, _ = placed
        k = self.pal.spacing
        marks.polyline(c, [px0, px0, px1, px1], [line - a['tip'], line, line, line - a['tip']], a['rgb'], a['lw'],
                       cap = 1 if self.pal.spine_round else 0, join = 1 if self.pal.spine_round else 0)
        text.draw(c, (px0 + px1) / 2, line + 2.0 * k, a['s'], a['size'], a['rgb'], a['bold'], ha = 'center',
                  va = 'bottom')

    def _d_hline(self, c, a):
        x0, _, w, _ = self.rect
        y = float(self._ty(a['v']))
        if math.isfinite(y):
            marks.polyline(c, [x0, x0 + w], [y, y], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_vline(self, c, a):
        _, y0, _, h = self.rect
        x = float(self._tx(a['v']))
        if math.isfinite(x):
            marks.polyline(c, [x, x], [y0, y0 + h], a['rgb'], a['lw'], cap = 0, dash = a['dash'])

    def _d_heat(self, c, a):
        ny, nx = a['m'].shape
        # Through the limits, as they decide which way up the cells are and which of them show
        xs, ys = self._tx(np.arange(nx + 1)), self._ty(ny - np.arange(ny + 1))
        # The rows and columns a highlight names keep their shades, and the other cells recede
        rgb, kept = a['rgb'], None
        if c.passing is not None:
            kept = np.ones((ny, nx), dtype = bool)
            focus = a['focus'] if a.get('focus') is not None else self._focus
            if focus is not None:
                named = lambda given, n: np.isin(np.array([str(v) for v in (given or range(n))]), list(focus))
                kept = named(a['rows'], ny)[:, None] | named(a['columns'], nx)[None, :]
                rgb = np.where(kept[..., None], rgb, self.pal.recede(rgb))
        if nx * ny > HEAT_IMAGE_CELLS:
            # One image holds the context and the focus alike
            if c.passing == 'focus':
                return
            op = dict(c.state)
            px = (rgb * 255.0 + 0.5).astype(np.uint8)
            seen = (np.isfinite(a['m']) * 255).astype(np.uint8)
            rows, columns = -1 if ys[-1] > ys[0] else 1, -1 if xs[-1] < xs[0] else 1
            op.update(kind = 'image', w = nx, h = ny, rgb = px[::rows, ::columns].tobytes(),
                      ctm = (float(abs(xs[-1] - xs[0])), 0.0, 0.0, float(abs(ys[-1] - ys[0])),
                             float(min(xs[0], xs[-1])), float(min(ys[0], ys[-1]))),
                      alpha = seen[::rows, ::columns].tobytes())
            c.ops.append(op)
            return
        cw, ch = float(abs(xs[1] - xs[0])), float(abs(ys[1] - ys[0]))
        # Each cell named in PowerPoint by its row and column, so that reordered rows carry their cells along
        rows, columns = a['rows'] or range(ny), a['columns'] or range(nx)
        hover = getattr(c, 'hover', False)
        scale = self._cbar['label'] if self._cbar and self._cbar['label'] else 'value'
        for r in range(ny):
            for j in range(nx):
                if np.isfinite(a['m'][r, j]) and (kept is None or kept[r, j] == (c.passing == 'focus')):
                    n = len(c.ops)
                    marks.heat_cell(c, self.pal, float(min(xs[j], xs[j + 1])), float(min(ys[r], ys[r + 1])), cw, ch,
                                    tuple(rgb[r, j]))
                    marks.tag(c, n, part = '%s | %s' % (rows[r], columns[j]))
                    if hover:
                        marks.tag(c, n, hover = ((str(rows[r]), str(columns[j])) if a['rows'] and a['columns'] else
                                                 (), (str(rows[r]), str(columns[j]), '%s %s' % (
                                                     scale, self._number(a['m'][r, j])))))

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
                top = mix(pal.color('a2'), 0.14 + 0.62 * i / max(n - 1, 1), pal.color('paper'))
                rgb = pal.color('a0')
            peak = float(self._ty(base + np.nanmax(y) * a['overlap']) - self._ty(base))
            marks.ridge(c, pal, tx, float(self._ty(base)), ys, rgb, top, peak)

    def _d_dendro(self, c, a):
        pal = self.pal
        lw = {'deep': 1.2, 'needle': 0.45, 'flat': 2.6, 'soft': 1.5, 'sear': 1.3}[pal.bar]
        rgb = {'deep': pal.color('a1'), 'needle': pal.color('ink'), 'flat': pal.color('a0'), 'soft': pal.color('a2'),
               'sear': pal.color('a1')}[pal.bar]
        lines = {}
        for k, (ic, dc) in enumerate(zip(a['ic'], a['dc'])):
            pos, dep = np.asarray(ic) / 10.0, np.asarray(dc, dtype = float)
            lines.setdefault(rgb if a['rgb'] is None else a['rgb'][k], []).append(
                (self._tx(dep), self._ty(pos)) if a['orient'] == 'left' else (self._tx(pos), self._ty(dep)))
        for color, group in lines.items():
            marks.polylines(c, group, color, lw)

    def _draw_axes(self, c):
        pal = self.pal
        x0, y0, w, h = self.rect
        ink = pal.color(pal.spine_color or pal.ink)
        if self.grid_on and 'grid' not in self._hidden:
            for j, v in enumerate(self.yticks):
                y = float(self._ty(v))
                n = len(c.ops)
                marks.polyline(c, [x0, x0 + w], [y, y], pal.color(pal.faint), 1.1, cap = 0)
                marks.tag(c, n, name = 'grid %s' % (self.ylabels[j] if j < len(self.ylabels) else j))
        off = pal.spine_offset
        cap = 1 if pal.spine_round else 0
        spines = self._spines()
        xt, yt = [float(self._tx(v)) for v in self.xticks], [float(self._ty(v)) for v in self.yticks]
        if 'left' in spines:
            tl = self._tick_room('y')[1]
            lo, hi = (min(yt), max(yt)) if off and yt else (y0, y0 + h)
            n = len(c.ops)
            marks.polyline(c, [x0 - off, x0 - off], [lo, hi], ink, pal.spine_width, cap = cap)
            marks.tag(c, n, name = 'spine left')
            for j, y in enumerate(yt if tl else ()):
                n = len(c.ops)
                marks.polyline(c, [x0 - off - tl, x0 - off], [y, y], ink, pal.spine_width, cap = 0)
                marks.tag(c, n, name = 'tick y %s' % (self.ylabels[j] if j < len(self.ylabels) else j))
        if 'bottom' in spines:
            tl = self._tick_room('x')[1]
            lo, hi = (min(xt), max(xt)) if off and xt else (x0, x0 + w)
            n = len(c.ops)
            marks.polyline(c, [lo, hi], [y0 - off, y0 - off], ink, pal.spine_width, cap = cap)
            marks.tag(c, n, name = 'spine bottom')
            for j, x in enumerate(xt if tl else ()):
                n = len(c.ops)
                marks.polyline(c, [x, x], [y0 - off - tl, y0 - off], ink, pal.spine_width, cap = 0)
                marks.tag(c, n, name = 'tick x %s' % (self.xlabels[j] if j < len(self.xlabels) else j))

    def _draw_text(self, c):
        pal = self.pal
        x0, y0, w, h = self.rect
        tick_rgb = pal.color(pal.tick_color or pal.muted)
        off, xtl, pad = self._tick_room('x')
        _, ytl, ypad = self._tick_room('y')
        size = pal.tick_size
        heat = 'heat' in {a['kind'] for a in self.artists}
        small, ytext = self._tick_sizes()
        ink = pal.color(pal.ink)
        above = self._tick_opts['x'].get('top', False)
        rise = y0 + h + self._tick_opts['x'].get('pad', 5.0 * pal.spacing)
        # Glyphs are cut out as stickers on a dark page, whose SNFG ink would vanish into it
        sticker = (1.0, 1.0, 1.0) if pal.is_dark else None
        for v, s in zip(self.xticks, self.xlabels):
            x = float(self._tx(v))
            n = len(c.ops)
            g = self._tick_glyph('x', s)
            if g is not None:
                left, bottom, right, top = self._glyph_extent('x', g)
                pts = small * text.GLYPH_SYMBOL * self._gscale['x']
                # Its root at the tick once it runs up the page, else centered on it, and its edge facing the axis
                text.draw_glyph(c, g, x if self.xrot else x - (left + right) / 2,
                                rise - bottom if above else y0 - pad - top, pts, sticker)
            else:
                rgb, bold = ((pal.color(self._xemph[s]), True) if s in self._xemph else
                             (ink if heat and self.xrot else tick_rgb, pal.tick_bold))
                figures = self._numbers['x']
                if self.xrot and above:
                    # The label starts at its tick and runs up and away from the axes
                    text.draw(c, x - small * 0.36 * math.sin(math.radians(self.xrot)), rise, s, small, rgb, bold,
                              ha = 'left' if self.xrot > 0 else 'right', rotation = self.xrot, va = 'baseline',
                              tabular = figures)
                elif self.xrot:
                    # The label's end sits at its tick, and it hangs below the axis either way
                    text.draw(c, x + small * 0.36 * math.sin(math.radians(self.xrot)), y0 - pad, s, small, rgb, bold,
                              ha = 'right' if self.xrot > 0 else 'left', rotation = self.xrot, va = 'baseline',
                              tabular = figures)
                else:
                    text.draw(c, x, rise if above else y0 - pad, s, small, rgb, bold, ha = 'center',
                              va = 'bottom' if above else 'top', tabular = figures)
                marks.tag(c, n, name = 'x label ' + s)
            # In an interactive SVG, a category's label, or its glyph, lights up what it names
            if getattr(c, 'hover', False) and s and not self._numbers['x']:
                marks.tag(c, n, hover = ((s, ), ()), tick = True)
        yw, offsets = self._ytick_columns(ytext)
        # Numbers line up on their decimal points, the column's widest fraction in from its edge away from the plot
        split = self._decimals(self.ylabels, ytext)
        point = (yw - max(b for _, b in split) if self._yright else max(b for _, b in split)) if split else 0.0
        split = split or [None] * len(self.ylabels)
        for v, s, o, part in zip(self.yticks, self.ylabels, offsets, split):
            y = float(self._ty(v))
            n = len(c.ops)
            # Where the label's column starts, from the axes outward
            start = x0 + w + ypad - 2.0 * pal.spacing + o if self._yright else x0 - ypad - o
            if o:
                # A staggered label's row runs out to it as a hairline
                way = 1.0 if self._yright else -1.0
                marks.polyline(c, [start - way * (o - pal.spacing), start - way * 1.5 * pal.spacing], [y, y], tick_rgb,
                               0.4 * pal.spacing, 0.7, cap = 0)
            g = self._tick_glyph('y', s)
            if g is not None:
                left, _, right, _ = self._glyph_extent('y', g)
                pts = ytext * text.GLYPH_SYMBOL * self._gscale['y']
                # Its edge facing the axes, and its root level with the tick
                text.draw_glyph(c, g, start - left if self._yright else start - right, y, pts, sticker)
            else:
                rgb, bold = ((pal.color(self._yemph[s]), True) if s in self._yemph else
                             (ink if heat else tick_rgb, pal.tick_bold))
                if part is not None:
                    text.draw(c, (start + point if self._yright else start - point) - part[0], y, s, ytext, rgb, bold,
                              ha = 'left', va = 'center', tabular = True)
                elif self._yright:
                    text.draw(c, start, y, s, ytext, rgb, bold, ha = 'left', va = 'center',
                              tabular = self._numbers['y'])
                else:
                    text.draw(c, start, y, s, ytext, rgb, bold, ha = 'right', va = 'center',
                              tabular = self._numbers['y'])
                marks.tag(c, n, name = 'y label ' + s)
            if getattr(c, 'hover', False) and s and not self._numbers['y']:
                marks.tag(c, n, hover = ((s, ), ()), tick = True)
        label_rgb = pal.color(pal.label_color or pal.ink)
        track = 0.9 if pal.caps else 0.0
        # Long axis names shrink to stay within their cell, centered on the axes
        rx, ry, rw, rh = self._room
        fits = (2 * min(x0 + w / 2 - rx, rx + rw - x0 - w / 2), 2 * min(y0 + h / 2 - ry, ry + rh - y0 - h / 2))
        base = [self._label_size[axis] or pal.label_size for axis in 'xy']
        size_x, size_y = [max(min(LABEL_MIN, given), min(given, given * room / wide)) if wide else given
                          for room, wide, given in zip(fits, [self._axis_name_widths(v)[i] for i, v in enumerate(base)],
                                                       base)]
        if self.xlabel:
            drop = (pad + self._xtick_height() if self.xlabels and not above else off + xtl if len(self.xticks) else
                    off)
            gap = base[0] * 0.55 if self._label_pad['x'] is None else self._label_pad['x']
            n = len(c.ops)
            text.draw(c, x0 + w / 2, y0 - drop - gap, text.spaced(pal, self.xlabel), size_x,
                      label_rgb, pal.label_bold, ha = 'center', va = 'top', tracking = track)
            marks.tag(c, n, name = 'x name')
        if self.ylabel:
            beside = ypad + yw if self.ylabels else off + ytl if len(self.yticks) else off
            # Further lines run away from the axes on the right, and towards them on the left
            ylx = (x0 + w + beside + base[1] * 1.3 if self._yright else
                   x0 - beside - self._ridge_width() - self._ylabel_room() + base[1])
            n = len(c.ops)
            text.draw(c, ylx, y0 + h / 2, text.spaced(pal, self.ylabel), size_y, label_rgb, pal.label_bold,
                      ha = 'center', va = 'baseline', rotation = 90.0, tracking = track)
            marks.tag(c, n, name = 'y name')
        if self.title_text:
            title_size = self._title_size()
            lift = rise - y0 - h + self._xtick_height() if above and self.xlabels and len(self.xticks) else 0.0
            pad = 8.0 * pal.spacing if self._title_pad is None else self._title_pad
            n = len(c.ops)
            text.draw(c, x0, y0 + h + lift + pad + 1.25 * title_size * _extra_lines(self.title_text),
                      self.title_text, title_size, pal.color(pal.title_color or pal.ink), pal.title_bold, ha = 'left')
            marks.tag(c, n, name = 'title')
        for a in self.artists:
            if a['kind'] == 'ridges' and a['labels'] is not None:
                n = len(a['curves'])
                for i, s in enumerate(self._ridge_labels):
                    t = text.spaced(pal, s)
                    tsize = 7.8 * pal.spacing if pal.caps else size + pal.spacing
                    text.draw(c, x0 - 8.0 * pal.spacing, float(self._ty(n - 1 - i)) + pal.spacing, t, tsize, ink,
                              pal.text_bold,
                              ha = 'right', tracking = track)
        lo, hi = sorted(self.ylim)
        # A label held to a share of the height shows while its x is within the limits, as an image so held does
        texts = [a for a in sorted((r for r in self.artists if r['kind'] == 'text'), key = lambda r: r['order'])
                 if math.isfinite(a['x'] + a['y'])
                 and not ((a['clip'] or a.get('yfrac')) and not (self._in_x(a['x']) and (a.get('yfrac')
                                                                                          or lo <= a['y'] <= hi)))]
        placed, groups, spans, spots = {}, [], [], self._place_glyphs()
        # Only labels whose spans across the page overlap can collide, on either side of their points, so each run of
        # those moves on its own; glyphs find their own places around their points
        for a in (a for a in texts if a['spread'] and id(a) not in spots):
            start = float(self._tx(a['x'])) + a['_dx']
            tw = self._note_size(a)[0]
            spans.append((start, start + tw, a) if a['_ha'] == 'left' else (start - tw, start, a))
        for start, end, a in sorted(spans, key = lambda v: v[0]):
            if groups and start < groups[-1][0]:
                groups[-1][0] = max(groups[-1][0], end)
                groups[-1][1].append(a)
            else:
                groups.append([end, [a]])
        for _, group in groups:
            want = np.array([float(self._ty(a['y'])) + a['dy'] for a in group])
            gap = max((1.25 + a['linespacing'] * _extra_lines(a['s'])) * (a['size'] or size) for a in group)
            order = np.argsort(want, kind = 'stable')
            # Closest to where they belong under a least gap: shifted down by their rank's gaps, the positions only have
            # to rise, which pooling adjacent violators solves exactly, and then the axes' height bounds all alike
            blocks = []
            for v in want[order] - gap * np.arange(len(group)):
                blocks.append([v, 1])
                while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                    total, count = blocks.pop()
                    blocks[-1][0] += total
                    blocks[-1][1] += count
            fit = np.concatenate([np.full(count, total / count) for total, count in blocks])
            floor = y0 + gap / 2
            fit = np.clip(fit, floor, max(floor, y0 + h - gap / 2 - gap * (len(group) - 1)))
            placed.update((id(group[i]), (y, gap)) for i, y in zip(order, fit + gap * np.arange(len(group))))
        # An annotation follows the point it sits on: where a highlight leaves that point as context, and names neither
        # the annotation nor another point there, it recedes with it, while notes placed away from the points stay
        focus = [f for f in [self._focus] + [r.get('focus') for r in self.artists] if f is not None]
        held, left = set(), set()
        for r in self.artists if focus else ():
            if r['kind'] == 'points':
                xs, ys = np.atleast_1d(r['x']).tolist(), np.atleast_1d(r['y']).tolist()
                for xy, kept in zip(zip(xs, ys), self._focused(r, len(xs)).tolist()):
                    (held if kept else left).add(xy)
        keys = set().union(*focus)
        back = lambda a: (not a.get('direct') and not a['frac'] and not a.get('yfrac') and a['s'] not in keys
                          and (a['x'], a['y']) in left
                          and (a['x'], a['y']) not in held)
        def fade(n):
            # A glyph's colors, its cut's included, recede all alike, as a heatmap's cells do, since SNFG's are fixed
            for op in c.ops[n:]:
                for paint in ('fill_rgb', 'stroke_rgb'):
                    if op.get(paint) is not None:
                        op[paint] = tuple(float(v) for v in pal.recede(np.array([[op[paint]]], dtype = float))[0, 0])
        for a in texts:
            bold = pal.text_bold if a['bold'] is None else a['bold']
            px, py = self._anchor(a)
            n = len(c.ops)
            if id(a) in spots:
                gx, gy, leader, pts = spots[id(a)]
                if leader:
                    marks.polyline(c, leader[::2], leader[1::2], a['rgb'], 0.5 * pal.spacing, 0.6, cap = 0)
                text.draw_glyph(c, self._note(a)[0], gx, gy, pts, (1.0, 1.0, 1.0) if pal.is_dark else
                                pal.color('paper') if a['halo'] > 0 else None)
                if focus and back(a):
                    fade(n)
                # In an interactive SVG, a drawing lights up what shares its label, such as its point, and says it
                marks.tag(c, n, hover = ((a['s'], ), (a['s'], )))
                continue
            if a.get('direct') in ('cloud', 'peak'):
                px, py = self._place_direct()[id(a)]
            ty, gap = placed.get(id(a), (py + a['dy'], 0.0))
            # Only a spread label that had to move gets a leader
            if id(a) in placed and abs(ty - py - a['dy']) > 0.3 * gap:
                way = 1.0 if a['_ha'] == 'left' else -1.0
                k = pal.spacing
                marks.polyline(c, [px + way * 2.0 * k, px + a['_dx'] - way * 1.5 * k], [py, ty], a['rgb'], 0.5 * k, 0.6,
                               cap = 0)
            g, pts, cut = self._note(a)
            if g is not None:
                left, bottom, right, top = g.extent(pts, cut)
                gx = px + a['_dx'] - {'left': left, 'center': (left + right) / 2, 'right': right}[a['_ha']]
                # Its root level with the point, or its edge at it
                gy = ty - {'top': top, 'bottom': bottom, 'baseline': bottom}.get(a['va'], 0.0)
                text.draw_glyph(c, g, gx, gy, pts, (1.0, 1.0, 1.0) if pal.is_dark else pal.color('paper') if
                                a['halo'] > 0 else None)
                if focus and back(a):
                    fade(n)
                # In an interactive SVG, a drawing lights up what shares its label, such as its point, and says it
                marks.tag(c, n, hover = ((a['s'], ), (a['s'], )))
                continue
            # A label that names a series takes its color, receded with it, and one on a receded point recedes too, as
            # dark or light as text must be to read
            rgb, receding = a['rgb'], focus and back(a)
            if (a.get('direct') and self._receding(a['order'] - 10 ** 9)) or receding:
                rgb = pal.recede(rgb)
            if (a.get('direct') or receding) and self.figure.legible:
                rgb = legible(rgb, pal.color('paper'), LC_TEXT)
            if receding:
                # Its leader, drawn before it, recedes with it
                for op in c.ops[n:]:
                    if op.get('stroke_rgb') is not None:
                        op['stroke_rgb'] = pal.recede(op['stroke_rgb'])
            n = len(c.ops)
            text.draw(c, px + a['_dx'], ty, a['s'], a['size'] or size, rgb, bold, ha = a['_ha'],
                      va = a['va'], rotation = a['rotation'], halo = a['halo'], halo_color = pal.color('paper'),
                      linespacing = a['linespacing'], tracking = a.get('tracking', 0.0))
            # In an interactive SVG, it lights up what it names, and needs no hover text of its own
            if a.get('direct'):
                marks.tag(c, n, series = self.legend_items[a['order'] - 10 ** 9][0], hover = ((), ()))
            elif not a['frac'] and not a.get('placed'):
                # An annotation lights up what shares its text, such as its point, and says nothing it does not show
                marks.tag(c, n, hover = ((a['s'], ), ()))
            if a['tag']:
                for op in c.ops[n:]:
                    op['label'] = (a['order'], id(self), a['s'])
        # Right of the axes, in order: tick labels, the axis name, the legend, and the color bar
        right = 0.0
        if self._yright:
            right = (yw + ypad if self.ylabels else off + ytl if len(self.yticks) else 0.0) + self._ylabel_room()
        self._draw_legend(c, right)
        if self._cbar:
            self._draw_colorbar(c, right)

    def _draw_colorbar(self, c, right):
        pal = self.pal
        k, bar = pal.spacing, COLORBAR_WIDTH * pal.spacing
        x0, y0, w, h = self.rect
        scale = self._cbar['source']._scale
        lo, hi = scale['lo'], scale['hi']
        # One value makes a scale of no length, whose only tick sits at its start
        span = hi - lo or 1.0
        stops = [tuple(v) for v in getattr(pal, scale['cmap'])(np.linspace(0.0, 1.0, 9))]
        ticks = [t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi and (t == int(t) or not self._cbar['integer'])]
        ink, label = pal.color(pal.ink), self._cbar['label']
        if self._cbar['loc'] == 'fill':
            bh = min(9.0 * k, h * 0.3)
            bw, by = w * 0.8, y0 + h * 0.34
            marks.polygon(c, [x0, x0 + bw, x0 + bw, x0], [by, by, by + bh, by + bh], None, 1.0,
                          ((x0, by, x0 + bw, by), stops))
            for t, s in zip(ticks, _minus(pal, tick_labels(ticks))):
                tx = x0 + (t - lo) / span * bw
                text.draw(c, tx, by - 3.0 * k, s, 7.4 * k, ink, ha = 'center', va = 'top', tabular = True)
            if label:
                text.draw(c, x0, by + bh + 4.0 * k, text.spaced(pal, label), (7.8 if pal.caps else 8.6) * k, ink,
                          ha = 'left')
            return
        right += self._legend_grid()[1][0] + 18.0 * k if self._legend_keep and self._legend_loc == 'right' else 0.0
        bx = x0 + w + right + 12.0 * k
        bh = min(h * 0.6, 170.0 * k)
        by = y0 + h - bh
        n = len(c.ops)
        marks.polygon(c, [bx, bx + bar, bx + bar, bx], [by, by, by + bh, by + bh], None, 1.0,
                      ((bx, by, bx, by + bh), stops))
        labels = _minus(pal, tick_labels(ticks))
        # Numbers line up on their decimal points
        split = self._decimals(labels, 8.4 * k, True)
        point = max(a for a, _ in split) if split else 0.0
        widest = point + max(b for _, b in split) if split else 0.0
        marks.tag(c, n, name = 'color bar')
        for i, (t, s) in enumerate(zip(ticks, labels)):
            n = len(c.ops)
            text.draw(c, bx + bar + 5.0 * k + (point - split[i][0] if split else 0.0), by + (t - lo) / span * bh, s,
                      8.4 * k, ink, va = 'center', tabular = True)
            marks.tag(c, n, name = 'color bar ' + s)
            widest = max(widest, text.width(s, 8.4 * k, tabular = True))
        if label:
            text.draw(c, bx + bar + 5.0 * k + widest + pal.label_size * 1.3, by + bh / 2,
                      text.spaced(pal, label), pal.label_size, pal.color(pal.label_color or pal.ink), pal.label_bold,
                      ha = 'center', va = 'baseline', rotation = 90.0)

    def _draw_legend(self, c, right):
        pal = self.pal
        if not self._legend_keep or self._legend_loc is None:
            return
        u, row = pal.spacing, LEGEND_ROW * pal.spacing
        x0, y0, w, h = self.rect
        size, track, entries = self._legend_entries()
        spots, (tw, th) = self._legend_grid()
        loc = self._legend_loc
        # Swatches in focus step up with their marks
        standing = self._standing()
        if loc == 'right':
            left, top = x0 + w + right + 16.0 * u, y0 + h - 8.0 * u
        elif loc == 'below':
            left = x0 if tw >= w else x0 + (w - tw) / 2
            top = (y0 - 40.0 * u if self._legend_y is None else self._legend_y) + (th - row)
        else:
            left = x0 + w - tw - 8.0 * u if 'right' in loc else x0 + 10.0 * u
            top = y0 + h - 12.0 * u if 'upper' in loc else y0 + 12.0 * u + th - row
        for (dx, dy), (s, rgb, k, marker, area, _, g, _), i in zip(spots, entries, self._legend_keep):
            x, y = left + dx, top - dy
            # In PowerPoint, an entry's swatch and name are named after it, so that they stay its own in the next figure
            n = len(c.ops)
            if k == 'header':
                text.draw(c, x, y, s, size, pal.color(pal.ink), True, va = 'center', tracking = track)
                marks.tag(c, n, name = 'legend ' + s)
                continue
            # An entry for what a highlight leaves as context recedes with it
            back = self._receding(i)
            rgb = pal.recede(rgb) if back else pal.stand(rgb) if standing else rgb
            if k == 'line':
                marks.polyline(c, [x, x + 12.0 * u], [y, y], rgb, max(pal.line_width, 1.2 * u), cap = 0, dash = marker)
            elif k == 'point':
                # As large as in the plot, up to what a legend row can hold
                marks.points(c, pal, [x + 6.0 * u], [y], min(area, 196.0 * u * u / pal.point_scale), rgb,
                             marker = marker)
            else:
                lo, hi = y - 4.5 * u, y + 4.5 * u
                marks.polygon(c, [x, x + 12.0 * u, x + 12.0 * u, x], [lo, lo, hi, hi], rgb)
            if g is not None:
                cut = self._legend_cut()
                gl, gb, _, gt = g.extent(size * text.GLYPH_SYMBOL, cut is not None or self._cut)
                text.draw_glyph(c, g, x + 17.0 * u - gl, y - (gb + gt) / 2, size * text.GLYPH_SYMBOL, cut)
            else:
                text.draw(c, x + 17.0 * u, y, s, size, pal.color(pal.muted if back else pal.ink), pal.text_bold,
                          va = 'center', tracking = track, halo = 0.0 if loc in ('right', 'below') else 2.6 * u,
                          halo_color = pal.color('paper'))
            # In an interactive SVG, it lights up what it names, and needs no hover text of its own
            marks.tag(c, n, name = 'legend ' + s, series = self.legend_items[i][0], hover = ((), ()))


def _recorded(method):
    """Notes a call on its figure for Figure.in_palette, as colors, widths, and the like are settled from the palette
    when a plot method runs. Only the outermost call: what it calls in turn would run again by itself in a replay."""
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        fig = self.figure
        fig._inside += 1
        try:
            result = method(self, *args, **kwargs)
        finally:
            fig._inside -= 1
        if not fig._inside:
            fig._calls.append((self, method.__name__, args, kwargs))
        return result
    return call


# All that only asks or converts is left out; a method that fails records nothing, as it changed nothing to replay
for _name, _method in list(vars(Axes).items()):
    if callable(_method) and _name[0] != '_' and _name not in ('get_xlim', 'get_ylim', 'page', 'data', 'boxes',
                                                               'hidden'):
        setattr(Axes, _name, _recorded(_method))
