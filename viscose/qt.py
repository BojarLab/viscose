"""Viscose figures in Qt widgets, with the mouse handling that plotting itself leaves to whoever shows a figure. Needs
the qt extra and is imported on its own, so that `import viscose` never loads Qt."""
from functools import lru_cache
from typing import NamedTuple
import numpy as np
from glycorender import pdfwrite
try:
    from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal
    from PySide6.QtGui import (QAction, QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                               QRadialGradient, QTransform)
    from PySide6.QtWidgets import QStyle, QToolBar, QVBoxLayout, QWidget
except ImportError as error:
    raise ImportError('viscose.qt needs PySide6, which comes with: pip install "viscose[qt]"') from error
from . import text
from .figure import Figure


# Widget pixels per inch of a figure, as figures follow their widget's size in inches
DPI = 100
RING_RADIUS, RING_WIDTH = 6.0, 1.4


def _path(ops):
    path = QPainterPath()
    for op in ops:
        if op[0] == 'm':
            path.moveTo(op[1], op[2])
        elif op[0] == 'l':
            path.lineTo(op[1], op[2])
        elif op[0] == 'c':
            path.cubicTo(*op[1:])
        else:
            path.closeSubpath()
    return path


@lru_cache(4096)
def _glyphs(ttf, string):
    """A string's outlines at 1000 units per em, cached, as the same few characters come back with every redraw; filled
    by the nonzero rule, as strokes of a glyph may overlap."""
    path = _path(text.outline(pdfwrite.Canvas(None, (1.0, 1.0)), ttf, string, 0.0, 0.0, 1000.0).ops)
    path.setFillRule(Qt.WindingFill)
    return path


@lru_cache(256)
def _bitmap(rgb, alpha, w, h):
    opaque = np.full((h, w), 255, np.uint8) if alpha is None else np.frombuffer(alpha, np.uint8).reshape(h, w)
    px = np.dstack([np.frombuffer(rgb, np.uint8).reshape(h, w, 3), opaque])
    return QImage(px.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def _color(rgb, alpha = 1.0):
    return QColor.fromRgbF(rgb[0], rgb[1], rgb[2], alpha)


def paint_page(figure, dpi, layout = True):
    """The page as a QImage, painted by Qt from the figure's display list: about ten times as fast as the numpy
    rasterizer that saved images go through, whatever the resolution, which makes zooming and panning smooth. For
    palettes without a soft shadow or spaced capitals, which the display list or this leaves out."""
    ops = figure.display_list(layout)
    scale = dpi / 72
    image = QImage(round(figure.width * scale), round(figure.height * scale), QImage.Format_RGBA8888_Premultiplied)
    image.fill(QColor(figure.pal.css('paper')))
    # Page points, from the lower left, to the image's pixels
    page = QTransform(scale, 0.0, 0.0, -scale, 0.0, image.height())
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    clip = None
    for op in ops:
        if op['clip'] is not clip:
            clip = op['clip']
            painter.setClipping(False)
            for k, (box, ctm) in enumerate(clip or ()):
                painter.setTransform(QTransform(*ctm) * page)
                painter.setClipRect(QRectF(*box), Qt.ReplaceClip if k == 0 else Qt.IntersectClip)
        painter.setTransform(QTransform(*op['ctm']) * page)
        if op['kind'] == 'text':
            size = op['size'] / 1000
            painter.save()
            painter.translate(op['x'], op['y'])
            painter.scale(size, size)
            painter.fillPath(_glyphs(op['ttf'], op['text']), _color(op['fill_rgb'], op['fill_alpha']))
            painter.restore()
        elif op['kind'] == 'image':
            # The unit square, first row at its top
            painter.setTransform(QTransform(1 / op['w'], 0.0, 0.0, -1 / op['h'], 0.0, 1.0) * QTransform(*op['ctm'])
                                 * page)
            painter.setOpacity(op['fill_alpha'])
            painter.drawImage(0, 0, _bitmap(op['rgb'], op['alpha'], op['w'], op['h']))
            painter.setOpacity(1.0)
        else:
            path = _path(op['path'])
            path.setFillRule(Qt.OddEvenFill if op['even_odd'] else Qt.WindingFill)
            if op['fill']:
                grad = op.get('fill_grad')
                brush = QBrush(_color(op['fill_rgb'], op['fill_alpha']))
                if grad:
                    fill = (QLinearGradient(*grad[1]) if grad[0] == 'linear'
                            else QRadialGradient(grad[1][0], grad[1][1], grad[1][2]))
                    for offset, stop in grad[2]:
                        fill.setColorAt(offset, _color(stop[:3], stop[3] * op['fill_alpha']))
                    brush = QBrush(fill)
                painter.fillPath(path, brush)
            if op['stroke']:
                pen = QPen(_color(op['stroke_rgb'], op['stroke_alpha']), op['line_width'])
                pen.setCapStyle((Qt.FlatCap, Qt.RoundCap, Qt.SquareCap)[op['cap']])
                pen.setJoinStyle((Qt.MiterJoin, Qt.RoundJoin, Qt.BevelJoin)[op['join']])
                if op['dash']:
                    # Qt counts a dash in line widths
                    pen.setDashPattern([v / op['line_width'] for v in op['dash'][0]])
                    pen.setDashOffset(op['dash'][1] / op['line_width'])
                painter.strokePath(path, pen)
    painter.end()
    return image


class ChartEvent(NamedTuple):
    """The mouse in page points, from the page's lower left."""
    x: float
    y: float
    button: int = 1
    step: float = 0.0


class Target:
    """Points in data coordinates that answer the mouse, or with stick the sticks from zero up to them; contains gives
    the nearest first."""

    def __init__(self, ax, x, y, radius = 5.0, stick = False):
        self.ax, self.radius, self.stick = ax, radius, stick
        self.x, self.y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)

    def contains(self, event):
        x0, y0, w, h = self.ax.rect
        r = self.radius
        px, py = self.ax.page(self.x, self.y)
        inside = (px >= x0 - r) & (px <= x0 + w + r) & (py >= y0 - r) & (py <= y0 + h + r)
        if self.stick:
            near = (np.abs(px - event.x) <= r) & (event.y >= self.ax.page(0.0, 0.0)[1] - r) & (event.y <= py + r)
        else:
            near = np.hypot(px - event.x, py - event.y) <= r
        found = np.flatnonzero(inside & near)
        order = np.argsort(np.hypot(px[found] - event.x, py[found] - event.y), kind = 'stable')
        return bool(len(found)), {'ind': found[order].tolist()}


class Canvas(QWidget):
    """Shows a viscose figure at the widget's size. Its page is kept as an image, so the mouse, a selection ring (rings:
    (axes, x, y)), and a dragged range cost no new drawing, and draw_idle draws it again at the next paint. With span,
    a function, dragging across the first axes calls it with the range in data coordinates."""
    moved = Signal(object)
    pressed = Signal(object)
    scrolled = Signal(object)
    resized = Signal()
    left = Signal()

    def __init__(self, figure):
        super().__init__()
        self.figure, self.toolbar, self.span, self.rings = figure, None, None, []
        self.image, self.band, self.start, self.pan_from = None, None, None, None
        self.setMouseTracking(True)
        figure.layout()

    def sizeHint(self):
        return QSize(round(self.figure.width * DPI / 72), round(self.figure.height * DPI / 72))

    def minimumSizeHint(self):
        return QSize(10, 10)

    def draw_idle(self):
        self.figure.layout()
        self.image = None
        self.update()

    def event_at(self, event, step = 0.0):
        position = event.position()
        left = event.buttons() & Qt.LeftButton or event.button() == Qt.LeftButton
        return ChartEvent(position.x() * 72 / DPI, self.figure.height - position.y() * 72 / DPI, 1 if left else 3, step)

    def to_widget(self, x, y):
        return x * DPI / 72, (self.figure.height - y) * DPI / 72

    def paintEvent(self, event):
        ratio = self.devicePixelRatioF()
        if self.image is None or self.image.devicePixelRatio() != ratio:
            # Laid out by draw_idle or, for a figure just shown, by __init__
            self.image = paint_page(self.figure, DPI * ratio, False)
            self.image.setDevicePixelRatio(ratio)
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(self.figure.pal.css('ink')), RING_WIDTH * DPI / 72))
        for ax, x, y in self.rings:
            cx, cy = self.to_widget(*ax.page(x, y))
            r = RING_RADIUS * DPI / 72
            painter.drawEllipse(round(cx - r), round(cy - r), round(2 * r), round(2 * r))
        if self.band is not None:
            color = QColor(self.figure.pal.css('a1'))
            color.setAlpha(50)
            painter.fillRect(self.band, color)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width, height = max(self.width(), 10) / DPI, max(self.height(), 10) / DPI
        if (self.figure.width, self.figure.height) != (width * 72.0, height * 72.0):
            self.figure.set_size(width, height)
            self.draw_idle()
            self.resized.emit()

    def mousePressEvent(self, event):
        at = self.event_at(event)
        mode = self.toolbar.mode if self.toolbar is not None else ''
        if at.button == 1 and (mode or self.span is not None) and self.figure.axes:
            ax = self.figure.axes[0]
            x0, y0, w, h = ax.rect
            if not (x0 <= at.x <= x0 + w and y0 <= at.y <= y0 + h):
                return
            self.start = event.position().toPoint()
            if mode == 'pan':
                self.toolbar.begin()
                self.pan_from = at, ax.xlim, ax.ylim
            return
        if not mode:
            self.pressed.emit(at)

    def mouseDoubleClickEvent(self, event):
        # Qt's default sends a second press, which undid whatever the first one toggled
        event.accept()

    def leaveEvent(self, event):
        self.left.emit()

    def mouseMoveEvent(self, event):
        if self.start is None:
            self.moved.emit(self.event_at(event))
            return
        position, mode = event.position().toPoint(), self.toolbar.mode if self.toolbar is not None else ''
        if mode == 'pan':
            at, xlim, ylim = self.pan_from
            now = self.event_at(event)
            ax = self.figure.axes[0]
            x0, y0, w, h = ax.rect
            dx, dy = (now.x - at.x) / w * (xlim[1] - xlim[0]), (now.y - at.y) / h * (ylim[1] - ylim[0])
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.draw_idle()
            return
        x0, y0, w, h = self.figure.axes[0].rect
        left, top = self.to_widget(x0, y0 + h)
        bottom = self.to_widget(x0, y0)[1]
        right = self.to_widget(x0 + w, y0)[0]
        lo, hi = sorted((min(max(self.start.x(), left), right), min(max(position.x(), left), right)))
        if mode == 'zoom':
            up, down = sorted((min(max(self.start.y(), top), bottom), min(max(position.y(), top), bottom)))
        else:
            up, down = top, bottom
        self.band = QRect(round(lo), round(up), round(hi - lo), round(down - up))
        self.update()

    def mouseReleaseEvent(self, event):
        band, start, self.band, self.start = self.band, self.start, None, None
        mode = self.toolbar.mode if self.toolbar is not None else ''
        if start is None:
            return
        if mode == 'pan':
            self.toolbar.push_current()
        elif band is not None and band.width() >= 3:
            ax = self.figure.axes[0]
            x_lo, y_lo = ax.data(band.x() * 72 / DPI, self.figure.height - (band.y() + band.height()) * 72 / DPI)
            x_hi, y_hi = ax.data((band.x() + band.width()) * 72 / DPI, self.figure.height - band.y() * 72 / DPI)
            if mode == 'zoom' and band.height() >= 3:
                self.toolbar.begin()
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                self.toolbar.push_current()
                self.draw_idle()
            elif not mode:
                self.span(x_lo, x_hi)
        self.update()

    def wheelEvent(self, event):
        self.scrolled.emit(self.event_at(event, event.angleDelta().y() / 120))


class Toolbar(QToolBar):
    """Home, back, and forward through the views of a canvas's first axes, and dragging to pan it or to zoom into a
    rectangle."""

    def __init__(self, canvas, parent = None):
        super().__init__(parent)
        self.canvas, self.mode, self.views, self.position = canvas, '', [], -1
        canvas.toolbar = self
        self.setIconSize(QSize(16, 16))
        style = self.style()
        self.addAction('Home', self.home).setToolTip('Show the whole plot')
        for icon, label, tip, action in ((QStyle.SP_ArrowBack, 'Back', 'The view before this one',
                                          lambda: self.move_by(-1)),
                                         (QStyle.SP_ArrowForward, 'Forward', 'The view after this one',
                                          lambda: self.move_by(1))):
            self.addAction(style.standardIcon(icon), label, action).setToolTip(tip)
        self.pan = QAction('Pan', self, checkable = True, toolTip = 'Drag to move the plot')
        self.zoom = QAction('Zoom', self, checkable = True, toolTip = 'Drag a rectangle to zoom into it')
        self.pan.toggled.connect(lambda on: self.set_mode('pan', on))
        self.zoom.toggled.connect(lambda on: self.set_mode('zoom', on))
        self.addAction(self.pan)
        self.addAction(self.zoom)

    def set_mode(self, mode, on):
        if on:
            self.mode = mode
            (self.zoom if mode == 'pan' else self.pan).setChecked(False)
            self.canvas.setCursor(Qt.SizeAllCursor if mode == 'pan' else Qt.CrossCursor)
        elif self.mode == mode:
            self.mode = ''
            self.canvas.setCursor(Qt.ArrowCursor)

    def begin(self):
        """The view before the first zoom is the one Home returns to."""
        if not self.views:
            self.push_current()

    def push_current(self):
        ax = self.canvas.figure.axes[0]
        self.views[self.position + 1:] = [(ax.get_xlim(), ax.get_ylim())]
        self.position = len(self.views) - 1

    def apply(self, position):
        self.position = position
        ax = self.canvas.figure.axes[0]
        ax.set_xlim(*self.views[position][0])
        ax.set_ylim(*self.views[position][1])
        self.canvas.draw_idle()

    def home(self):
        if self.views:
            self.apply(0)

    def move_by(self, step):
        if 0 <= self.position + step < len(self.views):
            self.apply(self.position + step)


class ChartView(QWidget):
    """A canvas with a new figure for each thing it shows, of the widget's size, in a palette or the current default."""

    def __init__(self, height = 3.0, palette = None):
        super().__init__()
        self.palette = palette
        self.canvas = Canvas(Figure(6, height, palette))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    @property
    def figure(self):
        return self.canvas.figure

    def new_figure(self):
        figure = Figure(self.figure.width / 72, self.figure.height / 72, self.palette)
        self.canvas.figure, self.canvas.span, self.canvas.rings = figure, None, []
        self.canvas.draw_idle()
        return figure
