import dataclasses
import os
import re
import subprocess
import sys
import warnings
from io import BytesIO
from unittest.mock import patch
import numpy as np
import pytest
import viscose as vc
from viscose import marks
from viscose.axes import fmt, nice_ticks, tick_labels
from viscose.figure import SHADOW_DPI
from viscose.palettes import get_palette, ramp
from viscose.text import runs, width
from glycorender import pdfwrite, raster
from glycorender.pdfwrite import pdfmetrics


NAMES = ['lumen', 'couture', 'poster', 'aura']


def pixels(fig, dpi = 72):
    """The figure as an (h, w, 3) array, with the page white."""
    op = fig._shadow_op(dpi)
    fig_ops = fig._canvas(stamp_dpi = dpi)
    if op is not None:
        fig_ops.ops.insert(0, op)
    return raster.render(fig_ops.ops, fig.width, fig.height, dpi / 72.0, dpi / 72.0, (1.0, 1.0, 1.0))


def all_marks(palette):
    rng = np.random.default_rng(0)
    fig, axes = vc.subplots(2, 3, figsize = (10, 7), palette = palette)
    fig.suptitle('Title', 'Subtitle')
    x = np.linspace(0, 10, 50)
    axes[0][0].plot(x, np.sin(x), label = 'sine', marker = True)
    axes[0][0].fill_between(x, np.sin(x) - .2, np.sin(x) + .2)
    axes[0][0].legend('upper right')
    axes[0][1].scatter(rng.normal(size = 200), rng.normal(size = 200), label = 'cloud', s = 20)
    axes[0][1].scatter(rng.normal(size = 50), rng.normal(size = 50), c = rng.normal(size = 50), s = 30,
                       cmap = 'diverging')
    axes[0][1].legend('below')
    axes[0][2].bar(['a', 'b', 'c'], [1, 2, 3], label = 'bars')
    axes[0][2].errorbar([0, 1, 2], [1, 2, 3], .3)
    axes[0][2].legend('right')
    axes[1][0].hist(rng.normal(size = 500), bins = 15)
    axes[1][0].set_xlabel('value')
    axes[1][0].set_ylabel('count')
    axes[1][1].heatmap(rng.normal(size = (5, 4)), list('wxyz'), list('abcde'))
    axes[1][2].barh(['p', 'q'], [2, 3])
    axes[1][2].axvline(1)
    axes[1][2].axhline(.5)
    axes[1][2].set_title('Panel')
    axes[1][2].text(1, 0, 'note')
    axes[1][2].annotate('point', (2, 1), offset = (3, 3))
    return fig


@pytest.mark.parametrize('name', NAMES)
def test_every_format_for_every_palette(name, tmp_path):
    fig = all_marks(name)
    assert fig.save(tmp_path / 'a.png', dpi = 72).read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    assert fig.save(tmp_path / 'a.pdf').read_bytes()[:5] == b'%PDF-'
    assert '<svg' in fig.save(tmp_path / 'a.svg').read_text(encoding = 'utf-8')


def test_unknown_format_and_palette_raise(tmp_path):
    with pytest.raises(ValueError, match = 'Unsupported'):
        all_marks('lumen').save(tmp_path / 'a.jpg')
    with pytest.raises(ValueError, match = 'Unknown palette'):
        vc.get_palette('nope')


def test_shadow_is_an_image_only_in_palettes_that_have_one():
    def images(name):
        return all_marks(name).to_pdf().count(b'/Subtype /Image')
    assert images('couture') == 0 and images('poster') == 0
    assert images('lumen') >= 1 and images('aura') >= 1
    assert '<image' in all_marks('aura').to_svg() and '<image' not in all_marks('poster').to_svg()


def test_lumen_bars_are_real_gradients_in_the_pdf():
    fig, ax = vc.subplots(palette = 'lumen')
    ax.bar(['a', 'b'], [1, 2])
    assert b'/ShadingType 2' in fig.to_pdf()
    fig, ax = vc.subplots(palette = 'poster')
    ax.bar(['a', 'b'], [1, 2])
    assert b'/ShadingType' not in fig.to_pdf()


def test_marks_land_where_the_data_says():
    fig, ax = vc.subplots(palette = 'poster', figsize = (4, 4))
    ax.bar([0.0], [1.0], width = 0.5, color = '#FF0000')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    img = pixels(fig, 144)
    x0, y0, w, h = ax.rect
    sc = 2.0
    def at(px, py):
        return img[int(img.shape[0] - py * sc), int(px * sc)]
    # Inside the bar: x = 0, y = 0.5 in data
    assert at(x0 + w / 2, y0 + h / 4).tolist() == [255, 0, 0]
    # Right of the bar, and above it
    assert at(x0 + w * 0.9, y0 + h / 4).tolist() == [255, 255, 255]
    assert at(x0 + w / 2, y0 + h * 0.75).tolist() == [255, 255, 255]


def test_heatmap_first_row_is_at_the_top():
    fig, ax = vc.subplots(palette = 'poster', figsize = (4, 4))
    ax.heatmap(np.array([[2.0, 2.0], [-2.0, -2.0]]), vmin = -2, vmax = 2)
    img = pixels(fig, 144)
    x0, y0, w, h = ax.rect
    # At the middle of a cell, as the palette leaves white gaps between cells
    top = img[int(img.shape[0] - (y0 + h * .75) * 2), int((x0 + w / 4) * 2)]
    low = img[int(img.shape[0] - (y0 + h * .25) * 2), int((x0 + w / 4) * 2)]
    assert top.tolist() == [int(round(v * 255)) for v in fig.pal.color('a0')]
    assert low.tolist() == [int(round(v * 255)) for v in fig.pal.color('b0')]


def test_labels_leave_room_inside_the_page():
    fig, ax = vc.subplots(palette = 'poster', figsize = (5, 4))
    ax.bar(['a long category', 'another long one'], [1000000, 2000000])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig._canvas()
    x0, y0, w, h = ax.rect
    assert x0 > 0 and y0 > 0 and x0 + w <= fig.width and y0 + h <= fig.height
    assert w > 100 and h > 100


def test_clustermap_dendrograms_line_up_with_the_matrix():
    m = np.random.default_rng(1).normal(size = (6, 4))
    fig, heat = vc.clustermap(m, list('abcd'), list('uvwxyz'), palette = 'couture', title = 'T')
    fig._layout()
    top, left = fig.axes[1], fig.axes[2]
    assert top.rect[0] == heat.rect[0] and top.rect[2] == heat.rect[2]
    assert left.rect[1] == heat.rect[1] and left.rect[3] == heat.rect[3]
    assert fig.to_png(72)[:4] == b'\x89PNG'


def test_ridgeline_and_stems_and_stack_in_all_palettes():
    x = np.linspace(0, 1, 50)
    for name in NAMES:
        fig, (a, b, s) = vc.subplots(1, 3, figsize = (9, 3), palette = name)
        a.ridgeline(x, [np.exp(-((x - m) / .1) ** 2) for m in (.2, .5, .8)], ['p', 'q', 'r'])
        b.stem([1, 2, 3], [5, -3, 2], cap = True)
        s.stacked_bar(['u', 'v'], [[1, 2, 3], [3, 2, 1]], ['x', 'y', 'z'], colors = fig.pal.groups(2, 1))
        s.legend()
        assert fig.to_png(72)[:4] == b'\x89PNG'


def test_palette_colors_are_few_and_grouped():
    pal = vc.get_palette('aura')
    assert pal.groups(3, 2)[:3] == [pal.color('a0'), pal.color('a1'), pal.color('a2')]
    assert pal.groups(3, 2)[3:] == [pal.color('b1'), pal.color('b2')]
    assert pal.groups(1, 1) == [pal.color('a1'), pal.color('b1')]
    assert len(pal.colors(8)) == 8
    # Past eight categories: still distinct, and still shades of the two hues
    assert len(pal.colors(12)) == 12 and len({tuple(np.round(c, 3)) for c in pal.colors(16)}) == 16
    with pytest.raises(ValueError):
        pal.groups(1, 1, 1)
    with pytest.raises(ValueError):
        pal.groups(5)
    assert pal.color((0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3) and pal.color('none') is None
    assert pal.color('muted') == pal.color(pal.muted)
    assert pal.shades('a1')[1] == pal.color('a1') and len(pal.shades('#123456')) == 3


def test_scales_run_as_documented():
    pal = vc.get_palette('poster')
    assert np.allclose(pal.diverging(0.5), (1, 1, 1))
    assert np.allclose(pal.diverging(1.0), pal.color('a0')) and np.allclose(pal.diverging(0.0), pal.color('b0'))
    assert np.allclose(pal.sequential(1.0), pal.color('a0')) and pal.sequential(np.zeros((2, 3))).shape == (2, 3, 3)
    assert np.allclose(ramp(['#000000', '#FFFFFF'], 0.5), 0.5)


def test_set_and_register_palette():
    custom = vc.Palette('mine', ['#000000', '#111111', '#222222', '#333333'],
                        ['#444444', '#555555', '#666666', '#777777'], '#000000', '#111111', '#222222')
    vc.register_palette(custom)
    assert vc.get_palette('mine') is custom and vc.get_palette(custom) is custom
    vc.set_palette('mine')
    try:
        assert vc.subplots()[0].pal is custom
    finally:
        vc.set_palette('lumen')
        del vc.palettes['mine']


def _seared():
    return vc.Palette('seared', ['#4A1D0A', '#A34E12', '#E2972E', '#F9E2AE'],
                      ['#1B2F3B', '#3C6478', '#7DA8BA', '#D3E4EA'], '#2E1508', '#8C6B55', '#F6EEE2', point = 'crust',
                      bar = 'sear', ridge = 'sear', crust = 1.75)


def _pixel(fig, ax, x, y, dpi = 200):
    fig.layout()
    rgba = fig.to_rgba(dpi)
    px, py = ax.page(x, y)
    return rgba[rgba.shape[0] - 1 - int(py * dpi / 72), int(px * dpi / 72), :3].astype(int)


def test_seared_bars_brown_with_height_on_one_scale():
    fig, axes = vc.subplots(palette = _seared())
    axes.bar([0, 1], [1.0, 3.0], color = 'a1')
    axes.set_ylim(0, 4)
    # Equal heights have equal colors whatever the bar's own height, so the tall bar's tip is the browner one
    assert np.abs(_pixel(fig, axes, 0, 0.5) - _pixel(fig, axes, 1, 0.5)).max() <= 2
    assert _pixel(fig, axes, 1, 2.9).sum() < _pixel(fig, axes, 0, 0.95).sum()


def test_seared_stems_and_bars_brown_away_from_zero_on_both_sides():
    fig, axes = vc.subplots(palette = _seared())
    axes.stem([1.0, 2.0], [3.0, -3.0], color = 'a1', lw = 12.0)
    axes.bar([3], [-3.0], color = 'a1')
    axes.set_ylim(-4, 4)
    for x, tip, near in ((1.0, 2.9, 0.2), (2.0, -2.9, -0.2), (3, -2.9, -0.2)):
        assert _pixel(fig, axes, x, tip).sum() < _pixel(fig, axes, x, near).sum()


def test_a_crust_rims_points_and_lies_under_lines():
    pal = _seared()
    fig, axes = vc.subplots(palette = pal)
    axes.scatter([1.0], [1.0], s = 900, color = 'a2')
    axes.set_xlim(0, 2)
    axes.set_ylim(0, 2)
    fig.layout()
    middle = _pixel(fig, axes, 1.0, 1.0)
    assert np.abs(middle - np.array(pal.color('a2')) * 255).max() <= 2
    rgba, (px, py) = fig.to_rgba(200), axes.page(1.0, 1.0)
    # 900 points squared is a radius of 15 points, and the rim is the outer 1.3 of them
    rim = rgba[rgba.shape[0] - 1 - int(py * 200 / 72), int((px + 14.4) * 200 / 72), :3].astype(int)
    assert np.abs(rim - np.array(pal.color('a1')) * 255).max() <= 2
    fig, axes = vc.subplots(palette = pal)
    axes.plot([0, 2], [1, 1], color = 'a2', lw = 8.0)
    axes.set_ylim(0, 2)
    fig.layout()
    rgba, (px, py) = fig.to_rgba(200), axes.page(1.0, 1.0)
    row, col = rgba.shape[0] - 1 - int(py * 200 / 72), int(px * 200 / 72)
    assert np.abs(rgba[row, col, :3].astype(int) - np.array(pal.color('a2')) * 255).max() <= 2
    assert np.abs(rgba[row - int(5.5 * 200 / 72), col, :3].astype(int) - np.array(pal.color('a1')) * 255).max() <= 2


def test_every_plot_draws_seared_in_every_format(tmp_path):
    pal = vc.register_palette(_seared())
    rng = np.random.default_rng(3)
    try:
        fig, axes = vc.subplots(2, 3, palette = 'seared')
        axes[0, 0].bar(['a', 'b'], [1.0, 2.0])
        axes[0, 1].barh(['a', 'b'], [1.0, 2.0])
        axes[0, 2].boxplot([rng.normal(size = 20), rng.normal(size = 20)])
        axes[1, 0].violinplot([rng.normal(size = 20), rng.normal(size = 20)])
        axes[1, 1].stacked_bar(['a', 'b'], [[1.0, 2.0], [2.0, 1.0]], labels = ['x', 'y'])
        axes[1, 2].ridgeline(np.linspace(0, 1, 30), [np.sin(np.linspace(0, 3, 30))] * 3)
        for suffix in ('png', 'pdf', 'svg'):
            fig.save(tmp_path / f'seared.{suffix}')
            assert (tmp_path / f'seared.{suffix}').stat().st_size > 1000
        dendro = vc.clustermap(rng.normal(size = (6, 5)), palette = 'seared')[0]
        assert dendro.to_png(72)[:4] == b'\x89PNG'
        with pytest.raises(ValueError):
            vc.Palette('x', pal.a, pal.b, '#000000', '#111111', '#222222', bar = 'seared')
    finally:
        del vc.palettes['seared']


@pytest.fixture
def qt():
    # Never a window: offscreen, and no widget is shown
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWidgets')
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from viscose import qt
    yield qt
    app.processEvents()


def test_importing_viscose_does_not_load_qt():
    code = ("import sys, viscose; assert not any(m.startswith('PySide6') for m in sys.modules); "
            "assert 'viscose.qt' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check = True)


def test_qt_paints_what_the_rasterizer_draws(qt):
    from PySide6.QtGui import QColor
    fig, axes = vc.subplots(palette = _seared())
    axes.bar([0, 1], [1.0, 3.0], color = 'a1')
    axes.plot([0, 1], [3.5, 3.5], color = 'b1', lw = 6.0)
    axes.set_ylim(0, 4)
    fig.layout()
    image = qt.paint_page(fig, 200, False)
    rgba = fig.to_rgba(200)
    assert (image.height(), image.width()) == rgba.shape[:2]
    # The middles of the bars at several heights, of the line, and of the empty plot
    for x, y in ((0, 0.5), (1, 0.5), (1, 2.9), (0.5, 3.5), (0.5, 2.0)):
        px, py = axes.page(x, y)
        col, row = int(px * 200 / 72), rgba.shape[0] - 1 - int(py * 200 / 72)
        color = QColor(image.pixel(col, row))
        assert np.abs(np.array([color.red(), color.green(), color.blue()]) - rgba[row, col, :3].astype(int)).max() <= 3


def test_qt_paints_every_kind_of_mark_as_the_rasterizer_does(qt):
    from PySide6.QtGui import QImage
    # Lines, dashes, sticks, bands, transparent points with edges, markers, text with subscripts, a gradient color bar,
    # a legend, and an image with alpha
    figure = vc.Figure(6, 4, _seared())
    ax = figure.add_axes([0.12, 0.15, 0.68, 0.75])
    rng = np.random.default_rng(0)
    ax.plot([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 4.0], color = 'a1', lw = 1.5, label = 'line')
    ax.plot([0.0, 3.0], [0.5, 0.5], color = 'b1', dash = (3.0, 1.3), label = 'dashed')
    ax.vlines([0.5, 1.5, 2.5], 0.0, [1.0, 2.0, 3.0], color = ['a0', 'a2', 'b1'], lw = 1.2)
    ax.axvspan(1.0, 1.4, color = 'a2')
    ax.scatter(rng.uniform(0, 3, 40), rng.uniform(0, 4, 40), s = rng.uniform(10, 80, 40), c = rng.uniform(0, 1, 40),
               edge = ('white', 0.5), alpha = 0.9)
    ax.scatter([0.3, 0.6], [3.5, 3.5], s = 40, color = 'b1', marker = 'v', edge = ('white', 0.6))
    ax.annotate('^{1,3}A_{2α}', (2.0, 1.0), size = 9, rotation = 90.0, halo = 0.0)
    ax.text(0.5, 1.02, 'A title', size = 11, ha = 'center', va = 'baseline', axes_fraction = True)
    square = np.zeros((20, 20, 4), np.uint8)
    square[..., 2], square[5:15, 5:15, 3] = 200, 255
    ax.image(square, 2.5, 1.1, 20 * 72 / 200, 20 * 72 / 200, y_fraction = True)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    ax.set_xlabel('m/z')
    ax.legend('upper right')
    ax.colorbar('Scale')
    painted = qt.paint_page(figure, 200).convertToFormat(QImage.Format_RGBA8888)
    rows = np.frombuffer(painted.constBits(), np.uint8).reshape(painted.height(), painted.bytesPerLine())
    pixels = rows[:, :4 * painted.width()].reshape(painted.height(), painted.width(), 4)[..., :3].astype(float)
    raster = figure.to_rgba(200)[..., :3].astype(float)
    assert pixels.shape == raster.shape
    # Nearly the same everywhere, once a pixel of antialiasing is forgiven: a shifted, mirrored, or missing mark would
    # stand out. This palette's darkest shade makes its edges the costliest, about 46 levels at most
    assert np.abs(pixels - raster).mean() < 1.5
    blurred = lambda image: sum(np.roll(np.roll(image, dx, 0), dy, 1) for dx in (-2, -1, 0, 1, 2)
                                for dy in (-2, -1, 0, 1, 2)) / 25
    assert np.abs(blurred(pixels) - blurred(raster)).max() < 50


def test_qt_paints_a_dark_page_in_its_own_color(qt):
    from PySide6.QtGui import QColor
    pal = _seared().dark()
    fig, axes = vc.subplots(palette = pal)
    axes.plot([0, 1], [0, 1])
    assert QColor(qt.paint_page(fig, 72).pixel(2, 2)).name().upper() == pal.css('paper')


def test_a_target_gives_the_nearest_point_first(qt):
    fig, axes = vc.subplots()
    axes.scatter([1.0, 1.1, 5.0], [1.0, 1.0, 1.0])
    axes.set_xlim(0, 6)
    axes.set_ylim(0, 2)
    fig.layout()
    x, y = axes.page(1.08, 1.0)
    found, details = qt.Target(axes, [1.0, 1.1, 5.0], [1.0, 1.0, 1.0], radius = 60.0).contains(qt.ChartEvent(x, y))
    assert found and details['ind'] == [1, 0]
    assert not qt.Target(axes, [5.0], [1.0]).contains(qt.ChartEvent(x, y))[0]
    # A stick answers along its length, from zero up to its point
    x, y = axes.page(5.0, 0.5)
    assert qt.Target(axes, [5.0], [1.0], stick = True).contains(qt.ChartEvent(x, y))[0]
    assert not qt.Target(axes, [5.0], [1.0]).contains(qt.ChartEvent(x, y))[0]


def test_a_canvas_answers_the_mouse_and_its_toolbar_keeps_views(qt):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    view = qt.ChartView(palette = 'poster')
    fig, axes = vc.subplots(palette = 'poster')
    axes.plot([0, 10], [0, 10])
    canvas = qt.Canvas(fig)
    canvas.resize(600, 400)
    canvas.grab()
    fig.layout()
    pressed, spans = [], []
    canvas.pressed.connect(pressed.append)
    x, y = canvas.to_widget(*axes.page(5.0, 5.0))
    QTest.mouseClick(canvas, Qt.LeftButton, pos = QPoint(round(x), round(y)))
    assert len(pressed) == 1 and pressed[0].button == 1
    assert np.allclose(axes.data(pressed[0].x, pressed[0].y), (5.0, 5.0), atol = 0.1)
    # With a span, a drag across the axes gives its range in data coordinates instead of a press
    canvas.span = lambda lo, hi: spans.append((lo, hi))
    x2, _ = canvas.to_widget(*axes.page(8.0, 5.0))
    QTest.mousePress(canvas, Qt.LeftButton, pos = QPoint(round(x), round(y)))
    QTest.mouseMove(canvas, QPoint(round(x2), round(y)))
    QTest.mouseRelease(canvas, Qt.LeftButton, pos = QPoint(round(x2), round(y)))
    assert len(pressed) == 1 and np.allclose(spans[0], (5.0, 8.0), atol = 0.1)
    bar = qt.Toolbar(canvas)
    home = axes.get_xlim()
    bar.zoom.setChecked(True)
    assert bar.mode == 'zoom'
    bar.begin()
    axes.set_xlim(2, 4)
    bar.push_current()
    bar.move_by(-1)
    assert axes.get_xlim() == home
    bar.move_by(1)
    assert axes.get_xlim() == (2, 4)
    bar.home()
    assert axes.get_xlim() == home
    bar.pan.setChecked(True)
    assert bar.mode == 'pan' and not bar.zoom.isChecked()
    # A chart view's new figures keep its palette
    assert view.figure.pal.name == 'poster' and view.new_figure().pal.name == 'poster'
    assert view.canvas.figure is view.figure


def test_ticks_and_number_format():
    assert nice_ticks(0, 14.2) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert nice_ticks(0, 1.08) == [0, .2, .4, .6, .8, 1.0]
    assert nice_ticks(0, 100) == [0, 20, 40, 60, 80, 100]
    assert nice_ticks(5, 5) == [5]
    assert fmt(0.0) == '0' and fmt(2.5) == '2.5' and fmt(1e6) == '10^{6}' and fmt(2e-5) == '2×10^{-5}'


def test_log_axes_and_categories():
    fig, ax = vc.subplots(palette = 'lumen')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter([1, 10, 100], [1, 100, 10000], s = 10)
    ax._limits()
    assert ax.yticks == [1.0, 10.0, 100.0, 1000.0, 10000.0] and fig.to_png(72)[:4] == b'\x89PNG'
    fig, ax = vc.subplots()
    ax.bar(['b', 'a', 'b'], [1, 2, 3])
    ax._limits()
    assert ax.xlabels == ['b', 'a'] and ax.xticks == [0, 1]


def test_nans_are_skipped():
    fig, ax = vc.subplots(palette = 'aura')
    ax.plot([0, 1, np.nan, 3], [1, 2, 3, 4])
    ax.scatter([0, np.nan], [1, 2])
    h = np.array([[1.0, np.nan], [2.0, 3.0]])
    fig2, ax2 = vc.subplots(palette = 'poster')
    ax2.heatmap(h, cmap = 'sequential')
    assert fig.to_png(72)[:4] == b'\x89PNG' and fig2.to_png(72)[:4] == b'\x89PNG'


def test_vlines_span_and_image_land_where_the_data_says():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.vlines([1.0, 3.0], 0.0, [2.0, 4.0], color = ['#FF0000', '#0000FF'], lw = 2.0)
    ax.axvspan(1.5, 2.5, color = '#00FF00', alpha = 1.0)
    # A black square of 20 points, one row of it white, so that its orientation shows
    square = np.zeros((10, 10, 4), np.uint8)
    square[..., 3] = 255
    square[0] = 255
    ax.image(square, 3.5, 2.0, 20.0, 20.0)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    fig.layout()
    px = fig.to_rgba(72.0)
    assert px.shape == (216, 288, 4) and (px[..., 3] == 255).all()
    at = lambda x, y: tuple(px[int(216 - ax.page(x, y)[1]), int(ax.page(x, y)[0]), :3])
    assert at(1.0, 1.0)[0] > 200 and at(1.0, 1.0)[2] < 50 and at(3.0, 3.0)[2] > 200 and at(1.0, 3.0) == (255, 255, 255)
    assert at(2.0, 2.0)[1] > 200 and at(2.0, 2.0)[0] < 50
    cx, cy = ax.page(3.5, 2.0)
    assert tuple(px[int(216 - cy), int(cx), :3]) == (0, 0, 0)
    assert tuple(px[int(216 - cy - 9), int(cx), :3]) == (255, 255, 255)
    # All formats take them, also an image without alpha
    ax.image(np.zeros((4, 4, 3), np.uint8), 0.5, 0.5, 8.0, 8.0)
    assert fig.to_pdf().startswith(b'%PDF') and '<image' in fig.to_svg()


def test_display_list_is_the_page_without_its_shadow():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'aura')
    ax.plot([0.0, 1.0], [0.0, 1.0], label = 'a line')
    ops = fig.display_list()
    assert ops and all('kind' in op and 'ctm' in op for op in ops) and not any(op['kind'] == 'image' for op in ops)
    assert any(op['kind'] == 'text' for op in ops) and any(op['kind'] == 'path' and op['stroke'] for op in ops)
    # Without a layout of its own it keeps what the last one settled
    ax.set_xlim(0.0, 0.5)
    assert fig.display_list(layout = False)
    assert ax.xlim != (0.0, 0.5) and fig.layout() and ax.xlim == (0.0, 0.5)


def test_page_and_data_are_each_others_reverse():
    for scale in ('linear', 'log'):
        fig, ax = vc.subplots(figsize = (5, 4), palette = 'poster')
        ax.plot([1.0, 100.0], [1.0, 100.0]).set_xscale(scale).set_yscale(scale)
        fig.layout()
        assert np.allclose(ax.data(*ax.page(np.array([2.0, 50.0]), np.array([3.0, 40.0]))), ([2.0, 50.0], [3.0, 40.0]))


def test_what_leaves_the_axes_goes_with_its_position():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.4])
    ax.plot([1.0, 1.0], [1.1, 1.3], color = '#FF0000', lw = 3.0, clip = False, y_fraction = True)
    ax.annotate('peak', (3.0, 2.0), clip = True)
    ax.image(np.zeros((4, 4, 4), np.uint8) + 255, 3.0, 1.2, 10.0, 10.0, y_fraction = True)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ops = fig._canvas().ops
    assert sum(op['kind'] == 'image' for op in ops) == 1
    assert any(op['kind'] == 'text' and 'peak' in op['text'] for op in ops)
    # Above the axes while in view
    px = fig.to_rgba(72.0)
    cx, cy = ax.page(1.0, 0.0)
    above = px[int(216 - ax.rect[1] - 1.2 * ax.rect[3]), int(cx)]
    assert above[0] > 200 > above[1]
    ax.set_xlim(1.5, 2.5)
    ops = fig._canvas().ops
    assert not any(op['kind'] == 'image' for op in ops)
    assert not any(op['kind'] == 'text' and 'peak' in op['text'] for op in ops)
    assert not any(op['kind'] == 'path' and op['stroke'] and tuple(op['stroke_rgb']) == (1.0, 0.0, 0.0) for op in ops)


def test_offset_moves_a_line_in_points():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.plot([2.0, 2.0], [2.0, 2.0], color = '#FF0000', lw = 3.0, offset = [(0.0, 0.0), (0.0, 30.0)])
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    px = fig.to_rgba(72.0)
    cx, cy = ax.page(2.0, 2.0)
    assert px[int(216 - cy - 15), int(cx), 1] < 100 and px[int(216 - cy - 40), int(cx), 1] > 200


def test_ticks_set_by_hand_follow_narrowed_limits():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.plot([0.0, 100.0], [0.0, 100.0])
    ax.set_yticks([0, 25, 50, 75, 100], ['a', 'b', 'c', 'd', 'e'])
    ax.set_ylim(30, 80)
    fig.layout()
    assert ax.yticks == [50, 75] and ax.ylabels == ['c', 'd']


def test_one_axis_keeps_one_notation():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.plot([0.0, 1.0], [0.0, 3e5])
    fig.layout()
    assert all('10^' in s or s == '0' for s in ax.ylabels)
    assert fmt(80000.0) == '80000' and fmt(80000.0, True) == '8×10^{4}' and fmt(0.0, True) == '0'


def test_colorbar_of_one_value_and_of_counts():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.1, 0.2, 0.6, 0.7])
    ax.scatter([1.0, 2.0], [1.0, 2.0], c = [2.0, 2.0])
    ax.colorbar('Runs', integer = True)
    assert fig.to_rgba(72.0).shape[2] == 4
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.1, 0.2, 0.6, 0.7])
    ax.scatter([1.0, 2.0], [1.0, 2.0], c = [1.0, 2.0])
    ax.colorbar('Runs', integer = True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.layout()
    assert [op['text'] for op in fig._canvas().ops if op['kind'] == 'text' and op['text'] != 'Runs'] == ['1', '2']


def test_sideways_markers_and_a_library_palette():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
    ax.scatter([1.0], [1.0], marker = '>')
    ax.scatter([2.0], [1.0], marker = '<')
    assert fig.to_png(72.0).startswith(b'\x89PNG')
    # A library's own palette wins over another library's default, and gives way to the user's choice
    from viscose.palettes import _default
    saved = list(_default)
    try:
        _default[:] = ['lumen', False]
        vc.default_palette('aura')
        assert vc.get_palette(None, 'poster').name == 'poster' and vc.get_palette('couture', 'poster').name == 'couture'
        vc.set_palette('lumen')
        assert vc.get_palette(None, 'poster').name == 'lumen'
    finally:
        _default[:] = saved


def test_boxplot_shows_quartiles_whiskers_and_fliers():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'lumen')
    ax.boxplot({'a': [1, 2, 3, 4, 5, 100], 'b': [2.0, np.nan, 2.0], 'c': []})
    box = next(a for a in ax.artists if a['kind'] == 'box')
    low, q1, median, q3, high, out = box['stats'][0]
    assert (low, q1, median, q3, high) == (1, 2.25, 3.5, 4.75, 5) and list(out) == [100]
    assert box['stats'][1][:5] == (2, 2, 2, 2, 2) and box['stats'][2] is None
    fig.layout()
    assert ax.xlabels == ['a', 'b', 'c'] and ax.get_ylim()[1] > 100
    assert pixels(fig).shape


def test_boxplot_without_fliers_leaves_them_out_of_the_limits():
    fig, ax = vc.subplots(figsize = (5, 4))
    ax.boxplot([[1, 2, 3, 4, 5, 100]], fliers = False, points = False)
    assert ax.get_ylim()[1] < 10


@pytest.mark.parametrize('name', NAMES)
def test_boxplot_in_every_palette(name, tmp_path):
    fig, ax = vc.subplots(figsize = (5, 4), palette = name)
    ax.boxplot([[1, 2, 3, 4, 9], [2, 3, 4, 5, 6]], labels = ['a', 'b'], colors = ['a3', 'b1'])
    for suffix in ('png', 'pdf', 'svg'):
        assert fig.save(tmp_path / f'box.{suffix}').stat().st_size > 0


def test_contour_lines_follow_their_level():
    grid = np.linspace(-2, 2, 41)
    xx, yy = np.meshgrid(grid, grid)
    fig, ax = vc.subplots(figsize = (4, 4))
    ax.contour(grid, grid, xx ** 2 + yy ** 2, levels = [1.0], color = 'a1')
    lines = next(a for a in ax.artists if a['kind'] == 'contour')['lines']
    assert len(lines) == 1
    xs, ys = lines[0]
    # A closed circle of radius one, walked in one go
    assert (xs[0], ys[0]) == (xs[-1], ys[-1]) and len(xs) > 30
    assert np.allclose(np.hypot(xs, ys), 1.0, atol = 0.01)
    assert pixels(fig).shape


def test_contour_levels_saddles_gaps_and_shape():
    grid = np.linspace(-1, 1, 5)
    xx, yy = np.meshgrid(grid, grid)
    fig, ax = vc.subplots(figsize = (4, 4))
    saddle = xx * yy + 0.01
    saddle[0, 0] = np.nan
    ax.contour(grid, grid, saddle, levels = 3)
    ax.contour(grid, grid, np.ones_like(xx), levels = [5.0])
    assert [len(a['lines']) > 0 for a in ax.artists if a['kind'] == 'contour'] == [True, False]
    assert pixels(fig).shape
    with pytest.raises(ValueError, match = 'one row per y'):
        ax.contour(grid, grid[:-1], xx)


def test_heatmap_annotations_read_over_their_cells():
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'aura')
    ax.heatmap([[1.0, 0.0], [np.nan, -0.5]], annot = True, fmt = '.1f', vmin = -1, vmax = 1)
    written = {a['s']: a for a in ax.artists if a['kind'] == 'text'}
    assert sorted(written) == ['-0.5', '0.0', '1.0']
    assert written['1.0']['rgb'] == (1.0, 1.0, 1.0) and written['0.0']['rgb'] == ax.pal.color('ink')
    assert (written['1.0']['x'], written['1.0']['y']) == (0.5, 1.5)
    assert pixels(fig).shape


def test_reversed_sequential_scale_starts_dark():
    pal = get_palette('aura')
    assert np.allclose(pal.sequential_r(0.0), pal.color('a0')) and np.allclose(pal.sequential_r(1.0), pal.color('a3'))
    fig, ax = vc.subplots(figsize = (4, 4))
    ax.heatmap([[0.0, 1.0]], cmap = 'sequential_r', colorbar = True)
    assert pixels(fig).shape
    with pytest.raises(ValueError, match = 'sequential_r'):
        ax.heatmap([[0.0]], cmap = 'magma')


def test_set_tick_size_and_labels_that_must_all_show_make_room():
    labels = [f'atom {i}' for i in range(100)]
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.heatmap(np.eye(100), labels, labels, show_all = True, cmap = 'sequential').set_tick_size(6.0)
    assert ax._tick_sizes() == (6.0, 6.0)
    fig.layout()
    assert len(ax.xticks) == len(ax.yticks) == 100
    assert ax.rect[2] / 100 >= 6.0 and ax.rect[3] / 100 >= 6.0
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.heatmap(np.eye(100), labels, labels, cmap = 'sequential')
    fig.layout()
    assert len(ax.yticks) < 100 and (fig.width, fig.height) < (4 * 72.0 * 3 + 1, 3 * 72.0 * 3 + 1)


def test_dendrogram_links_take_their_colors_and_keep_their_labels():
    icoord, dcoord = [[5, 5, 15, 15], [25, 25, 35, 35], [10, 10, 30, 30]], [[0, 1, 1, 0], [0, 2, 2, 0], [1, 3, 3, 2]]
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.dendrogram(icoord, dcoord, n = 4, labels = list('abcd'), scale = True, colors = ['a1', 'b1', 'muted'],
                  show_all = True)
    lines = [op for op in fig.display_list() if op['kind'] == 'path' and op['stroke']]
    drawn = {tuple(round(v, 3) for v in op['stroke_rgb']) for op in lines}
    assert {tuple(round(v, 3) for v in ax.pal.color(k)) for k in ('a1', 'b1', 'muted')} <= drawn
    assert ax.xlabels == list('abcd')


def test_small_ticks_keep_one_notation():
    assert tick_labels([0.0008, 0.001, 0.0012]) == ['8×10^{-4}', '10^{-3}', '1.2×10^{-3}']
    assert tick_labels([0, 0.5, 1]) == ['0', '0.5', '1']
    assert tick_labels([0, 0.0005]) == ['0', '5×10^{-4}']


def test_text_runs_and_measure():
    assert runs('log_{2} x') == [('log', 0.0, 1.0), ('2', -0.22, 0.7), (' x', 0.0, 1.0)]
    assert runs('10^{3}')[1] == ('3', 0.38, 0.7)
    assert width('abc', 10, tracking = 1.0) == pytest.approx(width('abc', 10) + 3.0)
    # 'abc' is as wide in Comfortaa Bold as in Comfortaa, and CI has no Century Gothic
    assert width('Title', 10, True) != width('Title', 10)


def test_transparent_background_and_fixed_axes():
    fig = vc.Figure(4, 3, 'poster')
    ax = fig.add_axes((.2, .2, .6, .6))
    ax.plot([0, 1], [0, 1])
    fig._layout()
    assert ax.rect[0] == pytest.approx(.2 * fig.width)
    assert fig.to_png(72, background = None)[25] == 6
    assert fig._repr_png_()[:4] == b'\x89PNG'


def test_svg_shadow_sits_under_the_marks():
    svg = all_marks('lumen').to_svg()
    assert svg.index('<image') < svg.index('<path ') and re.search(r'href="data:image/png;base64,', svg)


def test_svg_keeps_a_large_heatmap_in_drawing_order_and_clipped():
    fig, ax = vc.subplots(palette = 'poster')
    ax.heatmap(np.arange(10000.0).reshape(100, 100), [str(i) for i in range(100)], [str(i) for i in range(100)])
    ax.axvline(50.0)
    svg = fig.to_svg()
    image = svg.index('<image')
    assert svg.rindex('clip-path="url(#', 0, image) > svg.index('<g transform="matrix(')
    assert svg.rindex('<path ') > image


def test_svg_groups_of_a_figure_with_a_shadow_keep_their_ids():
    fig, ax = vc.subplots(palette = 'aura')
    ax.arrow(0, 0, 1, 1, gid = 'gwedge0')
    ax.bar(['a'], [1.0])
    svg = fig.to_svg()
    assert '<image' in svg and re.search(r'<g id="gwedge0">\s*<path d="', svg)


def test_slate_can_tell_a_figure():
    svg = all_marks('aura').to_svg()
    assert '<metadata>viscose' in svg[:3000]
    # annotate_figure's search for labels, which a marker written as a comment would match from the top of the page
    assert not re.findall(r'<!--.*-->[\s\S]*?</g>', svg)
    shown = []
    with patch('builtins.display', shown.append, create = True):
        vc.show_png(b'png')
    assert shown[0]._repr_png_() == b'png' and type(shown[0]).__module__.startswith('viscose.')
    assert vc.Figure.__module__.startswith('viscose.')


def test_shadow_at_high_resolution_matches_the_page_grid():
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'aura')
    ax.bar(['a'], [1])
    fig.layout()
    top, left, planes = fig._shadow_under(300.0)
    assert planes.shape[0] == 4 and top + planes.shape[1] <= 600 and left + planes.shape[2] <= 600
    # Premultiplied, so no channel exceeds its alpha
    assert planes.dtype == np.float32 and (planes[:3] <= planes[3] + 1e-6).all() and planes.min() >= 0
    # The soft shadow reaches past the bar but fades out before the corner of its box
    assert planes[3, 0, 0] < 1 / 255 and planes[3].max() > 0.1


def test_colliding_tick_labels_turn_and_then_thin():
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'lumen')
    ax.bar([f'a long category {i}' for i in range(12)], range(12))
    fig._layout()
    assert ax.xrot == 90.0 and len(ax.xlabels) == 12
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'poster')
    ax.heatmap(np.zeros((200, 4)), list('abcd'), [f'row{i}' for i in range(200)])
    fig._layout()
    assert 10 < len(ax.ylabels) < 200
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'poster')
    ax.heatmap(np.zeros((200, 4)), list('abcd'), [f'row{i}' for i in range(200)], show_all = True)
    fig._layout()
    assert len(ax.ylabels) == 200


def test_rotated_ticks_at_any_angle_leave_room():
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'lumen')
    ax.bar(['a label', 'another label'], [1, 2])
    ax.set_xticks([0, 1], ['a label', 'another label'], rotation = 45.0)
    fig._layout()
    assert ax._margins()[2] > 20 and fig.to_png(72)[:4] == b'\x89PNG'


def test_long_labels_widen_the_figure_rather_than_squeeze_the_plot():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'couture')
    ax.heatmap(np.zeros((3, 3)), list('abc'), ['G' * 90, 'x', 'y'])
    fig._layout()
    assert fig.width > 5 * 72 and ax.rect[2] > 0.3 * fig.width


def test_labels_stay_on_the_page():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate('A label far too long to fit on the right of this point', (0.9, 0.5), offset = (5, 0))
    c = fig._canvas()
    op = [o for o in c.ops if o['kind'] == 'text' and o['text'].startswith('A label')][0]
    # Flipped to end at its point: the text starts at x, in front of the anchor that the transform holds
    assert op['x'] < 0 and op['ctm'][4] < fig.width


def test_large_heatmap_is_one_image_with_holes_for_missing_values():
    m = np.random.default_rng(0).random((100, 60))
    m[0, 0] = np.nan
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'poster')
    ax.heatmap(m, cmap = 'sequential')
    c = fig._canvas()
    images = [o for o in c.ops if o['kind'] == 'image']
    assert len(images) == 1 and images[0]['alpha'][0] == 0 and images[0]['alpha'][1] == 255
    assert fig.to_pdf().count(b'/Subtype /Image') >= 1 and '<image' in fig.to_svg() and fig.to_png(72)[:4] == b'\x89PNG'


def test_svg_has_a_white_page():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.plot([0, 1], [0, 1])
    assert '<rect width="216" height="216" fill="#FFFFFF" />' in fig.to_svg()


def test_many_discs_are_stamped_in_raster_output_only():
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'poster')
    rng = np.random.default_rng(0)
    ax.scatter(rng.random(2000), rng.random(2000), s = 16, color = 'a1')
    kinds = lambda c: {o['kind'] for o in c.ops if o.get('mark')}
    assert kinds(fig._canvas(stamp_dpi = 150.0)) == {'image'} and kinds(fig._canvas()) == {'path'}
    assert b'/Subtype /Image' not in fig.to_pdf() and fig.to_png(72)[:4] == b'\x89PNG'


@pytest.mark.parametrize('stamped', [True, False])
def test_later_points_cover_the_rings_of_earlier_ones(stamped):
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, 0.3, 6000), rng.normal(0, 0.3, 6000), s = 20, color = '#FF0000')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    from viscose import marks
    old, marks.STAMP_MIN = marks.STAMP_MIN, 300 if stamped else 10 ** 9
    try:
        img = pixels(fig, 100)
    finally:
        marks.STAMP_MIN = old
    x0, y0, w, h = ax.rect
    sc = 100 / 72.0
    top, bottom = int((fig.height - y0 - h) * sc) + 5, int((fig.height - y0) * sc) - 5
    middle = img[top:bottom, int(x0 * sc) + 5:int((x0 + w) * sc) - 5]
    assert (middle[..., 1] < 128).mean() > 0.5


def test_transparent_points_build_up_where_they_overlap():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'aura')
    ax.scatter(np.full(600, 0.5), np.full(600, 0.5), s = 300, color = '#FF0000', alpha = 0.02)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.pal = vc.get_palette('poster')
    for a in fig.axes:
        a.pal = fig.pal
    assert pixels(fig, 100)[..., 1].min() < 80


def test_ticks_never_shrink_to_one():
    assert vc.axes.nice_ticks(0.1, 9.9, 4) == [2, 4, 6, 8]
    assert vc.axes.nice_ticks(0.0, 1.0) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_two_long_labels_grow_the_page_by_the_longer_one():
    def width(labels):
        fig, ax = vc.subplots(figsize = (4, 3), palette = 'couture')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i, s in enumerate(labels):
            ax.annotate(s, (0.5, 0.2 + 0.5 * i))
        fig._layout()
        return fig.width
    long = 'Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc'
    assert width([long, long]) == pytest.approx(width([long]))


def test_a_heatmap_y_label_clears_its_color_bar():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'lumen')
    ax.heatmap(np.arange(12.0).reshape(3, 4), list('abcd'), list('uvw'), colorbar = True)
    ax.set_ylabel('Glycans')
    c = fig._canvas()
    label = [o for o in c.ops if o['kind'] == 'text' and o['text'] == 'Glycans'][0]
    bar = [o for o in c.ops if o['kind'] == 'path' and o.get('fill_grad') is not None][0]
    assert label['ctm'][4] < min(s[1] for s in bar['path'] if s[0] in 'ml') - 1.0


def test_stamped_discs_match_traced_discs():
    def draw(n):
        rng = np.random.default_rng(1)
        fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
        ax.scatter(rng.random(n), rng.random(n), s = 60, color = 'a1')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    from viscose import marks
    old, marks.STAMP_MIN = marks.STAMP_MIN, 10 ** 9
    try:
        traced = pixels(draw(400), 150).astype(float)
    finally:
        marks.STAMP_MIN = old
    stamped = pixels(draw(400), 150).astype(float)
    # Their edges may differ by a fraction of a pixel, but never on average
    assert np.abs(traced - stamped).mean() < 2.0


def test_hue_style_and_size_map_to_color_shape_and_area():
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'poster')
    x = np.arange(6.0)
    ax.scatter(x, x, hue = list('aabbcc'), style = list('xyxyxy'), size = x + 1)
    pts = [a for a in ax.artists if a['kind'] == 'points']
    assert len(pts) == 6 and {a['marker'] for a in pts} == {'o', 'X'} and len({a['rgb'] for a in pts}) == 3
    assert pts[0]['s'][0] < pts[-1]['s'][0]
    labels = [i[0] for i in ax.legend_items]
    assert labels[:4] == ['Color', 'a', 'b',
                          'c'] and 'Shape' in labels and 'Size' in labels and ax._legend_opts == 'right'
    assert fig.to_png(72)[:4] == b'\x89PNG'


def _every_plot(palette):
    x = np.linspace(0.1, 9, 30)
    fig, axes = vc.subplots(3, 3, figsize = (12, 11), palette = palette)
    fig.suptitle('Every plot', 'in one figure')
    axes[0][0].plot(x, np.sin(x), label = 'sine', marker = True)
    axes[0][0].axhline(0.5, label = 'half')
    axes[0][0].legend()
    axes[0][1].scatter(x, np.cos(x), hue = ['a', 'b', 'c'] * 10, size = x, style = ['p', 'q'] * 15)
    axes[0][2].bar([1, 2, 3], [3, 1, 2], color = ['a1', '#112233', 'tomato'])
    axes[1][0].hist(np.sin(x), bins = 8)
    axes[1][0].fill_between(x[:5], 1, 2)
    axes[1][1].boxplot({'one': x[:15], 'two': x[15:]})
    axes[1][1].annotate('a note', (0, 3))
    axes[1][2].heatmap(np.outer(x[:6], x[:4]), annot = True)
    axes[2][0].stacked_bar([1, 2], [[1, 2], [3, 4]], labels = ['u', 'v'])
    axes[2][0].set(xlabel = 'x', ylabel = 'y')
    axes[2][1].ridgeline(x, [np.sin(x) + 1, np.cos(x) + 1], labels = ['r', 's'])
    axes[2][2].stem(x[:6], x[:6])
    axes[2][2].set_yscale('log')
    axes[2][2].errorbar(x[:6], x[:6], 0.1)
    return fig


def test_a_figure_in_another_palette_equals_one_drawn_in_it():
    moved, direct = _every_plot('poster').in_palette('couture'), _every_plot('couture')
    assert moved.pal.name == 'couture'
    assert moved.to_png(72.0) == direct.to_png(72.0)
    assert moved.to_pdf() == direct.to_pdf()
    twice = _every_plot('lumen').in_palette('aura').in_palette('poster')
    assert twice.to_png(72.0) == _every_plot('poster').to_png(72.0)


def test_rendering_records_nothing_and_leaves_the_original_alone():
    fig = _every_plot('poster')
    before = fig.to_png(72.0)
    calls = len(fig._calls)
    fig.to_svg()
    fig.in_palette('aura').to_svg()
    assert len(fig._calls) == calls and fig.pal.name == 'poster' and fig.to_png(72.0) == before


def test_a_clustermap_in_another_palette_equals_one_drawn_in_it():
    data = np.random.default_rng(1).normal(0, 1, (12, 6))
    fig, heat = vc.clustermap(data, palette = 'poster', title = 'Map', colorbar_label = 'z')
    direct = vc.clustermap(data, palette = 'aura', title = 'Map', colorbar_label = 'z')[0]
    moved = fig.in_palette('aura')
    assert moved.row_order == fig.row_order and moved.to_png(72.0) == direct.to_png(72.0)


def _source_of(path):
    """The JSON a saved PDF, SVG, or PNG carries, read without viscose."""
    import base64
    import json
    import zlib
    data = path.read_bytes()
    if data[:4] == b'%PDF':
        stream = re.search(rb'/Subtype /application#2Fvnd\.viscose\+json .*?/Length (\d+) >>\nstream\n', data)
        packed = data[stream.end():stream.end() + int(stream.group(1))]
    elif data[:4] == b'\x89PNG':
        at = data.index(b'vcSc')
        packed = data[at + 4:at + int.from_bytes(data[at - 4:at], 'big') + 4]
    else:
        packed = base64.b64decode(re.search(rb'<metadata id="viscose-source">([^<]*)</metadata>', data).group(1))
    return json.loads(zlib.decompress(packed).partition(b'\n')[0])


def test_a_figure_comes_back_from_its_own_file(tmp_path):
    fig = _every_plot('poster')
    direct = _every_plot('couture')
    for suffix in ('pdf', 'svg', 'png'):
        path = fig.save(tmp_path / f'every.{suffix}', dpi = 72)
        back = vc.open(path)
        assert back.pal.name == 'poster' and back.to_pdf() == fig.to_pdf() and back.to_png(72.0) == fig.to_png(72.0)
        assert vc.open(path.read_bytes(), 'couture').to_png(72.0) == direct.to_png(72.0)
        # Without its source, a file is what to_pdf, to_svg, and to_png give
        assert fig.save(tmp_path / f'plain.{suffix}', dpi = 72, source = False).read_bytes() == {
            'pdf': fig.to_pdf, 'svg': lambda: fig.to_svg().encode(), 'png': lambda: fig.to_png(72.0)}[suffix]()
    # Saving twice writes the same file, and a reopened figure saves its own source again
    assert fig.save(tmp_path / 'again.pdf').read_bytes() == fig.save(tmp_path / 'every.pdf').read_bytes()
    assert vc.open(vc.open(tmp_path / 'every.svg').save(tmp_path / 'third.svg')).to_png(72.0) == fig.to_png(72.0)
    assert 'open' not in vc.__all__


def test_a_clustermap_a_journal_page_and_a_dark_or_custom_palette_come_back(tmp_path):
    data = np.random.default_rng(1).normal(0, 1, (12, 6))
    fig = vc.clustermap(data, palette = 'aura', title = 'Map', colorbar_label = 'z')[0]
    back = vc.open(fig.save(tmp_path / 'map.pdf'))
    assert back.row_order == fig.row_order and back.to_png(72.0) == fig.to_png(72.0)
    # Clustered once: the file holds the tree's coordinates, and reopening never clusters again
    with patch('viscose.figure.linkage', side_effect = AssertionError):
        vc.open(tmp_path / 'map.pdf')
    page = vc.Figure(89, 60, 'lumen', unit = 'mm')
    ax = page.add_axes((12, 6, 70, 40), unit = 'mm')
    ax.plot(range(5), [1, 3, 2, 5, 4])
    page.letter(2, 3, 'a', 'Rises')
    page.text(0.5, 0.02, 'note', ha = 'center')
    assert vc.open(page.save(tmp_path / 'page.svg')).to_pdf() == page.to_pdf()
    wide = vc.open(tmp_path / 'page.svg', size = (183, 60), unit = 'mm')
    assert round(wide.width / vc.figure.UNITS['mm'], 6) == 183
    previous = vc.dark_mode(True, '#2B2B2B')
    try:
        dark = _every_plot('lumen')
    finally:
        vc.dark_mode(*previous)
    back = vc.open(dark.save(tmp_path / 'dark.png', dpi = 72))
    assert back.pal.paper == '#2B2B2B' and back.to_png(72.0) == dark.to_png(72.0)
    big = dataclasses.replace(vc.get_palette('couture'), name = 'couture-big', tick_size = 14.0)
    custom = _every_plot(big)
    assert vc.open(custom.save(tmp_path / 'custom.pdf')).to_png(72.0) == custom.to_png(72.0)
    assert 'settings' in _source_of(tmp_path / 'custom.pdf')['palette']
    assert _source_of(tmp_path / 'map.pdf')['palette'] == 'aura'


def test_a_source_is_json_that_names_methods_and_runs_no_code(tmp_path):
    fig, ax = vc.subplots(figsize = (5, 4))
    pct = lambda v: f'{v:+.1f}'
    ax.heatmap(np.arange(12.0).reshape(3, 4), annot = True, fmt = pct)
    ax.set_title('With "quotes", ü, and a line\nbreak')
    path = fig.save(tmp_path / 'fmt.svg')
    source = _source_of(path)
    assert source['format'] == 1 and source['calls'][0][1] == 'heatmap' and source['arrays'][0]['values'][:3] == [
        0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match = 'functions'):
        vc.open(path)
    assert vc.open(path, functions = {'<lambda>': pct}).to_png(72.0) == fig.to_png(72.0)
    # A file that names anything but a plot method opens nothing
    import base64
    import json
    import zlib
    source['calls'][0][1] = '_canvas'
    forged = zlib.compress(json.dumps(source).encode())
    text = path.read_text(encoding = 'utf-8')
    text = re.sub(r'(<metadata id="viscose-source">)[^<]*', lambda m: m.group(1) + base64.b64encode(forged).decode(),
                  text)
    with pytest.raises(ValueError, match = 'not a plot method'):
        vc.open(text.encode())
    with pytest.raises(ValueError, match = 'carries no figure source'):
        vc.open(fig.save(tmp_path / 'bare.pdf', source = False))


def test_large_arrays_follow_the_json_exactly_and_a_large_source_stays_out(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    n = 30000
    x = rng.normal(size = n)
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'poster')
    ax.scatter(x, x.astype(np.float32).astype(float), s = 4)
    ax.plot(np.arange(n, dtype = np.int64), np.where(np.arange(n) % 7 == 0, np.nan, x), alpha = 0.2)
    path = fig.save(tmp_path / 'large.png', dpi = 40, source = True)
    assert {a.get('store') for a in _source_of(path)['arrays']} == {'<f8', '<f4', '<i2'}
    assert vc.open(path).to_png(40.0) == fig.to_png(40.0)
    ctx = {'axes': {}, 'arrays': [], 'seen': {}, 'tail': []}
    vc.figure._encode([np.arange(n) % 3 == 0, np.arange(n) % 3 == 0], ctx)
    assert len(ctx['arrays']) == 1 and ctx['arrays'][0]['store'] == '|u1'
    # Past SOURCE_FLOOR and larger than the file, a source stays out, with a warning naming the way to keep it
    monkeypatch.setattr(vc.figure, 'SOURCE_FLOOR', 1000)
    small, ax = vc.subplots(figsize = (3, 2))
    # Names are kept for each point and never drawn, so these make a source far larger than the page
    ax.scatter(rng.normal(size = 500), rng.normal(size = 500), s = 2,
               names = [rng.bytes(800).hex() for _ in range(500)])
    for suffix in ('pdf', 'svg', 'png'):
        with pytest.warns(UserWarning, match = 'source = True keeps it'):
            plain = small.save(tmp_path / f'small.{suffix}', dpi = 20)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            kept = small.save(tmp_path / f'kept.{suffix}', dpi = 20, source = True)
        assert kept.stat().st_size > plain.stat().st_size and vc.open(kept).to_png(20.0) == small.to_png(20.0)
    # A PDF/X is the page alone, for print
    with pytest.raises(ValueError, match = 'carries no figure source'):
        vc.open(fig.save(tmp_path / 'print.pdf', press = True))


def test_long_form_keeps_only_its_columns_and_their_categories(tmp_path):
    pd = pytest.importorskip('pandas')
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'class': pd.Categorical(rng.choice(['high-Man', 'complex', 'hybrid'], 90),
                                               categories = ['hybrid', 'complex', 'high-Man']),
                       'tissue': rng.choice(['tumor', 'normal'], 90), 'value': rng.gamma(2, 3, 90),
                       'unused': ['x' * 500] * 90}, index = np.arange(100, 190))
    fig, ax = vc.subplots(figsize = (6, 4))
    ax.boxplot(df, x = 'class', y = 'value', hue = 'tissue')
    ax.bracket('hybrid', 'complex', 0.003)
    path = fig.save(tmp_path / 'long.pdf')
    frame = _source_of(path)['calls'][0][2][0]['$frame']
    assert [c for c, _ in frame] == ['class', 'value', 'tissue'] and frame[0][1]['categories'][0] == 'hybrid'
    assert vc.open(path).to_png(72.0) == fig.to_png(72.0)


def test_a_figure_held_to_a_window_stays_held_in_another_palette():
    fig, ax = vc.subplots(palette = 'poster')
    ax.plot([1, 2], [3, 4], label = 'a long name that needs room on the page')
    fig.set_size(3, 2)
    moved = fig.in_palette('aura')
    moved.layout()
    assert (moved.width, moved.height) == (3 * 72.0, 2 * 72.0)


def test_names_follow_their_points_through_the_groups():
    fig, ax = vc.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6], hue = ['a', 'b', 'a'], names = ['one', 'two', 'three'])
    points = [a for a in ax.artists if a['kind'] == 'points']
    assert [(a['x'].tolist(), a['names'].tolist()) for a in points] == [([1, 3], ['one', 'three']), ([2], ['two'])]
    plain, axis = vc.subplots()
    axis.scatter([1, 2], [3, 4], names = ['one', 'two'])
    assert axis.artists[0]['names'].tolist() == ['one', 'two']
    plain.to_svg()
    with pytest.raises(ValueError):
        axis.scatter([1, 2], [3, 4], names = ['one'])


def test_heatmap_cells_know_their_names_also_undrawn():
    fig, ax = vc.subplots()
    ax.heatmap(np.eye(2), ['s1', 's2'], ['g1', 'g2'])
    assert (ax.artists[0]['rows'], ax.artists[0]['columns']) == (['g1', 'g2'], ['s1', 's2'])
    ax.heatmap(np.eye(2), names = (['g1', 'g2'], None))
    assert (ax.artists[1]['rows'], ax.artists[1]['columns']) == (['g1', 'g2'], None)
    with pytest.raises(ValueError):
        ax.heatmap(np.eye(2), names = (['g1'], None))
    frame = type('Frame', (), {'columns': list('xyz'), 'index': list('abcd'),
                               '__array__': lambda self, *a, **k: np.arange(12.0).reshape(4, 3) ** 2})()
    fig, heat = vc.clustermap(frame, xticklabels = 2, yticklabels = False, palette = 'couture')
    cells = heat.artists[0]
    assert cells['rows'] == [frame.index[i] for i in fig.row_order]
    assert cells['columns'] == [frame.columns[i] for i in fig.col_order]
    assert not heat._ytl and fig.to_svg()
    fig, heat = vc.clustermap(np.arange(12.0).reshape(4, 3) ** 2, palette = 'couture')
    assert heat.artists[0]['rows'] is None and heat.artists[0]['columns'] is None


def test_hue_keeps_explicit_colors_and_markers():
    fig, ax = vc.subplots(palette = 'lumen')
    ax.scatter([0, 1], [0, 1], hue = ['up', 'down'], colors = {'up': 'a1', 'down': 'b1'}, style = ['p', 'q'],
               markers = {'p': 'D', 'q': '*'},
               legend = False)
    pts = [a for a in ax.artists if a['kind'] == 'points']
    assert pts[0]['rgb'] == fig.pal.color('a1') and pts[1]['rgb'] == fig.pal.color('b1')
    assert pts[0]['marker'] == 'D' and pts[1]['marker'] == '*' and ax._legend_opts is None


def test_numeric_hue_with_many_values_is_a_scale():
    fig, ax = vc.subplots(palette = 'lumen')
    ax.scatter(np.arange(20.0), np.arange(20.0), hue = np.arange(20.0))
    ax.colorbar('Value')
    assert fig.to_png(72)[:4] == b'\x89PNG' and ax._scale['hi'] == 19.0 and ax._margins()[1] > 20


@pytest.mark.parametrize('marker', list(vc.marks.MARKERS))
def test_every_marker_is_drawn(marker):
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'lumen')
    ax.scatter([0.5], [0.5], s = 400, color = '#FF0000', marker = marker)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    assert (pixels(fig, 144) < 200).any()


def test_a_long_legend_wraps_inside_the_figure():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'couture')
    ax.scatter(np.arange(30.0), np.arange(30.0), hue = [f'A long group name {i % 30}' for i in range(30)],
               legend = 'below')
    fig._layout()
    assert ax._legend_size[1] > 16.0 and ax.rect[1] > 16.0
    fig, ax = vc.subplots(figsize = (5, 3), palette = 'poster')
    ax.scatter(np.arange(40.0), np.arange(40.0), hue = [f'g{i}' for i in range(40)], legend = 'right')
    fig._layout()
    assert ax._legend_size[1] <= ax.rect[3] + 1.0


def test_colorbar_on_heatmap_and_in_clustermap_corner():
    m = np.random.default_rng(1).normal(size = (6, 4))
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'aura')
    ax.heatmap(m, list('abcd'), list('uvwxyz'), colorbar = True, colorbar_label = 'z-score')
    assert ax._cbar['label'] == 'z-score' and fig.to_png(72)[:4] == b'\x89PNG'
    drawn = [o['text'] for o in fig._canvas().ops if o['kind'] == 'text']
    assert 'z-score' in drawn
    fig, heat = vc.clustermap(m, palette = 'lumen', colorbar_label = 'Abundance')
    assert 'Abundance' in [o['text'] for o in fig._canvas().ops if o['kind'] == 'text']


def test_clustermap_parity():
    pytest.importorskip('scipy')
    from scipy.cluster.hierarchy import leaves_list, linkage
    pd = pytest.importorskip('pandas')
    m = np.random.default_rng(1).normal(size = (8, 5))
    df = pd.DataFrame(m, index = [f'g{i}' for i in range(8)], columns = list('abcde'))
    z_rows, z_cols = linkage(m, 'single'), linkage(m.T, 'single')
    fig, heat = vc.clustermap(df, row_linkage = z_rows, col_linkage = z_cols, xlabel = 'Samples', ylabel = 'Glycans',
                              vmin = -2, vmax = 2)
    assert list(fig.row_order) == list(leaves_list(z_rows)[::-1]) or sorted(fig.row_order) == list(range(8))
    assert heat._ytl[::-1] == [f'g{i}' for i in fig.row_order] and heat.xlabel == 'Samples' and heat._scale['hi'] == 2.0
    fig._layout()
    x0, _, w, _ = heat.rect
    drawn = [o for o in fig._canvas().ops if o['kind'] == 'text' and o['text'] == 'Glycans']
    assert drawn and drawn[0]['ctm'][4] > x0 + w
    fig, heat = vc.clustermap(df, row_cluster = False, col_cluster = False, colorbar = False, cmap = 'sequential')
    assert fig.row_order == list(range(8)) and fig.col_order == list(range(5)) and fig.to_png(72)[:4] == b'\x89PNG'


def test_svg_label_is_written_for_annotate_figure():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.scatter([0.2], [0.5], s = 30)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate('Neu5Ac(a2-3)Gal', (0.2, 0.5), offset = (5, 0), svg_label = True)
    svg = fig.to_svg()
    m = re.search(r'<!-- Neu5Ac\(a2-3\)Gal -->\s*<g transform="translate\(([\d.]+) ([\d.]+)\)', svg)
    assert m
    # In page coordinates, where SVG's y runs down from the top
    x, y = float(m.group(1)), float(m.group(2))
    assert abs(x - (float(ax._tx(0.2)) + 5)) < 0.01 and abs(y - (fig.height - float(ax._ty(0.5)))) < 8
    assert re.search(r'<!--.*-->[\s\S]*?</g>', svg).group(0).count('<g') == 1
    assert b'%PDF' == fig.to_pdf()[:4]


def test_svg_lines_can_be_read_for_placing_glycans():
    # glycowork's _figure_ink reads what a figure has drawn from these lines, to keep glycan drawings clear of it
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'aura')
    ax.scatter([1, 2], [1, 2])
    ax.scatter([1.5], [1.5], s = 100, edge = ('muted', 1.0), gid = 'node')
    ax.axhline(1.5)
    ax.plot([1, 2], [2, 1])
    ax.set(title = 'A title', xlabel = 'x')
    ax.annotate('Gal(b1-3)GalNAc', (1, 1), svg_label = True)
    svg = fig.to_svg()
    lines = svg.split('\n')
    # y runs up from the page's foot, and the smallest clip is the plot area, a point wider all around
    assert f'<g transform="matrix(1 0 0 -1 0 {fig.height:g})">' in lines
    clips = [[float(v) for v in m] for m in re.findall(
        r'<clipPath[^>]*><rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)]
    x0, y0, w, h = min(clips, key = lambda c: c[2] * c[3])
    assert np.allclose((x0 + 1, y0 + 1, w - 2, h - 2), ax.rect, atol = 0.01)
    numbers = lambda d: np.array(re.findall(r'-?\d*\.?\d+', d), dtype = float).reshape(-1, 2)
    # Marks stand in absolute page coordinates: points are filled paths within a clip, flat ones one subpath each
    point = next(line for line in lines if 'clip-path' in line and 'fill="none"' not in line)
    parts = [numbers(part) for part in re.search(r'<path d="([^"]*)"', point).group(1).split('M')[1:]]
    middles = [(xy.min(0) + xy.max(0)) / 2 for xy in parts]
    assert np.allclose(middles, [(ax._tx(1), ax._ty(1)), (ax._tx(2), ax._ty(2))], atol = 0.05)
    # a line is a stroke without a fill, and a guide also has dashes
    strokes = [line for line in lines if 'clip-path' in line and 'fill="none"' in line]
    assert sorted('stroke-dasharray' in line for line in strokes) == [False, True]
    # a node with a gid has no clip, and its fill and its stroke on one line
    assert re.search(r'<g id="node">\n<path d="[^"]*" fill="#[^"]*" stroke="#[^"]*" stroke-width="[^"]*" />', svg)
    # text is outlines, each with a transform and no clip
    text = [line for line in lines if re.match(r'<path d="[^"]*" transform="matrix\(', line)]
    assert len(text) > 2 and not any('clip-path' in line or 'stroke=' in line for line in text)
    title = numbers(re.search(r'<path d="([^"]*)"', text[-1]).group(1))
    a, b, c, d, _e, f = (float(v) for v in re.search(r'matrix\(([^)]*)\)', text[-1]).group(1).split())
    assert (a, b, c, d) == (1, 0, 0, 1) and f + title[:, 1].min() > y0 + h
    # A label left as text says how far it reaches, in page coordinates
    assert re.search(r'<g transform="translate\([\d.]+ [\d.]+\) matrix\([^)]*\)" data-right="[\d.]+">', svg)


def test_a_glycan_label_near_the_edge_keeps_its_side_and_grows_the_page():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', (0.9, 0.5), offset = (5, 0), svg_label = True)
    x = float(re.search(r'translate\(([\d.]+) ', fig.to_svg()).group(1))
    assert abs(x - (float(ax._tx(0.9)) + 5)) < 0.01 and fig.width > 4 * 72


def labeled_figure(*labels, shape = (1, 1), column = 0):
    fig, axes = vc.subplots(*shape, figsize = (4, 3), palette = 'poster')
    ax = np.ravel(axes)[column]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for label, tag in labels:
        ax.annotate(label, (0.9 if tag else 0.1, 0.5), offset = (5, 0), svg_label = tag)
    svg = fig.to_svg()
    bare = re.search(r'<rect width="[\d.]+" height="[\d.]+" fill="[^"]*" data-width-unlabeled="([\d.]+)"', svg)
    return fig, svg, float(bare.group(1)) if bare else None


def test_a_glycan_label_records_what_the_page_needs_without_it():
    # The plain label is too long for either side of its point, and its point leaves more room on the right
    glycan, plain = 'Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc' * 2, 'a label that stays ' * 3
    fig, svg, bare = labeled_figure((glycan, True))
    assert abs(bare - 4 * 72) < 0.01 and fig.width > 4 * 72
    # What is left as text says how far it reaches, and slate's mark stays right after the root tag
    right = float(re.search(r'data-right="([\d.]+)"', svg).group(1))
    assert abs(right - (float(fig.axes[0]._tx(0.9)) + 5 + width(glycan, 8.6, fig.pal.text_bold))) < 1
    assert re.search(r'<svg[^>]*>\n<metadata>viscose</metadata>\n', svg)
    # Another label keeps the room it needs, as much as on a page of its own
    fig, _, none = labeled_figure((plain, False))
    assert fig.width > 4 * 72 and none is None
    assert abs(labeled_figure((glycan, True), (plain, False))[2] - fig.width) < 0.01
    # Only the last column's growth can go, as an earlier one has a column to its right
    fig, _, bare = labeled_figure((glycan, True), shape = (1, 2))
    assert abs(bare - fig.width) < 0.01
    fig, _, bare = labeled_figure((glycan, True), shape = (1, 2), column = 1)
    assert abs(bare - 4 * 72) < 0.01 and fig.width > 4 * 72


def test_element_ids_wrap_their_paths():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'lumen')
    ax.arrow(0, 0, 1, 1, gid = 'gwedge0')
    ax.scatter([0.5], [0.5], s = 60, gid = 'gwnode0', edge = ('#888888', 1.0))
    ax.off()
    svg = fig.to_svg()
    assert re.search(r'<g id="gwedge0">\s*<path d="', svg) and re.search(r'<g id="gwnode0">\s*<path d="', svg)


def test_arrow_points_at_its_target():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.arrow(0.1, 0.5, 0.9, 0.5, color = '#FF0000', lw = 3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    img = pixels(fig, 144)
    red = (img[:, :, 0] > 200) & (img[:, :, 1] < 80)
    x0, y0, w, h = ax.rect
    ys, xs = np.nonzero(red)
    assert abs(xs.max() / 2.0 - (x0 + w * 0.9)) < 2.0 and abs(ys.mean() / 2.0 - (fig.height - y0 - h / 2)) < 2.0
    # Wider at the head than along the shaft
    assert red.sum(axis = 0).max() > 1.5 * red[:, xs.min() + 10].sum()


def test_axes_fraction_text_and_multiple_lines():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.plot([0, 100], [0, 100])
    ax.text(0.02, 0.98, 'F = 1.2\np = 0.03', axes_fraction = True, va = 'top')
    c = fig._canvas()
    a = [o for o in c.ops if o['kind'] == 'text' and o['text'] == 'F = 1.2'][0]
    b = [o for o in c.ops if o['kind'] == 'text' and o['text'] == 'p = 0.03'][0]
    x0, y0, w, h = ax.rect
    assert abs(a['ctm'][4] - (x0 + 0.02 * w)) < 0.01 and abs(a['ctm'][5] - (y0 + 0.98 * h)) < 0.01
    assert b['ctm'][5] < a['ctm'][5]
    assert ax.xlim[1] < 200


def test_hlines_set_invert_and_margins():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'aura')
    ax.hlines([0, 1, 2], [-1, -2, 0], [1, 0, 2])
    ax.set(xlabel = 'Effect', ylabel = 'Study', title = 'Forest', ylim = (2.5, -0.5))
    ax.margins(0.1)
    ax._limits()
    assert ax.xlabel == 'Effect' and ax.ylim == (2.5, -0.5) and ax.xlim == pytest.approx((-2.4, 2.4))
    fig, ax = vc.subplots()
    ax.plot([0, 1], [0, 1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax._limits()
    assert ax.ylim[0] > ax.ylim[1] and ax.xlim[0] > ax.xlim[1] and fig.to_png(72)[:4] == b'\x89PNG'


def test_a_label_too_long_for_either_side_grows_the_page():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'couture')
    ax.scatter([0.5], [0.5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label = 'Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc'
    ax.annotate(label, (0.5, 0.5), offset = (5, 0))
    c = fig._canvas()
    a = [o for o in c.ops if o['kind'] == 'text' and o['text'] == label][0]
    tw = vc.text.width(label, 8.6, False)
    assert fig.width > 4 * 72
    assert a['ctm'][4] + a['x'] >= 0 and a['ctm'][4] + a['x'] + tw <= fig.width + 1.0


def test_show_goes_to_display_when_there_is_one_and_is_silent_otherwise(monkeypatch):
    import builtins
    seen = []
    monkeypatch.setattr(builtins, 'display', lambda obj: seen.append(obj._repr_png_()), raising = False)
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'poster')
    ax.plot([0, 1], [0, 1])
    fig.show(72)
    assert seen and seen[0][:4] == b'\x89PNG'
    monkeypatch.delattr(builtins, 'display')
    # Under pytest, stdin is no terminal, so nothing opens
    assert fig.show(72) is None


@pytest.mark.parametrize('variable', ['CLAUDECODE', 'CI', 'PYTEST_CURRENT_TEST', 'VISCOSE_NO_VIEWER'])
def test_show_opens_no_viewer_in_a_console_run_by_a_tool(monkeypatch, variable):
    import os
    import sys
    from viscose import figure
    opened = []
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(os, 'startfile', opened.append, raising = False)
    monkeypatch.setattr(sys, 'stdin', type('Console', (), {'isatty': lambda self: True})())
    for name in figure.NO_VIEWER:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv(variable, '1')
    figure.show_png(b'png')
    assert not opened
    monkeypatch.delenv(variable)
    figure.show_png(b'png')
    assert len(opened) == 1
    os.remove(opened[0])


def test_clearing_the_users_choice_gives_way_to_the_library_again():
    module = sys.modules['viscose.palettes']
    before, library = list(module._default), list(module._library)
    try:
        module._default[:] = ['lumen', False]
        vc.default_palette('aura')
        vc.set_palette('poster')
        assert vc.subplots()[0].pal.name == 'poster'
        vc.set_palette(None)
        assert vc.subplots()[0].pal.name == 'aura'
        vc.default_palette('couture')
        assert vc.subplots()[0].pal.name == 'couture'
    finally:
        module._default[:], module._library[:] = before, library


def test_a_library_default_gives_way_to_the_users_choice():
    state = sys.modules['viscose.palettes']._default
    before = list(state)
    try:
        state[:] = 'lumen', False
        vc.default_palette('aura')
        assert vc.subplots()[0].pal.name == 'aura'
        vc.set_palette('poster')
        vc.default_palette('aura')
        assert vc.subplots()[0].pal.name == 'poster'
    finally:
        state[:] = before


def test_css_names_and_clear_color_errors():
    pal = vc.get_palette('aura')
    assert pal.color('CornflowerBlue') == pal.color('#6495ED') and pal.css('saddlebrown') == '#8B4513'
    assert pal.css('a1') == '#A3214E' and pal.color('white') == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match = 'notacolor'):
        pal.color('notacolor')
    fig, ax = vc.subplots(palette = 'aura')
    with pytest.raises(ValueError, match = 'vlag'):
        ax.heatmap(np.eye(2), cmap = 'vlag')
    with pytest.raises(ValueError, match = 'name'):
        ax.scatter([0.0], [0.0], c = [1.0], cmap = 'name')


def test_hue_groups_stand_apart():
    pal = vc.get_palette('aura')
    assert pal.contrast(2) == [pal.color('a1'), pal.color('b1')] and len(set(pal.contrast(8))) == 8
    assert len({tuple(np.round(c, 3)) for c in pal.contrast(12)}) == 12
    fig, ax = vc.subplots(palette = 'aura')
    ax.scatter([0, 1], [0, 1], hue = ['Control', 'Tumor'])
    assert [a['rgb'] for a in ax.artists] == pal.contrast(2)


def test_histogram_frequency_is_counts_per_unit():
    fig, ax = vc.subplots(palette = 'poster')
    ax.hist([0.1, 0.1, 0.1, 0.9], bins = [0.0, 0.5, 1.0], frequency = True)
    assert list(ax.artists[0]['h']) == [6.0, 2.0]


def test_empty_plots_draw_nothing_without_failing():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'aura')
    ax.bar([], [], label = 'none')
    ax.scatter(np.array([]), np.array([]), color = 'a1')
    assert fig.to_png(72)[:4] == b'\x89PNG'


@pytest.mark.parametrize('name', NAMES)
def test_shapes_too_small_for_a_scanline_draw_without_failing(name):
    fig, axes = vc.subplots(2, 3, figsize = (6, 6), palette = name)
    axes = np.asarray(axes).ravel()
    axes[0].bar(['a', 'b', 'c'], [100.0, 1e-3, 0.0])
    axes[0].stacked_bar(['d'], [[100.0, 1e-3]])
    axes[1].scatter([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], s = [0.0, 1e-6, 30.0])
    axes[2].arrow(0.0, 0.0, 1e-4, 0.0)
    axes[2].arrow(0.0, 1.0, 1.0, 1.0)
    # Enough to be stamped in a raster output, as discs and from a table
    axes[3].scatter(np.arange(60.0), np.arange(60.0), s = 1e-6)
    axes[4].scatter(np.arange(60.0), np.arange(60.0), s = 1e-6, marker = '^')
    axes[5].scatter(np.arange(60.0), np.arange(60.0), s = 0.0, marker = '^')
    for dpi in (72, 100):
        assert fig.to_png(dpi)[:4] == b'\x89PNG'
    assert fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg()


def test_a_flat_outline_and_a_zero_area_point_draw_nothing():
    c = pdfwrite.Canvas(BytesIO(), pagesize = (10, 10))
    marks.polygon(c, [0.0, 1.0, 1.0, 0.0], [5.0, 5.0, 5.0, 5.0], (0.0, 0.0, 0.0))
    marks.points(c, get_palette('poster'), [1.0], [1.0], 0.0, (0.0, 0.0, 0.0))
    assert not c.ops


def test_dendrogram_names_its_leaves_and_keeps_its_scale():
    fig, ax = vc.subplots(palette = 'aura')
    ax.dendrogram([[5.0, 5.0, 15.0, 15.0]], [[0.0, 2.0, 2.0, 0.0]], 'left', 2, labels = ['p', 'q'], scale = True)
    fig.to_svg()
    assert ax.ylabels == ['p', 'q'] and list(ax.yticks) == [0.5, 1.5] and ax._yright and len(ax.xticks) > 1
    fig, ax = vc.subplots(palette = 'aura')
    ax.dendrogram([[5.0, 5.0, 15.0, 15.0]], [[0.0, 2.0, 2.0, 0.0]], 'top', 2, labels = ['p', 'q'])
    fig.to_svg()
    assert ax.xlabels == ['p', 'q'] and not len(ax.yticks)


def test_clustermap_takes_tick_labels_in_seaborns_forms():
    frame = type('Frame', (), {'columns': list('abc'), 'index': [f'r{i}' for i in range(40)],
                               '__array__': lambda self, *a, **k: np.arange(120.0).reshape(40, 3) ** 1.5})()
    fig, heat = vc.clustermap(frame, xticklabels = False, yticklabels = 2, figsize = (4, 4), palette = 'aura')
    assert heat._xtl is None and heat.show_all == (False, True)
    assert sum(bool(s) for s in heat._ytl) == 20
    fig, heat = vc.clustermap(frame, xticklabels = True, figsize = (4, 4), palette = 'aura')
    assert heat.show_all == (True, False) and sorted(heat._xtl) == ['a', 'b', 'c']
    fig, heat = vc.clustermap(np.arange(9.0).reshape(3, 3) ** 2, xticklabels = True, figsize = (4, 4), palette = 'aura')
    assert sorted(heat._xtl) == ['0', '1', '2'] and heat._ytl is None
    assert fig.to_png(72)[:4] == b'\x89PNG'


def test_long_titles_stay_on_the_page():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'aura')
    ax.plot([0, 1], [0, 1])
    ax.set_title('Medium title that just about fits' * 4)
    title = [o for o in fig._canvas().ops if o['kind'] == 'text' and o['text'].startswith('Medium')][0]
    assert title['size'] == 9.0 and fig.width > 288.0
    assert title['ctm'][4] + width(ax.title_text, 9.0, True) <= fig.width
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'aura')
    ax.plot([0, 1], [0, 1])
    ax.set_title('Short')
    assert [o for o in fig._canvas().ops if o['kind'] == 'text' and o['text'] == 'Short'][0]['size'] == 15.0
    assert fig.width == 288.0
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'aura')
    fig.suptitle('A figure title much wider than its figure')
    ax.plot([0, 1], [0, 1])
    fig._canvas()
    assert fig.width >= width('A figure title much wider than its figure', fig.pal.title_size, True)


def test_a_legend_inside_the_axes_keeps_a_halo():
    def strokes(loc):
        fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
        ax.plot([0, 1], [0, 1], label = 'line')
        ax.legend(loc)
        ops = fig._canvas().ops
        return sum(o['kind'] == 'path' and o['stroke'] and o['stroke_rgb'] == (1.0, 1.0, 1.0) for o in ops)
    assert strokes('lower right') > strokes('below')


def test_a_figure_shown_with_viewer_off_opens_no_window(monkeypatch):
    # A terminal with none of the silencing variables, where only viewer = False stands between the figure and a window
    for name in vc.figure.NO_VIEWER:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(sys, 'stdin', type('Tty', (), {'isatty': lambda self: True})())
    monkeypatch.setattr(vc.figure.os, 'startfile', lambda *a: pytest.fail('opened a window'), raising = False)
    monkeypatch.setattr(vc.figure.subprocess, 'Popen', lambda *a, **k: pytest.fail('opened a window'))
    monkeypatch.setattr(vc.figure.tempfile, 'NamedTemporaryFile', lambda *a, **k: pytest.fail('wrote a file to open'))
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'aura')
    ax.plot([0, 1], [0, 1])
    fig.show(72, viewer = False)
    vc.show_png(b'', viewer = False)


def test_show_renders_nothing_where_nothing_can_show_it(monkeypatch):
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'aura')
    ax.plot([0, 1], [0, 1])
    monkeypatch.setattr(vc.Figure, 'to_png', lambda *a, **k: pytest.fail('rendered for nobody'))
    assert fig.show() is None


def test_small_heatmap_cells_keep_their_color():
    fig, ax = vc.subplots(figsize = (1.5, 1.5), palette = 'aura')
    ax.heatmap(np.ones((40, 40)), cmap = 'sequential', vmin = 0, vmax = 1)
    img = pixels(fig, 144)
    x0, y0, w, h = ax.rect
    top, bottom = int((fig.height - y0 - h) * 2) + 2, int((fig.height - y0) * 2) - 2
    assert img[top:bottom, int(x0 * 2) + 2:int((x0 + w) * 2) - 2, :3].mean() < 200


def test_stamped_overlaps_show_no_crescents():
    # Discs that overlap within one batch were written all at once, so the last one's edge cut into the one below it
    rng = np.random.default_rng(0)
    fig, ax = vc.subplots(palette = 'aura')
    ax.scatter(rng.normal(size = 400), rng.normal(size = 400), s = 40)
    stamped = fig._canvas(stamp_dpi = 200.0)
    drawn = raster.render(stamped.ops, fig.width, fig.height, 200 / 72, 200 / 72, (1.0, 1.0, 1.0))
    old = vc.marks.STAMP_MIN
    vc.marks.STAMP_MIN = 10 ** 9
    try:
        traced = fig._canvas(stamp_dpi = 200.0)
    finally:
        vc.marks.STAMP_MIN = old
    vector = raster.render(traced.ops, fig.width, fig.height, 200 / 72, 200 / 72, (1.0, 1.0, 1.0))
    assert not (np.abs(drawn.astype(int) - vector.astype(int)).max(axis = 2) > 80).any()


def test_stamped_order_holds_across_sizes():
    # A later small disc lies over an earlier large one, whatever order the sizes are handled in
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.scatter(np.zeros(60), np.zeros(60), s = np.r_[np.full(59, 400.0), 30.0], color = 'a0', edge = ('white', 1.0))
    arr = pixels(fig, 144)
    assert (arr.min(axis = 2) > 200).any()


@pytest.mark.parametrize('marker', ['s', 'D', '^', 'v', 'X', 'P', '*'])
@pytest.mark.parametrize('name', NAMES)
def test_stamped_markers_match_vector(name, marker):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(size = 60), rng.uniform(size = 60)
    fig, ax = vc.subplots(figsize = (4, 4), palette = name)
    ax.scatter(x * 30, y * 30, s = 30, marker = marker)
    canvas = fig._canvas(stamp_dpi = 144.0)
    assert any(op['kind'] == 'image' and op.get('mark') for op in canvas.ops)
    stamped = raster.render(canvas.ops, fig.width, fig.height, 2.0, 2.0, (1.0, 1.0, 1.0))
    old = vc.marks.STAMP_MIN
    vc.marks.STAMP_MIN = 10 ** 9
    try:
        canvas = fig._canvas(stamp_dpi = 144.0)
    finally:
        vc.marks.STAMP_MIN = old
    assert not any(op['kind'] == 'image' for op in canvas.ops)
    vector = raster.render(canvas.ops, fig.width, fig.height, 2.0, 2.0, (1.0, 1.0, 1.0))
    # Ink in the same places, whichever way it was laid down
    assert np.mean(np.abs(stamped.astype(float) - vector.astype(float))) < 2.5


def test_many_sizes_stay_vector():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.scatter(np.arange(60.0), np.arange(60.0), s = np.arange(60.0) + 5.0, marker = '^')
    assert not any(op['kind'] == 'image' for op in fig._canvas(stamp_dpi = 144.0).ops)


def test_shadow_is_cropped_and_stays_on_the_grid():
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'aura')
    ax.bar(['a', 'b'], [1.0, 2.0])
    # For PDF and SVG, an image at no more than SHADOW_DPI, on that resolution's own pixels and smaller than the page
    op = fig._shadow_op(300.0)
    a, _, _, d, e, f = op['ctm']
    sc = SHADOW_DPI / 72.0
    assert op['w'] < fig.width * sc and op['h'] < fig.height * sc
    for value, size in ((a * sc, op['w']), (d * sc, op['h'])):
        assert abs(value - size) < 1e-6
    assert abs(e * sc - round(e * sc)) < 1e-6 and abs(f * sc - round(f * sc)) < 1e-6
    assert len(op['alpha']) == op['w'] * op['h'] and len(op['rgb']) == 3 * op['w'] * op['h']
    # For PNG, a layer on the output's own pixels, as much smaller than the page
    top, left, planes = fig._shadow_under(300.0)
    assert planes.shape[1] < fig.height * 300 / 72 and planes.shape[2] < fig.width * 300 / 72


def test_shadow_does_not_wrap_around_the_page():
    # A mark at the very bottom of the page casts its shadow off the page, not back in at the top
    fig = vc.Figure(4, 4, palette = 'aura')
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0)).off()
    ax.bar([0, 1], [1.0, 1.0], width = 1.0)
    ax.set_ylim(0.0, 3.0)
    arr = pixels(fig, 100)
    assert arr[:10].min() == 255


@pytest.mark.parametrize('build', [lambda ax: ax.plot([], []), lambda ax: ax.hist([]),
                                   lambda ax: ax.bar([1, 2], [0, 0]), lambda ax: ax.plot([1, np.nan], [np.nan, 1]),
                                   lambda ax: ax.plot([1, 2]).set_xlim(5, 5), lambda ax: ax.plot([1, 2]).set_ylim(0, 0),
                                   lambda ax: ax.plot([1, 10]).set_yscale('log').set_ylim(0, 10),
                                   lambda ax: ax.plot([1, 10]).set_xscale('log').set_xlim(-1, 0)])
def test_data_or_limits_without_a_range_still_draw(tmp_path, build):
    for ext in ('png', 'pdf', 'svg'):
        fig, ax = vc.subplots(figsize = (3, 2))
        build(ax)
        fig.save(tmp_path / f'a.{ext}', dpi = 50)
        for lim in (ax.xlim, ax.ylim):
            assert np.isfinite(lim).all() and lim[0] != lim[1]


def test_bars_of_no_height_stand_on_the_axis():
    fig, ax = vc.subplots()
    ax.bar([1, 2], [0, 0])
    fig._canvas()
    assert ax.ylim == (0.0, 1.0)


def test_log_limit_keeps_the_end_that_is_valid():
    fig, ax = vc.subplots()
    ax.plot([1, 10]).set_yscale('log').set_ylim(0, 100)
    fig._canvas()
    assert 0 < ax.ylim[0] < 1 and ax.ylim[1] == 100


def test_bad_input_fails_where_it_is_given():
    _, ax = vc.subplots()
    for call, message in ((lambda: ax.plot([1, 2, 3], [1, 2]), 'same length'),
                          (lambda: ax.scatter([1, 2, 3], [1, 2]), 'same length'),
                          (lambda: ax.set_xscale('symlog'), 'linear'), (lambda: ax.set_yscale('symlog'), 'linear'),
                          (lambda: ax.set_xlim(np.nan, 1), 'finite'), (lambda: ax.set_ylim(0, np.inf), 'finite'),
                          (lambda: ax.heatmap(np.zeros((0, 0))), 'one cell')):
        with pytest.raises(ValueError, match = message):
            call()


def test_short_hex_colors():
    pal = get_palette('aura')
    assert pal.color('#abc') == pal.color('#AABBCC')
    with pytest.raises(ValueError, match = 'Unknown color'):
        pal.color('bad')


def test_clustermap_needs_no_scipy():
    with patch.dict(sys.modules, {'scipy': None, 'scipy.cluster': None, 'scipy.cluster.hierarchy': None}):
        fig, heat = vc.clustermap(np.random.default_rng(2).normal(size = (6, 4)), palette = 'aura')
        assert sorted(fig.row_order) == list(range(6)) and fig.to_png(36.0)[:4] == b'\x89PNG'


def test_linkage_matches_scipy():
    pytest.importorskip('scipy')
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(4)
    m = rng.normal(size = (40, 6))
    # Canberra gives exactly 1 per coordinate of opposite signs, and so exact ties, which either order may take
    raw = rng.gamma(1.0, 1.0, (40, 6))
    for metric in vc.figure.METRICS:
        data = raw if metric in ('jensenshannon', 'canberra') else m
        for method in vc.figure.LINKAGES:
            if method in ('centroid', 'median', 'ward') and metric != 'euclidean':
                continue
            ours = vc.linkage(data, method, metric, optimal_ordering = False)
            theirs = linkage(pdist(data, metric), method)
            assert np.array_equal(ours[:, [0, 1, 3]], theirs[:, [0, 1, 3]]), (metric, method)
            assert np.allclose(ours[:, 2], theirs[:, 2], rtol = 1e-12), (metric, method)
    drawn, theirs = vc.figure._tree(vc.linkage(m)), dendrogram(vc.linkage(m), no_plot = True)
    assert drawn['leaves'] == theirs['leaves'] and drawn['icoord'] == theirs['icoord']
    assert np.allclose(drawn['dcoord'], theirs['dcoord'], rtol = 1e-15)
    rows = lambda u, v: np.abs(u - v).sum()
    assert np.array_equal(vc.linkage(m[:12], 'complete', rows, False), linkage(m[:12], 'complete', 'cityblock'))


def test_optimal_order_is_the_best_the_tree_allows():
    from itertools import product
    rng = np.random.default_rng(7)
    for trial in range(60):
        n, method = int(rng.integers(3, 10)), vc.figure.LINKAGES[trial % 7]
        m = rng.normal(size = (n, 3))
        d = np.sqrt(((m[:, None] - m[None]) ** 2).sum(-1))
        z = vc.linkage(m, method, optimal_ordering = False)
        def leaves(turn):
            turned = z.copy()
            turned[turn, :2] = turned[turn, 1::-1]
            return vc.figure._tree(turned)['leaves']
        cost = lambda order: d[order[:-1], order[1:]].sum()
        best = min(cost(leaves(np.array(t, bool))) for t in product([False, True], repeat = n - 1))
        ordered = vc.linkage(m, method)
        assert cost(vc.figure._tree(ordered)['leaves']) == pytest.approx(best, abs = 1e-12)
        assert np.array_equal(np.sort(ordered[:, :2], 1), z[:, :2]) and np.array_equal(ordered[:, 2:], z[:, 2:])


def test_a_long_chain_clusters_without_recursion():
    # Each point a little farther from the last, so single linkage makes a chain as deep as the data is long
    x = np.cumsum(np.linspace(1.0, 2.0, 1500))[:, None]
    fig, heat = vc.clustermap(np.hstack([x, x]), col_cluster = False, method = 'single', yticklabels = False)
    assert fig.row_order in (list(range(1500)), list(range(1499, -1, -1)))


def test_linkage_says_what_it_cannot_do():
    m = np.random.default_rng(1).normal(size = (5, 3))
    for kwargs, match in (({'method': 'nearest'}, 'Unknown method'), ({'metric': 'hamming'}, 'Unknown metric'),
                          ({'method': 'ward', 'metric': 'cityblock'}, 'euclidean')):
        with pytest.raises(ValueError, match = match):
            vc.linkage(m, **kwargs)
    with pytest.raises(ValueError, match = 'undefined'):
        vc.linkage(np.vstack([m, np.ones(3)]), metric = 'correlation')
    with pytest.raises(ValueError, match = 'negative'):
        vc.linkage(m, metric = 'jensenshannon')
    with pytest.raises(ValueError, match = 'finite'):
        vc.linkage(np.where(np.eye(5, 3) > 0, np.nan, m))


def test_subplots_returns_axes_as_matplotlib_does():
    assert isinstance(vc.subplots()[1], vc.Axes)
    for shape in ((1, 3), (3, 1)):
        fig, line = vc.subplots(*shape)
        assert line.shape == (3,) and list(line) == fig.axes and isinstance(line[2], vc.Axes)
    fig, grid = vc.subplots(2, 3)
    assert grid.shape == (2, 3) and grid[0, 1] is grid[0][1] is fig.axes[1]
    assert list(grid.flat) == list(grid.ravel()) == fig.axes
    _, (left, right) = vc.subplots(1, 2)
    left.plot([1, 2])
    right.plot([1, 2])


def test_marker_forms_in_pdf():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size = 500), rng.normal(size = 500)
    for name in NAMES:
        fig, ax = vc.subplots(palette = name)
        ax.scatter(x, y, s = 20)
        small = fig.to_pdf()
        assert small.startswith(b'%PDF') and b'/Subtype /Form' in small


def test_varied_sizes_in_pdf_are_paths():
    fig, ax = vc.subplots(palette = 'poster')
    ax.scatter(np.arange(100.0), np.arange(100.0), s = np.arange(100.0) + 5.0)
    assert b'/Subtype /Form' not in fig.to_pdf()


@pytest.mark.parametrize('name', NAMES)
def test_scalars_and_missing_values_draw_in_every_format(name):
    calls = [lambda ax: ax.bar(1, 2), lambda ax: ax.barh(1, 2), lambda ax: ax.stem(1, 2), lambda ax: ax.scatter(1, 2),
             lambda ax: ax.fill_between([], [], []), lambda ax: ax.bar([0, 1, 2], [1, np.nan, 2]),
             lambda ax: ax.barh([0, 1, 2], [1, np.inf, 2]), lambda ax: ax.fill_between([0, 1, 2, 3], [1, np.nan, 2, 3]),
             lambda ax: ax.stem([0, 1, 2], [1, np.nan, 2]), lambda ax: ax.errorbar([0, 1, 2], [1, np.nan, 2], .1),
             lambda ax: ax.hlines([0, np.nan], [0, 0], 1), lambda ax: ax.vlines([0, np.nan], [0, 0], 1),
             lambda ax: ax.ridgeline(np.arange(4.0), [[0, 1, np.nan, 0], [0, 0, 0, 0]], ['a', 'b']),
             lambda ax: ax.arrow(0, 0, np.nan, 1), lambda ax: ax.axhline(np.nan), lambda ax: ax.axvline(np.nan),
             lambda ax: ax.axvspan(np.nan, 1), lambda ax: ax.axhspan(np.nan, 1), lambda ax: ax.text(np.nan, 1, 'a'),
             lambda ax: ax.annotate('a', (1, np.nan)), lambda ax: ax.scatter([0, 1], [0, 1], c = [np.nan, np.nan]),
             lambda ax: ax.contour(np.arange(3.0), np.arange(3.0), np.full((3, 3), np.nan)),
             lambda ax: ax.heatmap(np.full((3, 3), np.nan), colorbar = True),
             lambda ax: ax.boxplot([[0.0, 10.0]], whis = 0), lambda ax: ax.scatter([0, 1], [0, 1], s = -1.0),
             lambda ax: ax.violinplot([[np.nan, 1.0], [], [np.inf, 1.0, 2.0]]), lambda ax: ax.stripplot([[np.nan]])]
    for call in calls:
        fig, ax = vc.subplots(figsize = (3, 2), palette = name)
        call(ax)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert fig.to_png(36.0)[:4] == b'\x89PNG' and fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg()


def test_a_nan_breaks_a_line_and_a_band():
    fig, ax = vc.subplots(palette = 'poster')
    ax.plot([0, 1, 2, 3, 4, 5], [1, 2, np.nan, 3, 2, 1])
    ax.fill_between([0, 1, 2, 3], [1, np.nan, 2, 3])
    c = fig._canvas()
    lines = [op for op in c.ops if op['kind'] == 'path' and op['stroke'] and op.get('mark')]
    assert sum(len([s for s in op['path'] if s[0] == 'm']) for op in lines) == 2
    bands = [op for op in c.ops if op['kind'] == 'path' and op['fill'] and op.get('mark')]
    assert len(bands) == 1


def test_ticks_on_tiny_and_crowded_ranges_stay_apart():
    assert nice_ticks(1e-12, 5e-12)[:2] == pytest.approx([1e-12, 2e-12], rel = 1e-6)
    assert nice_ticks(0, 0.3) == [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    labels = tick_labels([1234567.0, 1234568.0, 1234569.0])
    assert len(set(labels)) == 3
    assert fmt(999900.0) == '10^{6}' and fmt(-999900.0) == '-10^{6}'
    assert len(set(tick_labels([1000.001, 1000.002]))) == 2
    fig, ax = vc.subplots(figsize = (3, 2))
    ax.plot([0, 1, 2], [1e-12, 3e-12, 5e-12])
    ax.set_xticks([0.5, 0.5, 1.0], ['a', 'b', 'c'], 90)
    ax.set_yticks([1e-12, 1e-12])
    assert fig.to_png(36.0)[:4] == b'\x89PNG'


def test_log_axes_leave_out_what_they_cannot_show():
    fig, ax = vc.subplots()
    ax.bar([0, 1, 2], [1, 10, 100])
    ax.set_yscale('log')
    ax._limits()
    assert 0.5 < ax.ylim[0] < 1.0 and ax.yticks == [1.0, 10.0, 100.0]
    fig, ax = vc.subplots()
    ax.plot([0, 1, 2], [2, 3, 4])
    ax.set_yscale('log')
    ax._limits()
    assert ax.ylim[1] < 5.0 and len(ax.yticks) >= 2
    fig, ax = vc.subplots()
    ax.plot([0, 1], [-1, -2])
    ax.set_yscale('log')
    ax._limits()
    assert ax.ylim[0] < 1.0 and 10.0 < ax.ylim[1] < 20.0
    # A bar's gradient runs from the bottom of the axes, not from far below the page
    fig, ax = vc.subplots(palette = 'lumen')
    ax.bar([0], [100])
    ax.bar([1], [1])
    ax.set_yscale('log')
    ops = fig._canvas().ops
    assert all(y > 0 for op in ops if op['kind'] == 'path' and op.get('mark') for s in op['path'] for y in s[2::2])


def test_a_mapped_color_survives_style_and_size():
    rng = np.random.default_rng(0)
    x, y, z = rng.normal(size = 40), rng.normal(size = 40), rng.normal(size = 40)
    fig, ax = vc.subplots(palette = 'poster')
    ax.scatter(x, y, hue = z, size = rng.uniform(1, 9, 40))
    assert len({a['rgb'] for a in ax.artists}) > 5 and sum(len(a['x']) for a in ax.artists) == 40
    fig, ax = vc.subplots(palette = 'poster')
    ax.scatter(x, y, c = z, style = ['a', 'b'] * 20, legend = False)
    assert sum(len(a['x']) for a in ax.artists) == 40 and {a['marker'] for a in ax.artists} == {'o', 'X'}
    assert ax._legend_opts is None


def test_a_marker_holds_without_a_style():
    fig, ax = vc.subplots()
    ax.scatter([0, 1], [0, 1], hue = ['a', 'b'], marker = 's')
    assert {a['marker'] for a in ax.artists} == {'s'}


def test_hist_bars_take_the_width_of_their_bins():
    fig, ax = vc.subplots()
    ax.hist([0.5, 1.5, 3.0, 10.0], bins = [0, 1, 2, 5, 20])
    assert ax.artists[0]['w'] == pytest.approx(np.array([1, 1, 3, 15]) * 0.94)
    fig, ax = vc.subplots()
    ax.bar([0, 1, 3], [1, 2, 3], width = [0.5, 1, 2])
    assert fig.to_png(36.0)[:4] == b'\x89PNG'


def test_needle_tops_stay_clear_of_neighbors():
    def top_span(**kw):
        fig, ax = vc.subplots(palette = 'couture')
        ax.bar([0, 1, 2], [1, 2, 3], **kw)
        tops = [op for op in fig._canvas().ops if op['kind'] == 'path' and op['stroke'] and op.get('mark')]
        return (tops[1]['path'][1][1] - tops[1]['path'][0][1]) / float(ax._tx(1) - ax._tx(0))
    assert top_span() == pytest.approx(0.25 * 2.4)
    assert top_span(width = 1.0) == pytest.approx(1.0)


def test_negative_bars_keep_their_base_at_zero():
    fig, ax = vc.subplots(palette = 'lumen')
    ax.bar([0], [-1.0])
    ax.bar([1], [1.0])
    down, up = [op['fill_grad'][1] for op in fig._canvas().ops if op.get('fill_grad') and op.get('mark')]
    assert down[1] > down[3] and up[1] < up[3]


@pytest.mark.parametrize('cells', [(3, 3), (60, 60)])
def test_heatmap_follows_its_limits(cells):
    m = np.arange(float(cells[0] * cells[1])).reshape(cells)
    def drawn(change):
        fig, ax = vc.subplots(figsize = (3, 3), palette = 'couture')
        ax.heatmap(m, cmap = 'sequential')
        change(ax)
        ax.off()
        return pixels(fig), fig, ax
    plain, fig, ax = drawn(lambda ax: None)
    flipped, fig, ax = drawn(lambda ax: ax.invert_yaxis())
    x0, y0, w, h = ax.rect
    upper, lower = int(fig.height - y0 - h * 0.75), int(fig.height - y0 - h * 0.25)
    col = int(x0 + w / 2)
    assert plain[upper, col].sum() > plain[lower, col].sum()
    assert flipped[upper, col].sum() < flipped[lower, col].sum()


def test_annotated_heatmap_shows_the_cells_in_view():
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.heatmap(np.arange(9.0).reshape(3, 3), annot = True, fmt = '.0f')
    ax.set_xlim(1, 2)
    fig.layout()
    assert len([a for a in ax.artists if a['kind'] == 'text' and a['clip']]) == 9
    with patch('viscose.text.draw', wraps = vc.text.draw) as draw:
        fig._canvas()
    shown = [call.args[3] for call in draw.call_args_list]
    assert '1' in shown and '4' in shown and '7' in shown and '0' not in shown and '8' not in shown


def test_turned_labels_turn_back_when_there_is_room():
    fig, ax = vc.subplots(figsize = (2, 2))
    ax.bar(['a long category', 'another long one'], [1, 2])
    fig.layout()
    assert ax.xrot == 90.0 and ax._xrot == 0.0
    fig.set_size(12, 2).layout()
    assert ax.xrot == 0.0


def test_saving_twice_gives_the_same_file():
    for name in NAMES:
        fig, ax = vc.subplots(figsize = (3, 2), palette = name)
        ax.bar(['a long category', 'another long one', 'and a third'], [1, 2, 3], label = 'bars')
        ax.annotate('a very long annotation at the edge', (2, 3))
        ax.legend('below')
        # Gradients take their ids from a counter across the process
        first = re.sub(r'grad\d+', 'grad', fig.to_svg())
        fig.to_png(36.0)
        ax.get_xlim()
        assert re.sub(r'grad\d+', 'grad', fig.to_svg()) == first


def test_get_xlim_leaves_a_settled_page_alone():
    fig, ax = vc.subplots(figsize = (2, 2))
    ax.bar([str(i) * 6 for i in range(30)], np.arange(30.0))
    fig.layout()
    settled = list(ax.xticks)
    ax.get_xlim()
    assert ax.xticks == settled


def test_room_on_the_right_is_shared_in_order():
    rng = np.random.default_rng(0)
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'lumen')
    ax.scatter(rng.normal(size = 30), rng.normal(size = 30), hue = rng.normal(size = 30), size = rng.uniform(1, 9, 30))
    ax.colorbar('Score')
    fig.layout()
    x0, _, w, _ = ax.rect
    with patch('viscose.marks.polygon', wraps = marks.polygon) as bar:
        fig._canvas()
    legend_right = x0 + w + 16.0 + ax._legend_grid()[1][0]
    bars = [min(call.args[1]) for call in bar.call_args_list if len(call.args) > 5 and call.args[5] is not None]
    assert max(bars) >= legend_right and max(bars) < fig.width


def test_axis_names_clear_a_spine_without_tick_labels():
    fig, ax = vc.subplots(figsize = (3, 2), palette = 'couture')
    ax.plot([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.layout()
    with patch('viscose.text.draw', wraps = vc.text.draw) as draw:
        fig._canvas()
    spots = {call.args[3]: (call.args[1], call.args[2]) for call in draw.call_args_list}
    x0, y0, _, _ = ax.rect
    off = fig.pal.spine_offset
    assert spots['X'][1] < y0 - off and spots['X'][1] > 8 and spots['Y'][0] < x0 - off and spots['Y'][0] > 6


def test_several_lines_take_their_room():
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.plot([0, 1])
    margins = ax._limits() or ax._margins()
    ax.set_xlabel('one\ntwo')
    ax.set_ylabel('one\ntwo')
    ax.set_title('one\ntwo')
    grown = ax._margins()
    assert grown[0] > margins[0] and grown[2] > margins[2] and grown[3] > margins[3]
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.bar(['a\nb', 'c\nd'], [1, 2])
    ax._limits()
    assert ax._xtick_height() > 2 * ax._tick_sizes()[0]
    # A block of lines sits where va says, as a whole
    with patch('viscose.text.draw', wraps = vc.text.draw) as draw:
        canvas = pdfwrite.Canvas(BytesIO(), pagesize = (100, 100))
        vc.text.draw(canvas, 50, 50, 'a\nb\nc', 10, (0, 0, 0), va = 'center')
    ys = [call.args[2] for call in draw.call_args_list if call.args[3] in 'abc']
    assert ys[1] == pytest.approx(50) and ys[0] > 50 > ys[2]


def test_long_axis_names_shrink_then_grow_the_page():
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.plot([0, 1])
    ax.set_xlabel('A name that is a little too long for this small figure, as they can be')
    fig.layout()
    with patch('viscose.text.draw', wraps = vc.text.draw) as draw:
        fig._canvas()
    size = next(call.args[4] for call in draw.call_args_list if call.args[3].startswith('A name'))
    assert size < fig.pal.label_size
    assert width(ax.xlabel, size) <= fig.width
    fig, ax = vc.subplots(figsize = (2, 2))
    ax.plot([0, 1])
    ax.set_ylabel('A name that would not fit this figure even at the smallest size that can be read')
    fig.layout()
    assert fig.height > 2 * 72.0


def test_mistakes_fail_at_the_call_with_a_clear_message():
    fig, ax = vc.subplots()
    with pytest.raises(ValueError, match = 'marker'):
        ax.scatter([0], [0], marker = 'x')
    with pytest.raises(ValueError, match = 'marker'):
        ax.plot([0, 1], marker = 'x')
    with pytest.raises(ValueError, match = 'marker'):
        ax.scatter([0], [0], style = ['a'], markers = ['x'])
    with pytest.raises(ValueError, match = 'legend'):
        ax.legend('center')
    with pytest.raises(ValueError, match = 'colorbar'):
        ax.colorbar(loc = 'left')
    with pytest.raises(ValueError, match = 'one label each'):
        ax.set_xticks([0, 1], ['a'])
    with pytest.raises(ValueError, match = 'rows and columns'):
        ax.heatmap([1.0, 2.0])
    with pytest.raises(ValueError, match = 'one label each'):
        ax.heatmap(np.eye(2), xticklabels = ['a'])
    with pytest.raises(ValueError, match = 'same length'):
        ax.stem([0, 1], [1])
    with pytest.raises(ValueError, match = 'Unknown color'):
        fig.pal.color('a9')
    with pytest.raises(ValueError, match = 'bar'):
        vc.Palette('typo', ['#000000'] * 4, ['#000000'] * 4, '#000000', '#000000', '#000000', bar = 'neddle')
    with pytest.raises(ValueError, match = 'four shades'):
        vc.Palette('short', ['#000000'] * 3, ['#000000'] * 4, '#000000', '#000000', '#000000')
    fig, ax = vc.subplots()
    ax.plot([0, 1])
    ax.colorbar()
    with pytest.raises(ValueError, match = 'color scale'):
        fig.to_svg()


def test_color_none_draws_nothing():
    for name in NAMES:
        fig, ax = vc.subplots(figsize = (3, 2), palette = name)
        ax.bar([0, 1], [1, 2], color = ['a1', 'none'], label = 'kept')
        ax.scatter([0, 1], [0, 1], color = 'none', label = 'gone')
        ax.vlines([0, 1], 0, 1, color = ['none', 'a1'])
        ax.legend()
        assert fig.to_png(36.0)[:4] == b'\x89PNG' and [s[0] for s in ax.legend_items] == ['kept']


def test_arrays_of_objects_are_numbers():
    fig, ax = vc.subplots()
    ax.plot(np.array([1, 2, 3], dtype = object), np.array([1.5, 2.5, 3.5], dtype = object))
    ax._limits()
    assert not ax._cats['x'] and not ax._cats['y'] and ax.artists[0]['x'].dtype == float


def test_image_takes_gray_and_floats():
    fig, ax = vc.subplots(figsize = (2, 2))
    ax.plot([0, 1], [0, 1])
    ax.image(np.array([[0.0, 1.0], [1.0, 0.0]]), 0.5, 0.5, 20, 20)
    ax.image(np.array([[0, 255], [255, 0]], dtype = np.uint8), 0.5, 0.5, 20, 20)
    assert ax.artists[-1]['rgb'] == ax.artists[-2]['rgb'] == bytes([0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0])


def test_asymmetric_errors_and_horizontal_spans():
    fig, ax = vc.subplots()
    ax.errorbar([0, 1], [1, 2], [[0.1, 0.2], [0.3, 0.4]])
    ax.axhspan(5, 6, label = 'band')
    ax._limits()
    assert ax.artists[0]['lo'] == pytest.approx([0.9, 1.8]) and ax.artists[0]['hi'] == pytest.approx([1.3, 2.4])
    assert ax.ylim[1] > 6.0 and ax.legend_items[0][0] == 'band'


def test_clustermap_of_little_or_missing_data():
    for shape in ((1, 4), (4, 1), (1, 1)):
        fig, heat = vc.clustermap(np.arange(float(np.prod(shape))).reshape(shape), palette = 'couture')
        assert fig.to_png(36.0)[:4] == b'\x89PNG'
    gappy = np.eye(3)
    gappy[0, 1] = np.nan
    with pytest.raises(ValueError, match = 'finite'):
        vc.clustermap(gappy)
    fig, heat = vc.clustermap(gappy, row_cluster = False, col_cluster = False)
    assert fig.to_png(36.0)[:4] == b'\x89PNG'


def test_any_background_color():
    fig, ax = vc.subplots(figsize = (1, 1))
    assert fig.to_rgba(72.0, background = '#000000')[0, 0, :3].tolist() == [0, 0, 0]
    assert fig.to_rgba(72.0, background = (1.0, 1.0, 1.0))[0, 0, :3].tolist() == [255, 255, 255]
    assert fig.to_rgba(72.0, background = None)[0, 0, 3] == 0
    assert fig.pal.groups(2, 0) == fig.pal.groups(2)


def test_a_page_of_any_color():
    dark = vc.Palette('night', ['#F7C6B8', '#EC5B3F', '#B53D25', '#7A2A1A'],
                      ['#C2CFF7', '#3E63DD', '#2F4CB0', '#223576'], '#E8EAF0', '#8C92A3', '#5C6273', point = 'cut',
                      ridge = 'deep', paper = '#171A21')
    assert dark.css('paper') == '#171A21' and vc.palette('aura').css('paper') == '#FFFFFF'
    assert dark.on('#FFFFFF') == 'black' and dark.on('#000000') == 'white' and vc.palette('aura').on('#FFFFFF') == 'ink'
    assert np.allclose(dark.diverging(0.5), dark.color('paper'))
    rng = np.random.default_rng(0)
    fig, axes = vc.subplots(1, 3, figsize = (8, 3), palette = dark)
    axes[0].scatter(rng.normal(size = 60), rng.normal(size = 60), s = 30, label = 'points')
    axes[0].annotate('a name', (0.0, 0.0))
    axes[0].legend('upper right')
    axes[1].ridgeline(np.linspace(0, 1, 40), [np.sin(np.linspace(0, 3, 40))] * 3)
    axes[2].heatmap(rng.normal(size = (4, 4)), annot = True)
    px = fig.to_rgba(72.0)
    assert px[0, 0, :3].tolist() == [0x17, 0x1A, 0x21]
    # A white ring, ridge, or halo would be the brightest thing on such a page
    assert not (px[..., :3] > 250).all(axis = 2).any()
    assert fig.to_rgba(72.0, background = None)[0, 0, 3] == 0
    assert fig.to_png(72.0)[:4] == b'\x89PNG'
    assert 'fill="#171A21"' in fig.to_svg()
    # The page's color lies under everything in a PDF, whose page is otherwise white
    with patch.object(marks, 'polygon', wraps = marks.polygon) as drawn:
        fig.to_pdf()
    assert any(isinstance(c.args[1], list) and c.args[1] == [0.0, fig.width, fig.width, 0.0]
               and c.args[3] == dark.color('paper') for c in drawn.call_args_list)
    # A soft shadow over a dark page shows as a glow, and fails in no format
    glow = vc.Palette('glow', dark.a, dark.b, dark.ink, dark.muted, dark.faint, shadow = 0.5, paper = dark.paper)
    fig, ax = vc.subplots(figsize = (2, 2), palette = glow)
    ax.bar([0, 1], [1, 2])
    assert fig.to_png(72.0)[:4] == b'\x89PNG' and fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg()


@pytest.mark.parametrize('name', NAMES)
def test_every_palette_has_a_dark_version(name):
    from viscose.palettes import DARK_L, _oklch
    light = vc.get_palette(name)
    dark = light.dark()
    assert dark.is_dark and not light.is_dark and dark.name == name + '-dark' and light.paper == '#FFFFFF'
    assert dark.dark() is dark and light.dark() is dark and vc.get_palette(name + '-dark') is dark
    lum = lambda c: float(np.dot(dark.color(c), (0.2126, 0.7152, 0.0722)))
    # Shades run from the strongest against the page to the faintest, all clear of it, and the text is lighter still
    for k in 'ab':
        shades = [lum(f'{k}{i}') for i in range(4)]
        assert shades == sorted(shades, reverse = True) and shades[3] > 4 * lum('paper')
        assert all(_oklch(dark.color(f'{k}{i}'))[0] > DARK_L[0] - 0.01 for i in range(4))
    assert lum('ink') > lum('muted') > lum('faint') > lum('paper') and dark.on('ink') == 'black'
    assert np.allclose(dark.diverging(0.5), dark.color('paper'))
    # The page is a near-black in the main hue, or the color given, such as an app's own surface
    assert light.dark('#1E1E1E').css('paper') == '#1E1E1E' and lum('paper') < 0.1
    # No orange or gold darkens into olive or brown: warm shades keep a red share well above their blue one
    for c in dark.b if name != 'aura' else ():
        r, g, b = dark.color(c)
        assert r > g > b and r - b > 0.3
    fig, ax = vc.subplots(figsize = (3, 2), palette = dark)
    ax.bar([0, 1, 2], [1, 2, 3], color = ['a1', 'b1', 'a3'])
    ax.scatter([0, 1, 2], [2, 1, 3], s = 60)
    ax.plot([0, 1, 2], [1, 3, 2], label = 'line')
    ax.legend()
    px = fig.to_rgba(72.0)
    assert px[0, 0, :3].tolist() == [int(dark.css('paper')[i:i + 2], 16) for i in (1, 3, 5)]
    assert fig.to_pdf()[:4] == b'%PDF' and f'fill="{dark.paper}"' in fig.to_svg()


def test_dark_settings_replace_what_is_derived():
    pal = vc.Palette('mine', ['#14213D', '#27477E', '#5C85C5', '#C4D5EE'], ['#8F5B00', '#D08C00', '#F2B83B', '#FBE5B0'],
                     '#14213D', '#5A6478', '#E6EAF1',
                     dark_settings = {'b': ['#FFE2A0', '#F2B83B', '#C98A1A', '#8F6010'], 'title_color': '#F2B83B'})
    dark = pal.dark()
    assert dark.b == ['#FFE2A0', '#F2B83B', '#C98A1A', '#8F6010'] and dark.title_color == '#F2B83B'
    assert dark.a != pal.a and dark.dark_settings == {} and dark.is_dark
    # An already dark palette stays as it is
    night = vc.Palette('night', dark.a, dark.b, dark.ink, dark.muted, dark.faint, paper = '#101010')
    assert night.dark() is night


def test_dark_mode_turns_every_palette_dark():
    module = sys.modules['viscose.palettes']
    before, library, mode = list(module._default), list(module._library), list(module._dark)
    try:
        module._default[:] = ['lumen', False]
        vc.default_palette('aura')
        vc.dark_mode(True)
        assert vc.subplots()[0].pal.name == 'aura-dark' and vc.subplots(palette = 'poster')[0].pal.name == 'poster-dark'
        # A user's choice made in dark mode is the light palette's, so that it stays when the mode ends
        vc.set_palette('couture')
        assert module._default[0] == 'couture' and vc.subplots()[0].pal.name == 'couture-dark'
        assert vc.get_palette(None, vc.get_palette('lumen')).name == 'couture-dark'
        assert vc.dark_mode(True, paper = '#202124') == (True, None)
        assert vc.subplots()[0].pal.paper == '#202124'
        previous = vc.dark_mode(False)
        assert previous == (True, '#202124') and not vc.subplots()[0].pal.is_dark
        vc.dark_mode(*previous)
        assert vc.subplots()[0].pal.paper == '#202124'
        # A library's own palette, passed as an instance, turns dark too
        vc.set_palette(None)
        assert vc.get_palette(None, module.LUMEN).name == 'lumen-dark'
        vc.dark_mode(False)
        assert vc.subplots()[0].pal.name == 'aura' and vc.subplots(palette = 'lumen-dark')[0].pal.is_dark
        with pytest.raises(ValueError, match = 'each also with -dark'):
            vc.get_palette('dusk-dark')
    finally:
        module._default[:], module._library[:], module._dark[:] = before, library, mode


def test_a_figure_in_dark_mode_equals_one_drawn_in_the_dark_palette():
    module = sys.modules['viscose.palettes']
    mode = list(module._dark)
    try:
        vc.dark_mode(True)
        fig = all_marks('lumen')
        vc.dark_mode(False)
        assert fig.to_png(72.0) == all_marks('lumen-dark').to_png(72.0)
        assert fig.in_palette('aura-dark').to_png(72.0) == all_marks('aura-dark').to_png(72.0)
    finally:
        module._dark[:] = mode


def test_a_dark_sphere_is_lit_from_above():
    pal = vc.get_palette('lumen').dark()
    rim, base, spot = (layer[3] for layer in marks._layers(pal, np.zeros(1), np.zeros(1), np.full(1, 5.0),
                                                           pal.color('a1'), None))
    lum = lambda c: float(np.dot(c, (0.2126, 0.7152, 0.0722)))
    assert lum(rim) < lum(base) < lum(spot)


@pytest.mark.parametrize('n, dpi', [(10, 72.0), (200, 72.0), (200, None)])
def test_hollow_rings(n, dpi):
    rng = np.random.default_rng(0)
    for name in NAMES:
        fig, ax = vc.subplots(figsize = (2, 2), palette = name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter(rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n), s = 400 if n == 10 else 40, color = 'none',
                   edge = ('#FF0000', 1.5), label = 'none in the legend')
        ax.scatter([0.5], [0.5], s = 900, color = 'none', edge = ('#0000FF', 1.0), z = 9)
        assert not ax.legend_items
        px = fig.to_rgba(144.0 if dpi else 72.0)
        r, g, b = (px[..., k].astype(int) for k in range(3))
        assert ((r > 200) & (g < 80) & (b < 80)).any() and ((b > 200) & (g < 80) & (r < 80)).any()
        assert fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg()
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'poster')
    ax.scatter([0, 1], [0, 1], s = 900, color = 'none', edge = ('#0000FF', 1.0))
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    fig.layout()
    x, y = ax.page(0.0, 0.0)
    assert fig.to_rgba(72.0)[int(fig.height - y), int(x), :3].tolist() == [255, 255, 255]
    # Color 'none' without an edge still draws nothing
    fig, ax = vc.subplots(figsize = (2, 2), palette = 'poster')
    ax.scatter([0, 1], [0, 1], s = 900, color = 'none')
    assert (fig.to_rgba(72.0)[..., :3] > 200).mean() > 0.9


@pytest.mark.parametrize('scale', [1e-4, 1.0, 1e6])
def test_a_histogram_fills_its_axis_at_any_scale(scale):
    rng = np.random.default_rng(0)
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.hist(rng.uniform(0.0, 1.0, 500) * scale, bins = 20)
    lo, hi = ax.get_xlim()
    assert -0.1 * scale < lo < 0.0 and 1.0 * scale < hi < 1.1 * scale


def test_script_characters_are_runs():
    assert runs('log₂ CPM') == runs('log_{2} CPM') == [('log', 0.0, 1.0), ('2', -0.22, 0.7), (' CPM', 0.0, 1.0)]
    assert runs('10⁻³') == [('10', 0.0, 1.0), ('-', 0.38, 0.7), ('3', 0.38, 0.7)]
    assert width('log₂ CPM', 10.0) == width('log_{2} CPM', 10.0)
    # Which the fonts have no glyphs for, so without this they would draw as boxes
    fig, ax = vc.subplots(figsize = (2, 2))
    ax.set_ylabel('log₂ CPM')
    ax.plot([0, 1], [0, 1])
    fig.layout()
    assert fig.to_rgba(72.0)[0, 0, 3] == 255


def test_a_page_held_to_a_window_never_grows():
    def busy(fig, ax):
        ax.bar(['a very long category name indeed', 'another very long category name'], [1, 2])
        ax.set_ylabel('a very long axis name for a page this small')
        ax.annotate('a very long annotation that fits on neither side of its point', (0.5, 1.5))
        fig.suptitle('A title that is far wider than the page it sits on')
    fig, ax = vc.subplots(figsize = (2, 1.2))
    busy(fig, ax)
    fig.layout()
    assert fig.width > 2 * 72.0
    fig.set_size(2, 1.2).layout()
    assert (fig.width, fig.height) == (2 * 72.0, 1.2 * 72.0)
    assert fig.to_rgba(100.0).shape[:2] == (round(1.2 * 100.0), round(2 * 100.0))
    # Saving it twice stays the same, and so does a later size
    assert fig.to_png(72.0) == fig.to_png(72.0)
    fig.set_size(3, 2).layout()
    assert (fig.width, fig.height) == (3 * 72.0, 2 * 72.0)


def _paper_page(palette):
    """A journal page with every piece of the paper layer."""
    fig = vc.Figure(183, 90, palette = palette, unit = 'mm')
    x = np.linspace(0, 1, 12)
    a = fig.add_axes((16, 12, 60, 50), unit = 'mm')
    for k in range(4):
        a.plot(x, 0.5 * x ** (1 + k / 4))
        a.annotate(f'line {k}', (1, 0.5), offset = (4, 0), spread = True)
    a.set_yscale('sqrt')
    a.set_yticks([0, 0.05, 0.1, 0.3, 0.5], fmt = '{:.0%}')
    b = fig.add_axes((100, 12, 60, 50), unit = 'mm')
    b.step(x, x ** 2, where = 'post')
    b.tick_params(axis = 'y', length = 0, labels = False)
    b.set_spines(left = False)
    b.set_xscale('function', functions = (np.sqrt, np.square))
    fig.letter(2, 4, 'a', 'Heading')
    fig.letter(86, 4, 'b')
    fig.text(0.98, 0.02, 'note', ha = 'right')
    return fig


def _texts(c):
    return [o['text'] for o in c.ops if o['kind'] == 'text']


@pytest.mark.parametrize('name', NAMES)
def test_a_page_in_millimeters_keeps_its_size_and_places_axes_from_the_top_left(name):
    fig = _paper_page(name)
    c = fig._canvas()
    mm = 72.0 / 25.4
    assert fig.width == pytest.approx(183 * mm) and fig.height == pytest.approx(90 * mm)
    assert fig.axes[0].rect == pytest.approx((16 * mm, (90 - 62) * mm, 60 * mm, 50 * mm))
    heading = 'HEADING' if fig.pal.caps else 'Heading'
    assert {'a', heading, 'b', 'note', '0%', '50%'} <= set(_texts(c))
    assert fig.to_png(72.0)[:4] == b'\x89PNG' and fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg()


def test_a_paper_page_in_another_palette_equals_one_drawn_in_it():
    assert _paper_page('poster').in_palette('couture').to_png(72.0) == _paper_page('couture').to_png(72.0)
    assert _paper_page('lumen').in_palette('aura').to_pdf() == _paper_page('aura').to_pdf()


def test_a_letter_heading_sits_on_the_letter_baseline():
    fig = vc.Figure(100, 50, palette = 'couture', unit = 'mm')
    fig.letter(5, 5, 'a', 'Heading')
    letter, heading = [o for o in fig._canvas().ops if o['kind'] == 'text']
    baseline = lambda o: o['ctm'][5] + o['y']
    assert baseline(letter) == pytest.approx(baseline(heading)) and heading['ctm'][4] > letter['ctm'][4]
    with pytest.raises(ValueError, match = 'unit'):
        fig.text(0, 0, 'x', unit = 'cm')


def test_sqrt_and_function_scales_space_by_their_map():
    for scale, functions in (('sqrt', None), ('function', (np.sqrt, np.square))):
        fig, ax = vc.subplots(figsize = (4, 4))
        ax.plot([0, 1], [0, 1])
        ax.set_xlim(0, 1)
        ax.set_xscale(scale, functions)
        fig.layout()
        x0, _, w, _ = ax.rect
        assert float(ax._tx(0.25)) == pytest.approx(x0 + w / 2)
        assert float(ax.data(*ax.page(0.36, 0.5))[0]) == pytest.approx(0.36)
    with pytest.raises(ValueError, match = 'scale'):
        ax.set_yscale('symlog')
    with pytest.raises(ValueError, match = 'functions'):
        ax.set_yscale('sqrt', (np.sqrt, np.square))
    with pytest.raises(ValueError, match = 'functions'):
        ax.set_yscale('function')


def test_a_sqrt_axis_keeps_zero_and_below_in_order():
    fig, ax = vc.subplots(figsize = (4, 4))
    ax.scatter([-0.04, 0, 0.01, 0.6], [0, 1, 2, 3])
    ax.set_xscale('sqrt')
    fig.layout()
    assert np.all(np.diff(ax._tx([-0.04, 0, 0.01, 0.6])) > 0) and ax.xlim[0] < -0.04


def test_tick_formats_write_automatic_and_given_ticks():
    fig, ax = vc.subplots(figsize = (4, 4))
    ax.plot([0, 1], [0, 2])
    ax.set_xticks(fmt = '{:.0%}')
    ax.set_yticks([0, 1, 2], fmt = lambda v: f'{v:g} mM')
    fig.layout()
    assert ax.xlabels[0] == '0%' and ax.xlabels[-1] == '100%' and ax.ylabels == ['0 mM', '1 mM', '2 mM']
    with pytest.raises(ValueError, match = 'need the ticks'):
        ax.set_xticks(labels = ['a'])


@pytest.mark.parametrize('name', NAMES)
def test_tick_params_set_length_pad_size_and_labels_per_axis(name):
    def margins(**kw):
        fig, ax = vc.subplots(figsize = (4, 4), palette = name)
        ax.plot([0, 1], [0, 1])
        ax.tick_params(**kw)
        fig.layout()
        return ax, ax._margins()
    _, plain = margins()
    ax, hidden = margins(axis = 'y', labels = False)
    assert hidden[0] < plain[0] and hidden[2] == plain[2] and ax.ylabels == [] and len(ax.yticks)
    assert not set(_texts(ax.figure._canvas())) & {'0.2', '0.4', '0.6', '0.8'} - set(ax.xlabels)
    _, padded = margins(axis = 'x', pad = 20)
    assert padded[2] == pytest.approx(plain[2] + 15) and padded[0] == plain[0]
    ax, small = margins(labelsize = 5)
    assert ax._tick_sizes() == (5, 5) and small[0] < plain[0]
    with pytest.raises(ValueError, match = 'axis'):
        ax.tick_params(axis = 'z')


def test_tick_length_zero_drops_the_ticks_from_the_page():
    def ticks(**kw):
        fig, ax = vc.subplots(figsize = (4, 4), palette = 'couture')
        ax.plot([0, 1], [0, 1])
        ax.tick_params(**kw)
        return len(fig._canvas().ops), ax._margins()[0]
    (with_ticks, left), (without, tight) = ticks(), ticks(axis = 'y', length = 0)
    assert without < with_ticks and tight == pytest.approx(left - get_palette('couture').tick_length)


def test_set_spines_overrides_the_palette():
    def spines(name, **kw):
        fig, ax = vc.subplots(figsize = (4, 4), palette = name)
        ax.bar([1, 2], [1, 2])
        ax.set_spines(**kw)
        return ax._spines()
    assert spines('aura') == () and spines('aura', left = True) == ('left', )
    assert spines('couture', left = False) == ('bottom', ) and spines('couture') == ('left', 'bottom')


def test_steps_turn_at_their_points():
    for where, xs, ys in (('pre', [0, 0, 1, 1, 2], [1, 2, 2, 3, 3]), ('post', [0, 1, 1, 2, 2], [1, 1, 2, 2, 3]),
                          ('mid', [0, 0.5, 0.5, 1.5, 1.5, 2], [1, 1, 2, 2, 3, 3])):
        fig, ax = vc.subplots(figsize = (3, 3))
        ax.step([0, 1, 2], [1, 2, 3], where = where, label = where)
        line = ax.artists[0]
        assert line['x'].tolist() == xs and line['y'].tolist() == ys and ax.legend_items[0][0] == where
    with pytest.raises(ValueError, match = 'where'):
        ax.step([0, 1], [0, 1], where = 'after')


def test_spread_labels_part_as_little_as_they_must():
    fig, ax = vc.subplots(figsize = (5, 5), palette = 'poster')
    ax.plot([0, 10], [0, 10])
    for s, y in (('one', 5.0), ('two', 5.0), ('three', 5.05)):
        ax.annotate(s, (5, y), offset = (6, 0), size = 8, spread = True)
    # A label in a column of its own, or on the other side, stays where it belongs
    ax.annotate('apart', (0.2, 5.0), offset = (6, 0), size = 8, spread = True)
    ax.annotate('still', (5, 5.0), offset = (-6, 0), size = 8, ha = 'right', spread = True)
    c = fig._canvas()
    y = {o['text']: o['ctm'][5] for o in c.ops if o['kind'] == 'text'}
    column = [y[s] for s in ('one', 'two', 'three')]
    assert np.diff(sorted(column)) == pytest.approx([1.25 * 8] * 2)
    # Least squares under the gap puts the column's middle where its points' middle is
    want = [float(ax._ty(v)) for v in (5.0, 5.0, 5.05)]
    assert np.mean(column) == pytest.approx(np.mean(want))
    assert y['apart'] == pytest.approx(want[0]) and y['still'] == pytest.approx(want[0])
    # Only moved labels get a leader
    leaders = [o for o in c.ops if o['kind'] == 'path' and o.get('stroke') and o['line_width'] == 0.5]
    assert len(leaders) == 2


def _journal(name = 'journal', **settings):
    """A palette for small type, from couture's colors."""
    pal = get_palette('couture')
    base = dict(point = 'flat', bar = 'flat', spines = ('left', 'bottom'), tick_size = 5.8, label_size = 6.5,
                title_size = 7.0, spacing = 0.55)
    base.update(settings)
    return vc.Palette(name, pal.a, pal.b, pal.ink, pal.muted, pal.faint, **base)


def test_spacing_scales_the_gaps_that_the_plot_code_sets():
    def margins(spacing):
        fig, ax = vc.subplots(figsize = (4, 4), palette = _journal(spacing = spacing))
        ax.plot([0, 1], [0, 1], label = 'line')
        ax.set_title('Title', size = 7)
        ax.legend('below')
        fig.layout()
        return np.array(ax._margins()), ax._legend_grid()[1][1]
    (wide, row), (tight, short) = margins(1.0), margins(0.5)
    assert np.all(tight < wide) and short == pytest.approx(row / 2)
    fig, ax = vc.subplots(figsize = (4, 4), palette = _journal(spacing = 0.5))
    ax.annotate('note', (0.5, 0.5))
    assert ax.artists[0]['size'] == pytest.approx(4.3) and ax.artists[0]['halo'] == pytest.approx(1.3)


def test_a_palette_font_is_used_where_the_machine_has_it_and_falls_back_where_not():
    from viscose import text
    missing = _journal('nofont', font = ('NoSuchFamily', 'no_such_font.ttf', 'no_such_font_bold.ttf'))
    fig, ax = vc.subplots(figsize = (3, 3), palette = missing)
    ax.set_title('Falls back', size = 7)
    fonts = {o['font'] for o in fig._canvas().ops if o['kind'] == 'text'}
    assert fonts and fonts <= set(text.fonts())
    with pytest.raises(ValueError, match = 'font'):
        _journal('bad', font = ('Arial', ))


def test_a_palette_brings_its_own_scales():
    pal = _journal(sequential_stops = ('#ffffff', '#ff0000', '#000000'),
                   diverging_stops = ('#0000ff', '#ffffff', '#ff0000'))
    assert tuple(pal.sequential(0.5)) == pytest.approx((1.0, 0.0, 0.0))
    assert tuple(pal.diverging(0.0)) == pytest.approx((0.0, 0.0, 1.0))
    assert tuple(get_palette('aura').sequential(1.0)) == pytest.approx(get_palette('aura').color('a0'))
    with pytest.raises(ValueError, match = 'stops'):
        _journal('one', sequential_stops = ('#ffffff', ))


@pytest.mark.parametrize('marker', ['h', 'p'])
def test_hexagons_and_pentagons_cover_about_a_circle(marker):
    from viscose.marks import MARKERS
    xs, ys = np.array(MARKERS[marker]).T
    area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    assert area == pytest.approx(np.pi, rel = 0.03)
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.scatter(np.arange(60.0), np.arange(60.0), marker = marker)
    assert fig.to_png(72)[:4] == b'\x89PNG' and fig.to_pdf()[:4] == b'%PDF'


def test_axline_crosses_the_axes_and_leaves_the_limits_alone():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'couture')
    ax.scatter([1, 2], [1, 3])
    before = ax.get_xlim(), ax.get_ylim()
    ax.axline((0, 0), slope = 1, label = 'identity')
    assert (ax.get_xlim(), ax.get_ylim()) == before and ax.legend_items[-1][0] == 'identity'
    c = fig._canvas()
    line = [o for o in c.ops if o.get('stroke') and o.get('dash')][-1]
    (_, x0, y0), (_, x1, y1) = line['path'][0], line['path'][1]
    assert (y1 - y0) / (x1 - x0) > 0 and line['clip'] is not None
    with pytest.raises(ValueError, match = 'slope'):
        ax.axline((0, 0))


def test_rectangles_fill_and_outline_in_data_or_axes_fractions():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.rectangle(0, 0, 2, 1, color = 'a2', radius = 3)
    ax.rectangle(0.1, 0.1, 0.5, 0.5, color = 'none', edge = ('ink', 0.5), axes_fraction = True)
    assert ax.get_xlim()[1] > 2
    ops = [o for o in fig._canvas().ops if o['kind'] == 'path' and not o.get('mark')]
    filled = [o for o in ops if o['fill'] and o['fill_rgb'] == fig.pal.color('a2')]
    outlined = [o for o in ops if o['stroke'] and not o['fill'] and o['line_width'] == 0.5]
    assert filled and any(s[0] == 'c' for s in filled[0]['path']) and outlined


def test_arrows_shrink_take_a_head_length_and_bend():
    def shaft(**kw):
        fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.arrow(1, 5, 9, 5, **kw)
        ops = [o for o in fig._canvas().ops if o.get('mark')]
        return ops[0]['path'], ops[-1]['path']
    (line, _), (short, _) = shaft(), shaft(shrink = (10, 10))
    assert short[0][1] == pytest.approx(line[0][1] + 10)
    _, head = shaft(head = 3)
    assert max(s[1] for s in head if len(s) > 1) - min(s[1] for s in head if len(s) > 1) == pytest.approx(3)
    bent, _ = shaft(bend = 0.2)
    assert len(bent) > 3 and max(s[2] for s in bent if len(s) > 1) > line[0][2] + 1


def test_linespacing_sets_the_distance_between_lines():
    def gap(spacing):
        fig, ax = vc.subplots(figsize = (3, 3))
        ax.text(0.5, 0.5, 'one\ntwo', size = 10, axes_fraction = True, linespacing = spacing)
        one, two = [o['ctm'][5] for o in fig._canvas().ops if o['kind'] == 'text' and o['text'] in ('one', 'two')]
        return one - two
    assert gap(1.0) == pytest.approx(10.0) and gap(1.25) == pytest.approx(12.5)


def test_emphasized_tick_labels_are_bold_in_their_color():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'couture')
    ax.barh(['one', 'two', 'ours'], [1, 2, 3])
    ax.set_yticks([0, 1, 2], ['one', 'two', 'ours'], emphasis = {'ours': 'a1'})
    ops = {o['text']: o for o in fig._canvas().ops if o['kind'] == 'text'}
    assert ops['ours']['fill_rgb'] == fig.pal.color('a1') and ops['ours']['font'].endswith('Bold')
    assert not ops['one']['font'].endswith('Bold')


def test_x_labels_on_top_take_their_room_above():
    def layout(**kw):
        fig, ax = vc.subplots(figsize = (4, 4), palette = 'couture')
        ax.heatmap(np.eye(3), ['alpha', 'beta', 'gamma'], ['a', 'b', 'c'])
        ax.set_title('Map', size = 9)
        ax.tick_params(axis = 'x', **kw)
        fig.layout()
        labels = {o['text']: o['ctm'][5] for o in fig._canvas().ops if o['kind'] == 'text'}
        return ax, labels
    low, below = layout()
    ax, above = layout(top = True)
    assert below['alpha'] < low.rect[1] and above['alpha'] > ax.rect[1] + ax.rect[3] and above['Map'] > above['alpha']
    with pytest.raises(ValueError, match = 'top'):
        ax.tick_params(axis = 'y', top = True)


def test_axis_names_take_their_own_size():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'poster')
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('Power', size = 5.6)
    ax.set_ylabel('FDR', size = 9)
    sizes = {o['text']: o['size'] for o in fig._canvas().ops if o['kind'] == 'text'}
    assert sizes['POWER' if fig.pal.caps else 'Power'] == 5.6 and sizes['FDR'] == 9


def test_heatmap_annotations_take_a_function():
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.heatmap(np.array([[0.25, 0.5]]), annot = True, fmt = lambda v: f'{100 * v:.0f}%')
    assert {a['s'] for a in ax.artists if a['kind'] == 'text'} == {'25%', '50%'}


def test_a_mosaic_spans_cells_and_leaves_dots_empty():
    fig, axes = vc.subplot_mosaic([['a', 'b', '.'], ['c', 'c', 'c']], figsize = (6, 4), height_ratios = [2, 1])
    assert set(axes) == {'a', 'b', 'c'} and axes['c']._span == (1, 3)
    for ax in axes.values():
        ax.plot([0, 1], [0, 1])
    fig.layout()
    a, b, c = (axes[k].rect for k in 'abc')
    assert c[2] > 2.5 * a[2] and c[1] + c[3] < a[1] and a[3] > c[3] and b[0] > a[0] + a[2]
    direct, twins = vc.subplot_mosaic([['a', 'b', '.'], ['c', 'c', 'c']], figsize = (6, 4), height_ratios = [2, 1],
                                      palette = 'aura')
    for ax in twins.values():
        ax.plot([0, 1], [0, 1])
    assert fig.in_palette('aura').to_png(72) == direct.to_png(72)
    with pytest.raises(ValueError, match = 'rectangle'):
        vc.subplot_mosaic('ab;ba')
    fig, axes = vc.subplot_mosaic('ab;cc')
    assert axes['c']._span == (1, 2)


def test_a_palette_minus_writes_numbers_only():
    fig, ax = vc.subplots(figsize = (3, 3), palette = _journal(minus = '−'))
    ax.bar(['-a', 'b'], [-2, 3])
    fig.layout()
    assert '−2' in ax.ylabels and ax.xlabels == ['-a', 'b']
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.plot([-2, 2], [0, 1])
    fig.layout()
    assert '-2' in ax.xlabels


def test_labels_that_fit_false_keeps_stay_as_set():
    def labels(**kw):
        fig, ax = vc.subplots(figsize = (2, 2), palette = 'poster')
        ax.plot([0, 100], [0, 1])
        ax.set_xticks(list(range(0, 101, 10)), [f'{v} units' for v in range(0, 101, 10)])
        ax.set_yticks(list(np.linspace(0, 1, 21)))
        ax.tick_params(**kw)
        fig.layout()
        return ax
    crowded, kept = labels(), labels(fit = False)
    assert crowded.xrot == 90.0 and len(crowded.yticks) < 21
    assert kept.xrot == 0.0 and len(kept.xticks) == 11 and len(kept.yticks) == 21


def test_a_title_pad_moves_the_title_and_its_room():
    def place(pad):
        fig, ax = vc.subplots(figsize = (3, 3))
        ax.plot([0, 1], [0, 1])
        ax.set_title('Title', size = 9, pad = pad)
        title = [o for o in fig._canvas().ops if o['kind'] == 'text' and o['text'] == 'Title'][0]
        return title['ctm'][5] - (ax.rect[1] + ax.rect[3]), ax._margins()[3]
    (near, room), (far, more) = place(2.0), place(12.0)
    assert near == pytest.approx(2.0) and far == pytest.approx(12.0) and more == pytest.approx(room + 10.0)


def test_nice_ticks_take_quarter_steps():
    assert nice_ticks(0, 0.8, 4) == [0, 0.2, 0.4, 0.6, 0.8]
    assert nice_ticks(0, 0.8, 4, steps = (1, 2, 2.5, 5, 10)) == [0, 0.25, 0.5, 0.75]


def test_an_offset_label_that_is_not_spread_has_no_leader():
    fig, ax = vc.subplots(figsize = (3, 3), palette = 'couture')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for v in np.linspace(0.1, 0.9, 7):
        ax.annotate('n', (v, v * 0.731), offset = (2.2, -2.2), va = 'top', halo = 0)
    assert not [o for o in fig._canvas().ops if o['kind'] == 'path' and o.get('stroke')
                and o['line_width'] == pytest.approx(0.5 * fig.pal.spacing)]


def test_boxes_cover_points_lines_bars_and_text():
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'poster')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([2, 50], [2, 2], s = 100)
    ax.plot([0, 10], [5, 5], lw = 2)
    ax.bar([8], [3], width = 1)
    ax.text(5, 8, 'label', size = 10, ha = 'center', va = 'center')
    fig.layout()
    boxes = ax.boxes()
    px, py = ax.page(2, 2)
    r = np.sqrt(100 * fig.pal.point_scale) / 2
    assert [px - r, py - r, px + r, py + r] in boxes.tolist()
    assert not np.any(boxes[:, 0] > ax.page(10, 0)[0] + 20)
    line = boxes[np.isclose(boxes[:, 3] - boxes[:, 1], 2)]
    assert len(line) > 50 and np.allclose((line[:, 1] + line[:, 3]) / 2, ax.page(0, 5)[1])
    bx0, by0 = ax.page(7.5, 0)
    assert any(np.allclose(b, [bx0, by0, ax.page(8.5, 3)[0], ax.page(8.5, 3)[1]]) for b in boxes)
    tx, ty = ax.page(5, 8)
    label = [b for b in boxes if abs((b[0] + b[2]) / 2 - tx) < 1e-6 and b[1] < ty < b[3]]
    assert label and label[0][2] - label[0][0] == pytest.approx(width('label', 10, fig.pal.text_bold))
    # Asking records nothing for in_palette
    assert len(fig._calls) == 6


def test_placed_text_keeps_its_side_where_an_annotation_turns():
    def ha(method):
        fig = vc.Figure(80, 40, unit = 'mm')
        ax = fig.add_axes((5, 5, 70, 30), unit = 'mm')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        args = ('name near the edge', (0.98, 0.5)) if method == 'annotate' else (0.98, 0.5, 'name near the edge')
        getattr(ax, method)(*args)
        fig.layout()
        return ax.artists[-1]['_ha']
    assert ha('annotate') == 'right' and ha('text') == 'left'


def _group_fills(fig):
    """The fill colors of the marks drawn, in drawing order."""
    return [op['fill_rgb'] for op in fig._canvas().ops if op['kind'] == 'path' and op.get('mark') and op['fill']]


def test_boxplot_draws_every_value_over_a_paler_box():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'poster')
    values = [1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 50.0]
    ax.boxplot({'a': values}, colors = ['a1'])
    offsets = ax.artists[0]['offsets'][0]
    # Every value is a point, equal values apart, within the jitter's share of the box, the same on every call
    assert len(offsets) == 7 and len(set(offsets[1:4])) == 3 and np.abs(offsets).max() <= 0.7 * 0.5 / 2
    again = vc.subplots(palette = 'poster')[1].boxplot({'a': values}, colors = ['a1']).artists[0]['offsets'][0]
    assert np.array_equal(offsets, again)
    fig.layout()
    assert ax.get_ylim()[1] > 50
    pal = fig.pal
    fills = _group_fills(fig)
    assert pal.color('a2') in fills and fills.count(pal.color('a1')) == 7
    # Without points, a value beyond the whiskers shows as a flier only, and the box keeps its color
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'poster')
    ax.boxplot({'a': values}, colors = ['a1'], points = False)
    fills = _group_fills(fig)
    assert pal.color('a2') not in fills and fills.count(pal.color('a1')) == 2


def test_the_palest_box_takes_darker_points():
    fig, ax = vc.subplots(palette = 'poster')
    ax.boxplot([[1, 2, 3]], colors = ['a3'])
    assert _group_fills(fig).count(fig.pal.color('a2')) == 3


def test_stripplot_draws_points_alone():
    fig, ax = vc.subplots(figsize = (5, 4), palette = 'aura')
    ax.stripplot({'a': [1.0, 1.0, 1.0, np.nan], 'b': [2.0, 3.0], 'c': []})
    strip = ax.artists[0]
    assert not strip['box'] and len(strip['values'][0]) == 3 and len(set(strip['offsets'][0])) == 3
    fig.layout()
    # Aura's flat points are one path per group, and no box goes under them
    assert ax.xlabels == ['a', 'b', 'c'] and _group_fills(fig) == fig.pal.contrast(2)


def test_violins_hold_one_area_and_match_a_gaussian_kernel():
    stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 40), rng.normal(3, 2, 25)
    fig, ax = vc.subplots(palette = 'lumen')
    ax.violinplot([a, b], width = 0.8)
    shapes = ax.artists[0]['shapes']
    # Seaborn's kernel: Scott's bandwidth, cut two bandwidths past the extremes
    for v, (grid, half) in zip((a, b), shapes):
        kde = stats.gaussian_kde(v)
        bw = np.sqrt(kde.covariance[0, 0])
        assert grid[0] == pytest.approx(v.min() - 2 * bw) and grid[-1] == pytest.approx(v.max() + 2 * bw)
        ratio = half / kde(grid)
        assert np.allclose(ratio, ratio[0])
    areas = [np.trapezoid(half, grid) for grid, half in shapes]
    assert areas[0] == pytest.approx(areas[1], rel = 0.02)
    assert max(half.max() for _, half in shapes) == pytest.approx(0.4)


def test_violin_points_stay_inside_and_inner_is_checked():
    v = np.random.default_rng(5).normal(0, 1, 60)
    fig, ax = vc.subplots(palette = 'couture')
    ax.violinplot([v, [2.0], [2.0, 2.0]], inner = 'points')
    art = ax.artists[0]
    grid, half = art['shapes'][0]
    assert np.all(np.abs(art['offsets'][0]) <= 0.8 * np.interp(v, grid, half) + 1e-12)
    # A single value, or equal ones, has no spread and shows as points
    assert art['shapes'][1] is None and art['shapes'][2] is None
    assert fig.to_png(72)[:4] == b'\x89PNG'
    with pytest.raises(ValueError, match = 'inner'):
        ax.violinplot([v], inner = 'quartiles')


@pytest.mark.parametrize('name', NAMES)
def test_violins_and_strips_in_every_palette_and_format(name, tmp_path):
    rng = np.random.default_rng(6)
    fig, axes = vc.subplots(1, 3, figsize = (9, 3), palette = name)
    data = {'u': rng.normal(0, 1, 30), 'v': rng.normal(1, 0.5, 20)}
    axes[0].violinplot(data)
    axes[1].violinplot(data, inner = 'points')
    axes[2].stripplot(data)
    for suffix in ('png', 'pdf', 'svg'):
        assert fig.save(tmp_path / f'groups.{suffix}').stat().st_size > 0
    fig.layout()
    assert len(axes[0].boxes()) == 2 and len(axes[2].boxes()) == 2


def test_shared_axes_share_limits_and_label_only_outer_panels():
    fig, axes = vc.subplots(2, 3, figsize = (9, 6), sharex = True, sharey = 'row')
    for i, ax in enumerate(axes.ravel()):
        ax.plot([0, i + 1], [0, 10 * (i // 3 + 1) + i])
        ax.set_xlabel('time')
        ax.set_ylabel('level')
    fig.layout()
    assert len({ax.get_xlim() for ax in axes.ravel()}) == 1
    assert axes[0, 0].get_ylim() == axes[0, 2].get_ylim() != axes[1, 0].get_ylim()
    assert [bool(ax.xlabels) for ax in axes.ravel()] == [False] * 3 + [True] * 3
    assert [bool(ax.ylabels) for ax in axes.ravel()] == [True, False, False] * 2
    assert [ax.xlabel for ax in axes.ravel()] == [''] * 3 + ['time'] * 3
    assert [ax.ylabel for ax in axes.ravel()] == ['level', '', ''] * 2
    # Plot areas line up along rows and columns
    rects = np.array([ax.rect for ax in axes.ravel()]).reshape(2, 3, 4)
    assert np.allclose(rects[0, :, 1], rects[0, 0, 1]) and np.allclose(rects[:, 0, 0], rects[0, 0, 0])
    assert np.allclose(rects[..., 2], rects[0, 0, 2]) and np.allclose(rects[..., 3], rects[0, 0, 3])


def test_shared_axes_share_set_limits_scales_and_inversion():
    fig, axes = vc.subplots(1, 2, sharey = True)
    axes[0].plot([1, 10], [1, 100])
    axes[1].plot([1, 10], [5, 50])
    axes[1].set_yscale('log')
    axes[0].invert_yaxis()
    fig.layout()
    assert axes[0]._yscale == 'log' and axes[0].get_ylim() == axes[1].get_ylim() and axes[1].get_ylim()[1] > 0
    axes[1].set_ylim(2, 20)
    fig.layout()
    assert axes[0].get_ylim() == axes[1].get_ylim() == (20, 2)
    with pytest.raises(ValueError, match = 'sharex'):
        vc.subplots(1, 2, sharex = 'rows')


def test_group_plots_and_shared_axes_in_another_palette_equal_ones_drawn_in_it():
    rng = np.random.default_rng(7)
    data = {'a': rng.normal(0, 1, 20), 'b': rng.normal(1, 1, 15)}
    def draw(palette):
        fig, axes = vc.subplots(1, 3, figsize = (9, 3), palette = palette, sharey = True)
        axes[0].boxplot(data)
        axes[0].set_ylabel('value')
        axes[1].violinplot(data)
        axes[2].stripplot(data)
        axes[2].set_ylabel('hidden')
        return fig
    moved, direct = draw('poster').in_palette('aura'), draw('aura')
    assert moved.to_png(72.0) == direct.to_png(72.0) and moved.to_pdf() == direct.to_pdf()


def _bracketed(palette, invert = False, log = False, **kw):
    rng = np.random.default_rng(5)
    data = {name: rng.lognormal(m, 0.4, 25) for name, m in zip('abcd', (0.0, 0.5, 1.2, 1.6))}
    fig, ax = vc.subplots(figsize = (5, 4), palette = palette, **kw)
    ax.boxplot(data)
    if log:
        ax.set_yscale('log')
    if invert:
        ax.invert_yaxis()
    ax.bracket(0, 1, 0.012)
    ax.bracket('b', 'c', '**')
    ax.bracket(0, 2, 2.3e-5)
    ax.bracket(2, 3, 'n.s.')
    ax.bracket('a', 'd', '***\nall')
    return fig, ax, data


def _bracket_parts(ax):
    """Each bracket's span, line, and top on the page, in drawing order."""
    plan = ax._bracket_plan()
    return [plan[id(a)] for a in ax.artists if a['kind'] == 'bracket']


@pytest.mark.parametrize('invert, log', [(False, False), (True, False), (False, True)])
def test_brackets_sit_over_their_groups_stack_and_fit_the_axes(invert, log):
    fig, ax, data = _bracketed('lumen', invert, log)
    fig.layout()
    x0, y0, w, h = ax.rect
    parts = _bracket_parts(ax)
    for (left, right, line, top), (g0, g1) in zip(parts, [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]):
        assert left == pytest.approx(ax.page(g0, 0)[0]) and right == pytest.approx(ax.page(g1, 0)[0])
        # Above every value of the groups it spans, on the page, and within the axes
        highest = max(float(np.max(ax.page(0, v)[1])) for v in list(data.values())[g0:g1 + 1])
        assert line - 4.0 > highest and top <= y0 + h
    # Brackets that meet lie one above the other, the narrower lower
    for i, j in [(0, 1), (1, 2), (0, 2), (2, 4), (3, 4)]:
        assert parts[i][3] < parts[j][2] - 4.0 or parts[j][3] < parts[i][2] - 4.0
    assert parts[2][2] > parts[0][3] and parts[4][2] > max(p[3] for p in parts[:4])
    # The limits grew only at the page's top
    bare = vc.subplots(figsize = (5, 4))[1].boxplot(data)
    if log:
        bare.set_yscale('log')
    if invert:
        bare.invert_yaxis()
    assert bare.get_ylim()[0] == pytest.approx(ax.ylim[0])
    assert (ax.ylim[1] < bare.get_ylim()[1]) if invert else (ax.ylim[1] > bare.get_ylim()[1])


def test_brackets_write_p_values_and_find_groups_by_name():
    fig, ax, _ = _bracketed('poster')
    assert [a['s'] for a in ax.artists if a['kind'] == 'bracket'] == ['p = 0.012', '**', 'p = 2.3×10^{-5}', 'n.s.',
                                                                    '***\nall']
    assert vc.subplots(palette = _journal(minus = '−'))[1].bracket(0, 1, 1e-7).artists[0]['s'] == 'p = 10^{−7}'
    # Categories of a line join the boxes' after them, rather than land on them
    ax.plot(['u', 'v'], [1, 2])
    assert (ax.bracket('u', 'v').artists[-1]['x0'], ax.artists[-1]['x1']) == (4.0, 5.0)
    for ends in (('a', 'zz'), (1, 1), (0, np.nan)):
        with pytest.raises(ValueError, match = 'bracket'):
            ax.bracket(*ends)


def test_brackets_keep_limits_set_by_hand_and_cast_no_shadow():
    fig, ax, _ = _bracketed('aura')
    ax.set_ylim(0, 30)
    fig.layout()
    assert ax.ylim == (0.0, 30.0) and len(ax._bracket_plan()) == 5
    c = fig._canvas()
    texts = [op for op in c.ops if op['kind'] == 'text' and op['text'] in ('**', 'n.s.')]
    assert len(texts) == 2 and not any(op.get('mark') or op.get('clip') for op in texts)
    # Limits that leave no room above the data, or leave its ends behind, hide a bracket, as zooming in a window would
    ax.set_ylim(0, 3)
    fig.layout()
    shown = ax._bracket_plan()
    assert 0 < len(shown) < 5 and max(v[3] for v in shown.values()) <= ax.rect[1] + ax.rect[3] + 0.5
    ax.set_xlim(-0.5, 1.5)
    fig.layout()
    assert all(a['x1'] <= 1 for a in ax.artists if a['kind'] == 'bracket' and id(a) in ax._bracket_plan())
    assert '**' not in [op['text'] for op in fig._canvas().ops if op['kind'] == 'text']


def test_brackets_share_their_lift_with_axes_sharing_y_and_stay_put_on_saving_again():
    rng = np.random.default_rng(2)
    fig, (a, b) = vc.subplots(1, 2, figsize = (8, 4), sharey = True)
    a.boxplot([rng.normal(0, 1, 20), rng.normal(1, 1, 20)])
    b.violinplot([rng.normal(0, 1, 20), rng.normal(3, 1, 20)])
    b.bracket(0, 1, '*')
    first = fig.to_png(72.0)
    assert a.ylim == b.ylim and a._lift == b._lift > 1.0
    a.get_ylim()
    assert a.ylim == b.ylim and fig.to_png(72.0) == first


def test_bracket_boxes_cover_line_and_text():
    fig, ax, _ = _bracketed('couture')
    fig.layout()
    boxes, (left, right, line, top) = ax.boxes(), _bracket_parts(ax)[1]
    on_line = boxes[np.isclose((boxes[:, 1] + boxes[:, 3]) / 2, line) & (boxes[:, 0] >= left - 1)
                    & (boxes[:, 2] <= right + 1)]
    assert len(on_line) > 20
    label = boxes[np.isclose((boxes[:, 0] + boxes[:, 2]) / 2, (left + right) / 2) & (boxes[:, 1] > line)
                  & (boxes[:, 1] < top)]
    assert len(label) == 1 and label[0][3] <= top + 1e-6
    assert label[0][2] - label[0][0] == pytest.approx(width('**', fig.pal.tick_size, fig.pal.text_bold))


def test_brackets_in_another_palette_equal_ones_drawn_in_it():
    moved, direct = _bracketed('poster', True)[0].in_palette('couture'), _bracketed('couture', True)[0]
    assert moved.to_png(72.0) == direct.to_png(72.0) and moved.to_pdf() == direct.to_pdf()
    assert re.sub(r'grad\d+', 'grad', moved.to_svg()) == re.sub(r'grad\d+', 'grad', direct.to_svg())


@pytest.mark.parametrize('name', NAMES)
def test_boxes_and_inner_boxes_draw_on_a_turned_axis(name):
    def ops(turn):
        fig, (a, b) = vc.subplots(1, 2, palette = name)
        a.boxplot([[1, 2, 3, 4, 9], [2, 3, 5, 6]])
        b.violinplot([[1, 2, 3, 4, 9], [2, 3, 5, 6]])
        if turn:
            a.invert_yaxis()
            b.invert_yaxis()
        return len(fig._canvas().ops)
    assert ops(True) == ops(False)


def _rasterized(palette, rasterized = True, **kw):
    rng = np.random.default_rng(4)
    fig, ax = vc.subplots(figsize = (3, 3), palette = palette)
    ax.scatter(rng.normal(size = 300), rng.normal(size = 300), hue = rng.choice(['a', 'b', 'c'], 300),
               rasterized = rasterized, **kw)
    ax.set_xlabel('log2 fold change')
    return fig, ax


@pytest.mark.parametrize('name', NAMES)
def test_a_rasterized_scatter_is_one_image_in_vector_output_only(name):
    fig, ax = _rasterized(name)
    ops = fig._canvas().ops
    images = [op for op in ops if op['kind'] == 'image' and op.get('mark')]
    texts = [op for op in ops if op['kind'] == 'text']
    # One image for all three colors, clipped to the axes, with the axes, the legend, and the text still vector
    assert len(images) == 1 and images[0]['clip'] and len(texts) > 5
    assert any(op['kind'] == 'path' and op.get('mark') is None for op in ops)
    x0, y0, w, h = ax.rect
    a, _, _, d, e, f = images[0]['ctm']
    assert e > x0 - 10.0 and f > y0 - 10.0 and e + a < x0 + w + 10.0 and f + d < y0 + h + 10.0
    assert images[0]['w'] == pytest.approx(a * 300 / 72) and images[0]['h'] == pytest.approx(d * 300 / 72)
    # PNG stays as it was, and the image matches what it draws
    plain = _rasterized(name, False)[0]
    assert fig.to_png(72.0) == plain.to_png(72.0)
    drawn = raster.render(ops, fig.width, fig.height, 300 / 72, 300 / 72, (1.0, 1.0, 1.0)).astype(int)
    stamped = raster.render(plain._canvas(stamp_dpi = 300).ops, fig.width, fig.height, 300 / 72, 300 / 72,
                            (1.0, 1.0, 1.0)).astype(int)
    assert np.abs(drawn - stamped).max() <= 2
    # Beside a soft shadow, which is an image of its own
    assert b'/Subtype /Image' in fig.to_pdf() and fig.to_svg().count('<image') == 1 + bool(fig.pal.shadow)


def test_rasterized_points_take_their_dpi_any_marker_and_no_clip():
    fig, ax = _rasterized('couture', 72, s = np.linspace(5, 60, 300), marker = 's')
    fig.layout()
    op = [op for op in fig._canvas().ops if op['kind'] == 'image'][0]
    assert op['w'] == pytest.approx(op['ctm'][0], abs = 1e-9)
    # A named scatter has no clip, so its image covers what its points reach on the page
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.scatter([0, 1], [0, 1], gid = 'dots', rasterized = True)
    ops = fig._canvas().ops
    assert [op['gid'] for op in ops if op['kind'] == 'image'] == ['dots'] and '<g id="dots">' in fig.to_svg()
    # Nothing to draw, or nothing in view, makes no image
    fig, ax = vc.subplots(figsize = (3, 3))
    ax.scatter([], [], rasterized = True)
    ax.scatter([5.0], [5.0], rasterized = True)
    ax.set_xlim(0, 1)
    assert not [op for op in fig._canvas().ops if op['kind'] == 'image']


def test_a_rasterized_scatter_in_another_palette_equals_one_drawn_in_it():
    moved, direct = _rasterized('poster')[0].in_palette('aura'), _rasterized('aura')[0]
    assert moved.to_pdf() == direct.to_pdf() and moved.to_png(72.0) == direct.to_png(72.0)
    assert re.sub(r'grad\d+', 'grad', moved.to_svg()) == re.sub(r'grad\d+', 'grad', direct.to_svg())


def test_data_without_a_range_keeps_its_ticks_off_the_corners():
    fig, ax = vc.subplots(figsize = (3, 2.4), palette = 'poster')
    ax.scatter(1.0, 2.0)
    fig.layout()
    for ticks, (lo, hi) in ((ax.xticks, ax.xlim), (ax.yticks, ax.ylim)):
        assert lo < min(ticks) and max(ticks) < hi


@pytest.mark.parametrize('name, marker', [('poster', 'o'), ('lumen', 'o'), ('aura', '^')])
def test_marks_under_opaque_blocks_are_skipped_and_change_nothing(name, marker):
    rng = np.random.default_rng(9)
    x, y = rng.normal(size = 6000) * 8 + 50, rng.normal(size = 6000) * 8 + 50
    def stamped():
        c = pdfwrite.Canvas(BytesIO(), pagesize = (100, 100))
        c.stamp_dpi = 150.0
        c.state['clip'] = [((0, 0, 100, 100), c.state['ctm'])]
        marks.points(c, get_palette(name), x, y, 30.0, (0.2, 0.4, 0.6), marker = marker)
        return c.ops[-1]['rgb'] + c.ops[-1]['alpha']
    calls = []
    hidden = marks._hidden
    with patch.object(marks, '_hidden', lambda *a: calls.append(hidden(*a)) or calls[-1]):
        skipping = stamped()
    with patch.object(marks, '_hidden', lambda buf, cx, *a: np.zeros(len(cx), bool)):
        assert stamped() == skipping
    assert sum(v.sum() for v in calls) > 3000


def _tidy(n = 12, seed = 9):
    """Long-form data: a value per sample, with its class, condition, and name."""
    rng = np.random.default_rng(seed)
    rows = [(c, h, rng.normal(i + 2 * j, 1.0), f'{c}-{h}-{k}') for i, c in enumerate(['sialyl', 'fucosyl', 'mannose'])
            for j, h in enumerate(['healthy', 'cancer']) for k in range(n)]
    return {key: [r[i] for r in rows] for i, key in enumerate(['cls', 'cond', 'value', 'sample'])}


def _same_stats(a, b):
    return len(a) == len(b) and all(s is t is None or s is not None and t is not None and np.allclose(s[:5], t[:5])
                                    and np.array_equal(s[5], t[5]) for s, t in zip(a, b))


def test_long_form_input_matches_a_dict_of_groups():
    pd = pytest.importorskip('pandas')
    tidy = _tidy()
    by = {c: [v for v, k in zip(tidy['value'], tidy['cls']) if k == c] for c in dict.fromkeys(tidy['cls'])}
    for method in ('boxplot', 'stripplot', 'violinplot'):
        wide = getattr(vc.subplots()[1], method)(by).artists[0]
        for table in (tidy, pd.DataFrame(tidy)):
            ax = getattr(vc.subplots()[1], method)(table, x = 'cls', y = 'value')
            long = ax.artists[0]
            assert np.array_equal(long['x'], wide['x']) and all(np.array_equal(a, b) for a, b in
                                                                 zip(long['values'], wide['values']))
            assert _same_stats(long['stats'], wide['stats']) and not long['horizontal'] and ax._cats['x'] == {
                'sialyl': 0, 'fucosyl': 1, 'mannose': 2}
    # Columns given as values work alike, numbers as categories go by size, and order chooses and orders them
    ax = vc.subplots()[1].boxplot(x = [3, 1, 2, 1], y = [1.0, 2.0, 3.0, 4.0])
    assert list(ax._cats['x']) == ['1', '2', '3'] and list(ax.artists[0]['values'][0]) == [2.0, 4.0]
    ax = vc.subplots()[1].boxplot(tidy, x = 'cls', y = 'value', order = ['mannose', 'sialyl'])
    assert list(ax._cats['x']) == ['mannose', 'sialyl'] and len(ax.artists[0]['values'][0]) == 24
    with pytest.raises(ValueError, match = 'orient'):
        vc.subplots()[1].boxplot(tidy, x = 'cls', y = 'value', orient = 'sideways')
    with pytest.raises(ValueError, match = 'hue'):
        vc.subplots()[1].boxplot([[1, 2]], hue = ['a', 'b'])


def test_hue_sets_groups_side_by_side_with_one_legend():
    tidy = _tidy()
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'poster')
    ax.boxplot(tidy, x = 'cls', y = 'value', hue = 'cond', points = False)
    ax.stripplot(tidy, x = 'cls', y = 'value', hue = 'cond')
    box = ax.artists[0]
    # Two levels share 0.8 of each category, each box 0.8 of its share
    assert np.allclose(box['x'], [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2]) and np.allclose(box['w'], 0.32)
    assert box['rgb'][::2] == [fig.pal.contrast(2)[0]] * 3 and box['rgb'][1::2] == [fig.pal.contrast(2)[1]] * 3
    assert box['groups'][:2] == [('sialyl', 'healthy'), ('sialyl', 'cancer')]
    # The strip over the boxes adds points to the legend, and nothing twice
    assert [e[0] for e in ax.legend_items] == ['cond', 'healthy', 'cancer', 'healthy', 'cancer']
    assert [e[2] for e in ax.legend_items] == ['header', 'patch', 'patch', 'point', 'point']
    assert ax.bracket(('fucosyl', 'healthy'), ('fucosyl', 'cancer'), 0.01).artists[-1]['x0'] == pytest.approx(0.8)
    with pytest.raises(ValueError, match = 'hue'):
        ax.bracket(('fucosyl', 'unknown'), 'sialyl')
    nested = {c: {h: [v for v, k, g in zip(tidy['value'], tidy['cls'], tidy['cond']) if (k, g) == (c, h)]
                  for h in ('healthy', 'cancer')} for c in ('sialyl', 'fucosyl', 'mannose')}
    again = vc.subplots()[1].boxplot(nested, points = False).artists[0]
    assert np.allclose(again['x'], box['x']) and _same_stats(again['stats'], box['stats'])
    fig.layout()
    # One box per group of the boxes, then of the strip, side by side across the page, before the bracket's pieces
    boxes = ax.boxes()[:12]
    assert ax.xlabels == ['sialyl', 'fucosyl', 'mannose'] and np.all(np.diff(boxes[:6, 0]) > 0)


def test_horizontal_groups_run_across_with_the_first_on_top():
    tidy = _tidy()
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'lumen')
    ax.violinplot(tidy, x = 'value', y = 'cls', hue = 'cond')
    art = ax.artists[0]
    assert art['horizontal'] and list(ax._cats['y']) == ['sialyl', 'fucosyl', 'mannose'] and ax._yinv
    fig.layout()
    assert ax.ylabels == ['sialyl', 'fucosyl', 'mannose'] and ax.ylim[0] > ax.ylim[1]
    # Each violin is wider than tall on the page, and the first group is the highest
    boxes = ax.boxes()
    assert np.all(boxes[:, 2] - boxes[:, 0] > boxes[:, 3] - boxes[:, 1]) and boxes[0, 3] == boxes[:, 3].max()
    assert ax.get_xlim()[0] < min(tidy['value']) and ax.get_xlim()[1] > max(tidy['value'])
    for orient in ('h', 'v'):
        drawn = vc.subplots(figsize = (6, 4), palette = 'lumen')[1].boxplot(tidy, x = 'cls', y = 'value',
                                                                             orient = orient)
        assert drawn.artists[0]['horizontal'] == (orient == 'h')


@pytest.mark.parametrize('name', NAMES)
def test_hue_horizontal_and_swarms_in_every_palette_and_format(name, tmp_path):
    tidy = _tidy(30)
    fig, axes = vc.subplots(2, 2, figsize = (9, 7), palette = name)
    axes[0, 0].boxplot(tidy, x = 'cls', y = 'value', hue = 'cond', swarm = True)
    axes[0, 1].violinplot(tidy, x = 'value', y = 'cls', hue = 'cond', inner = 'swarm', density_norm = 'width')
    axes[1, 0].stripplot(tidy, x = 'value', y = 'cls', swarm = True, names = 'sample')
    axes[1, 1].boxplot(tidy, x = 'value', y = 'cls', points = False)
    axes[1, 1].violinplot(tidy, x = 'value', y = 'cls', inner = 'box', alpha = 0.5)
    for suffix in ('png', 'pdf', 'svg'):
        assert fig.save(tmp_path / f'groups.{suffix}').stat().st_size > 0


def test_names_follow_their_values():
    fig, ax = vc.subplots()
    ax.stripplot({'a': [1.0, np.nan, 3.0], 'b': [2.0]}, names = {'a': ['x', 'y', 'z'], 'b': ['w']})
    assert [list(n) for n in ax.artists[0]['names']] == [['x', 'z'], ['w']]
    tidy = _tidy(3)
    ax.boxplot(tidy, x = 'cls', y = 'value', names = 'sample')
    assert list(ax.artists[1]['names'][0]) == ['sialyl-healthy-0', 'sialyl-healthy-1', 'sialyl-healthy-2',
                                               'sialyl-cancer-0', 'sialyl-cancer-1', 'sialyl-cancer-2']
    assert vc.subplots()[1].boxplot([[1.0]]).artists[0]['names'] is None
    with pytest.raises(ValueError, match = 'names'):
        vc.subplots()[1].boxplot({'a': [1.0, 2.0]}, names = {'a': ['x']})
    with pytest.raises(ValueError, match = 'names'):
        vc.subplots()[1].boxplot(tidy, x = 'cls', y = 'value', names = ['x'])


@pytest.mark.parametrize('horizontal', [False, True])
def test_swarm_points_never_overlap_and_stay_within_their_group(horizontal):
    rng = np.random.default_rng(4)
    data = {'a': rng.normal(0, 1, 60), 'b': rng.normal(1, 0.5, 30)}
    fig, ax = vc.subplots(figsize = (6, 6), palette = 'couture')
    ax.stripplot(data, swarm = True, orient = 'h' if horizontal else 'v')
    fig.layout()
    art = ax.artists[0]
    offsets, sizes = ax._spread(art)
    assert offsets is ax._spread(art)[0]
    for x, v, u, s in zip(art['x'], art['values'], offsets, sizes):
        assert 0 < np.abs(u).max() <= 0.4 + 1e-9
        px, py = ax.page(v, x + u) if horizontal else ax.page(x + u, v)
        gaps = np.hypot(px[:, None] - px[None], py[:, None] - py[None]) + np.eye(len(v)) * 1e9
        assert gaps.min() >= np.sqrt(s * fig.pal.point_scale) - 1e-6
    # Equal values beyond the width squeeze into it, where the others keep their places
    tied = vc.subplots(figsize = (6, 2))[1].stripplot([[0.0] * 150 + list(np.linspace(1, 9, 9))], swarm = True)
    tied.figure.layout()
    u = tied._spread(tied.artists[0])[0][0]
    assert np.abs(u[:150]).max() == pytest.approx(0.4) and np.all(u[150:] == 0)


def test_shared_axes_merge_their_categories():
    fig, axes = vc.subplots(1, 2, figsize = (8, 3), sharex = True, sharey = True)
    axes[0].boxplot({'a': [1, 2, 3], 'b': [2, 3, 4]})
    axes[1].stripplot({'c': [1, 2], 'a': [3, 4]})
    assert axes[0]._cats['x'] is axes[1]._cats['x'] and axes[1].artists[0]['x'].tolist() == [2.0, 0.0]
    fig.layout()
    assert axes[0].xlabels == axes[1].xlabels == ['a', 'b', 'c'] and axes[0].get_xlim() == axes[1].get_xlim()
    # Unshared panels keep categories of their own, and a replay in another palette shares them as drawn
    fig2, axes2 = vc.subplots(1, 2, sharey = True)
    axes2[0].boxplot({'a': [1, 2]})
    axes2[1].boxplot({'c': [1, 2]})
    assert axes2[1].artists[0]['x'].tolist() == [0.0]
    moved = fig.in_palette('aura')
    assert moved.axes[0]._cats['x'] is moved.axes[1]._cats['x']
    assert moved.axes[1].artists[0]['x'].tolist() == [2.0, 0.0]


def test_hue_and_swarms_in_another_palette_equal_ones_drawn_in_it():
    tidy = _tidy()
    def draw(palette):
        fig, axes = vc.subplots(1, 2, figsize = (8, 3), palette = palette, sharex = True)
        axes[0].boxplot(tidy, x = 'cls', y = 'value', hue = 'cond', swarm = True)
        axes[1].violinplot(tidy, x = 'cls', y = 'value', hue = 'cond', inner = 'swarm', order = ['mannose', 'sialyl'])
        return fig
    moved, direct = draw('lumen').in_palette('poster'), draw('poster')
    assert moved.to_png(72.0) == direct.to_png(72.0) and moved.to_pdf() == direct.to_pdf()


def test_violins_scale_by_area_count_or_width():
    rng = np.random.default_rng(3)
    data = [rng.normal(0, 1, 40), rng.normal(0, 0.1, 10)]
    peaks = {norm: [h.max() for _, h in vc.subplots()[1].violinplot(data, density_norm = norm).artists[0]['shapes']]
             for norm in ('area', 'count', 'width')}
    assert peaks['width'] == pytest.approx([0.4, 0.4]) and max(peaks['area']) == pytest.approx(0.4)
    # The narrow group peaks highest by area, and its fewer values bring it down by count
    assert peaks['area'][1] / peaks['area'][0] == pytest.approx(4 * peaks['count'][1] / peaks['count'][0])
    with pytest.raises(ValueError, match = 'density_norm'):
        vc.subplots()[1].violinplot(data, density_norm = 'height')


def _beads(label, vertical = False):
    """A stand-in for GlycoDraw, as CI has no glycowork: a row of outlined discs per '|'-separated part of a label of
    letters, the first part lowest and longest, its rightmost disc the root (the lowest in a vertical drawing), and a
    bit of text; anything else is no drawing."""
    if not label.replace('|', '').isalpha():
        return None
    rows = label.split('|')
    w, h = 50 * max(map(len, rows)), 50 * len(rows)
    body = []
    for r, row in enumerate(rows):
        for i, _ in enumerate(row):
            x, y = w - 25 - 50 * i, h - 25 - 50 * r
            x, y = (y, w - x) if vertical else (x, y)
            body.append(f'<circle cx="{x}" cy="{y}" r="18" fill="#058F60" stroke="#000000" stroke-width="2"/>')
    body.append(f'<text x="{w / 2}" y="{h - 2}" font-size="12" text-anchor="middle">b4</text>')
    w, h = (h, w) if vertical else (w, h)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'{"".join(body)}</svg>')


def _symbols(ops):
    """Middles and sizes of the discs among a display list's ops, on the page, one per subpath, as a glyph's discs of
    one paint are one path."""
    out = []
    for op in ops:
        if op['kind'] == 'path' and op['fill'] and op['stroke'] and op['fill_rgb'] == pytest.approx(
                (5 / 255, 143 / 255, 96 / 255)):
            m, subs = op['ctm'], []
            for s in op['path']:
                if s[0] == 'm':
                    subs.append([])
                subs[-1] += [(m[0] * s[i] + m[2] * s[i + 1] + m[4], m[1] * s[i] + m[3] * s[i + 1] + m[5])
                             for i in range(1, len(s), 2)]
            for pts in map(np.array, subs):
                out.append((*(pts.min(0) + pts.max(0)) / 2, np.ptp(pts, axis = 0).max()))
    return np.array(out).reshape(-1, 3)


def test_glyph_labels_sit_their_root_on_the_tick():
    fig, ax = vc.subplots(figsize = (5, 4))
    ax.barh(['abc|d', 'ab', '42'], [1, 2, 3])
    ax.tick_params('y', glyphs = _beads)
    ops = fig.display_list()
    beads = _symbols(ops)
    # Four discs and two, and the label that draws nothing stays text
    assert len(beads) == 6 and any(op['kind'] == 'text' and op['text'] == '42' for op in ops)
    for y in ax._ty([0.0, 1.0]):
        # Each glyph's root, its rightmost disc, is level with its tick, and the drawing ends left of the axes
        row = beads[np.abs(beads[:, 1] - y) < 1e-6]
        assert len(row) and row[:, 0].max() + row[0, 2] / 2 < ax.rect[0]
    assert np.ptp(beads[:, 2]) < 1e-6 and beads[0, 2] <= ax._tick_sizes()[1] * vc.text.GLYPH_SYMBOL + 1e-6


def test_glyph_labels_grow_the_page_before_any_is_left_out():
    names = ['abcdefghijklmn'[k] + 'a' * (k % 3 + 1) + '|ab|a' for k in range(14)]
    fig, ax = vc.subplots(figsize = (5, 3))
    ax.barh(names, np.arange(14))
    ax.tick_params('y', glyphs = _beads)
    fig.layout()
    assert ax.ylabels == names and fig.height > 3 * 72
    # Neighbors in a column never meet, whether the labels share one or stagger in two
    boxes = sorted((y + e[1], y + e[3]) for y, e in zip(ax._ty(ax.yticks), (
        ax._glyph_extent('y', vc.text.glyph(_beads, s)) for s in ax.ylabels)))
    step = ax._gcols
    assert all(lo[1] <= hi[0] for lo, hi in zip(boxes[:-step], boxes[step:]))
    # Past what the page may grow, labels are left out as text would be
    fig, ax = vc.subplots(figsize = (5, 1))
    ax.barh([n + 'a' * k for k in range(4) for n in names], np.arange(56))
    ax.tick_params('y', glyphs = _beads)
    fig.layout()
    assert len(ax.ylabels) < 56 and fig.height <= 3 * 72 + 1e-6


def test_crowded_x_glyphs_run_up_the_page():
    asked = []
    def maker(label, vertical):
        asked.append(vertical)
        return _beads(label, vertical)
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.boxplot({'abcdefghijkl'[:k] + 'xyz'[k % 3]: [1, 2, k] for k in range(5, 12)})
    ax.tick_params('x', glyphs = maker)
    beads = _symbols(fig.display_list())
    assert ax.xrot == 90.0 and True in asked
    for x in ax._tx(ax.xticks):
        # The root, the lowest disc, sits on the tick, and the drawing hangs below the axes
        column = beads[np.abs(beads[:, 0] - x) < 1e-6]
        assert len(column) >= 5 and column[:, 1].max() < ax.rect[1]


def test_glyphs_are_stickers_on_a_dark_page_and_halos_over_marks():
    fig, ax = vc.subplots(palette = 'aura-dark')
    ax.scatter([0, 1], [0, 1])
    ax.annotate('abc', (0.5, 0.5), glyph = _beads)
    cut = [op for op in fig.display_list() if op['kind'] == 'path' and op['fill_rgb'] == (1.0, 1.0, 1.0)]
    # One cut runs around the discs, and one around the text's box where the text is drawn
    assert all(op['cap'] == 1 and op['stroke'] for op in cut)
    assert sorted([s[0] for s in op['path']].count('m') for op in cut) in ([3], [1, 3])
    fig, ax = vc.subplots()
    ax.annotate('abc', (0.5, 0.5), glyph = _beads)
    assert sum(op['fill_rgb'] == fig.pal.color('paper') for op in fig.display_list() if op['kind'] == 'path') in (1, 2)
    fig, ax = vc.subplots()
    ax.annotate('abc', (0.5, 0.5), glyph = _beads, halo = 0.0)
    assert not any(op['kind'] == 'path' and op['cap'] == 1 and op['fill_rgb'] == fig.pal.color('paper')
                   for op in fig.display_list())


def test_spread_glyphs_stack_across_both_sides_of_their_points():
    fig, ax = vc.subplots(figsize = (5, 4))
    ax.scatter([0, 1, 2, 3], [1.0, 1.02, 1.04, 0.98])
    for x, y, side in ((0, 1.0, 'left'), (1, 1.02, 'left'), (3, 1.04, 'right'), (2, 0.98, 'right')):
        ax.annotate('abcd|ab', (x, y), offset = (4, 0) if side == 'left' else (-4, 0), ha = side, glyph = _beads,
                    spread = True)
    fig.layout()
    boxes = ax.boxes()[-4:]
    assert len(_symbols(fig.display_list())) == 24
    # The four drawings span one another across the page, so they stack, and no two meet
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = boxes[i], boxes[j]
            assert a[3] <= b[1] + 1e-6 or b[3] <= a[1] + 1e-6 or a[2] <= b[0] or b[2] <= a[0]


def test_legend_rows_hold_their_glyphs():
    fig, ax = vc.subplots(figsize = (5, 4))
    for name in ('abc|ab|a', 'ab', 'x1'):
        ax.plot([0, 1], [0, 1], label = name)
    ax.legend('right', glyphs = _beads)
    ops = fig.display_list()
    assert len(_symbols(ops)) == 8 and any(op['kind'] == 'text' and op['text'] == 'x1' for op in ops)
    spots, (w, h) = ax._legend_grid()
    assert h > 3 * vc.axes.LEGEND_ROW and spots[1][1] - spots[0][1] > vc.axes.LEGEND_ROW


def test_glyph_figures_save_everywhere_and_replay_in_another_palette(tmp_path):
    def draw(palette):
        fig, heat = vc.clustermap(np.arange(12.0).reshape(4, 3) % 5, yticklabels = ['abc', 'ab|a', 'abcd', 'zz9'],
                                  palette = palette, glyphs = _beads)
        heat.annotate('ab', (1, 1), glyph = _beads)
        return fig
    for kind in ('png', 'pdf', 'svg'):
        draw('lumen').save(tmp_path / f'glyphs.{kind}')
    # The drawing's text is embedded with the page's own
    assert b'/FontFile2' in (tmp_path / 'glyphs.pdf').read_bytes()
    moved, direct = draw('couture').in_palette('poster'), draw('poster')
    assert moved.to_png(72.0) == direct.to_png(72.0) and moved.to_pdf() == direct.to_pdf()


def test_glyph_makers_are_checked_and_registered():
    with pytest.raises(ValueError, match = 'Glyphs come from'):
        vc.subplots()[1].tick_params('y', glyphs = 'snfg')
    vc.register_glyphs('beads', _beads)
    fig, ax = vc.subplots()
    ax.bar(['abc', 'de'], [1, 2])
    ax.tick_params('x', glyphs = 'beads')
    assert len(_symbols(fig.display_list())) == 5
    ax.tick_params('x', glyphs = False)
    assert not len(_symbols(fig.display_list()))


def test_a_glyph_writes_each_paint_once_and_leaves_out_unreadable_text():
    g = vc.text.glyph(_beads, 'abcd|ab')
    # Six discs of one paint are one path, and the cut one path for shapes and one for the text's box
    assert len([op for op in g.ops if op['kind'] == 'path']) == 1 and len(g.cut) == 2
    c = pdfwrite.Canvas(BytesIO(), (200, 200))
    vc.text.draw_glyph(c, g, 100, 100, 3.0, (1.0, 1.0, 1.0))
    assert not any(op['kind'] == 'text' for op in c.ops) and len(c.ops) == 2 and not c.glyphs
    c = pdfwrite.Canvas(BytesIO(), (200, 200))
    vc.text.draw_glyph(c, g, 100, 100, 30.0, (1.0, 1.0, 1.0))
    assert sum(op['kind'] == 'text' for op in c.ops) == 1 and len(c.ops) == 4 and c.glyphs


def test_crowded_glyph_rows_stagger_in_two_columns():
    names = ['abcdefghijklmnopqrstuvwxyz'[k] + 'a' * (k % 3 + 1) + '|ab|a|a' for k in range(26)]
    fig, ax = vc.subplots(figsize = (5, 3))
    ax.barh(names, np.arange(26))
    ax.tick_params('y', glyphs = _beads)
    ops = fig.display_list()
    # One column would need more than the threefold page, and two fit within it
    assert ax._gcols == 2 and ax.ylabels == names and fig.height <= 3 * 3 * 72 + 1e-6
    # Where the page may grow enough, the labels keep one column, and past what it may, some are left out once
    fig, ax = vc.subplots(figsize = (5, 8))
    ax.barh(names, np.arange(26))
    ax.tick_params('y', glyphs = _beads)
    fig.layout()
    assert ax._gcols == 1 and ax.ylabels == names
    fig, ax = vc.subplots(figsize = (5, 2))
    ax.barh(names, np.arange(26))
    ax.tick_params('y', glyphs = _beads)
    fig.layout()
    assert len(ax.ylabels) == 13 and fig.height == pytest.approx(3 * 2 * 72)
    # Every other row reaches its label through a hairline, and both columns face the axes from the left
    widths, offsets = ax._ytick_columns(ax._tick_sizes()[1])
    assert sorted(set(offsets))[0] == 0.0 and len(set(offsets)) == 2 and widths > max(offsets)
    hairlines = [op for op in ops if op['kind'] == 'path' and op['stroke'] and not op['fill']
                 and abs(op['line_width'] - 0.4 * fig.pal.spacing) < 1e-9]
    assert len(hairlines) == 13
    beads = _symbols(ops)
    assert beads[:, 0].max() < ax.rect[0]


def _crossings(spots):
    """How many pairs of the leaders among placed glyphs cross."""
    lines = [v[2] for v in spots.values() if v[2] is not None]
    turn = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return sum(turn(u[:2], u[2:], v[:2]) * turn(u[:2], u[2:], v[2:]) < 0 and
               turn(v[:2], v[2:], u[:2]) * turn(v[:2], v[2:], u[2:]) < 0
               for n, u in enumerate(lines) for v in lines[n + 1:])


def test_spread_glyphs_find_room_inside_the_axes():
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 0.3, 60), rng.normal(0, 0.3, 60)
    fig, ax = vc.subplots(figsize = (6, 5))
    ax.scatter(x, y)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    for i in range(8):
        ax.annotate('abc|ab', (x[i], y[i]), glyph = _beads, spread = True)
    fig.layout()
    boxes = ax.boxes()[-8:]
    x0, y0, w, h = ax.rect
    assert (boxes[:, 0] >= x0 - 1e-6).all() and (boxes[:, 2] <= x0 + w + 1e-6).all()
    assert (boxes[:, 1] >= y0 - 1e-6).all() and (boxes[:, 3] <= y0 + h + 1e-6).all()
    for i in range(8):
        for j in range(i + 1, 8):
            a, b = boxes[i], boxes[j]
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    # Drawn where boxes() says, each with a leader where it stands apart, and no two leaders cross
    spots = ax._place_glyphs()
    assert len(spots) == 8 and not _crossings(spots)
    # A crowd too dense to label point by point keeps its drawings apart, uncrossed, on the page, and off its points
    fig, ax = vc.subplots(figsize = (6, 5))
    ax.scatter(np.zeros(14), np.linspace(0, 0.5, 14))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-3, 3)
    for i, y in enumerate(np.linspace(0, 0.5, 14)):
        ax.annotate('abcde'[:i % 4 + 2] + '|ab', (0.0, y), glyph = _beads, spread = True)
    fig.layout()
    spots, boxes = ax._place_glyphs(), ax.boxes()[-14:]
    assert not _crossings(spots)
    # No drawing covers another's point
    px, ys = float(ax._tx(0.0)), ax._ty(np.linspace(0, 0.5, 14))
    assert (boxes[:, 1] >= 0).all()
    assert not any(b[0] + 1 < px < b[2] - 1 and b[1] + 1 < y < b[3] - 1 for i, b in enumerate(boxes)
                   for j, y in enumerate(ys) if i != j)
    for i in range(14):
        for j in range(i + 1, 14):
            a, b = boxes[i], boxes[j]
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    # The same page places them the same way
    assert fig.to_png(72.0) == fig.to_png(72.0)
    # A few drawings with room around them grow past their type's size, and a size given by hand is the largest
    for size, largest in ((None, max(vc.axes.GLYPH_SIZES)), (6.0, 1.0)):
        fig, ax = vc.subplots(figsize = (6, 5))
        ax.scatter([0, 1], [0, 1])
        ax.annotate('ab', (0.2, 0.8), glyph = _beads, spread = True, size = size)
        fig.layout()
        (_, _, _, pts), = ax._place_glyphs().values()
        assert pts == pytest.approx((size or 8.6) * vc.text.GLYPH_SYMBOL * largest)


def test_a_halo_follows_tracked_letters():
    c = pdfwrite.Canvas(BytesIO(), (400, 100))
    vc.text.draw(c, 10, 50, 'HALO', 20, (0, 0, 0), tracking = 6.0, halo = 2.0)
    halo, letters = c.ops[0], c.ops[1]
    xs = [s[i] + halo['ctm'][4] for s in halo['path'] for i in range(1, len(s), 2)]
    _, w, (lead, tail) = vc.text._pieces('HALO', vc.text.fonts()[0], 20, 6.0, False)
    # The ink starts at the anchor, and the halo's last letter ends where the tracked text's ink does, not a tracking
    # per letter short
    assert w == pytest.approx(width('HALO', 20) + 4 * 6.0) and min(xs) == pytest.approx(10, abs = 0.05)
    assert max(xs) == pytest.approx(10 + w - lead - tail, abs = 0.05) and letters['char_space'] == 6.0


def test_glycans_draw_through_glycowork():
    pytest.importorskip('glycowork')
    assert vc.text.glyph('glycan', 'Gal(b1-4)GlcNAc') is not None and vc.text.glyph('glycan', 'not a glycan') is None


def _cloud(n = 20000, seed = 3):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0, 0.3, (n, 2)), rng.uniform(-4, 4, (40, 2))]


def test_points_that_pile_up_give_way_to_a_field_and_sparse_ones_stay():
    pal = get_palette('poster')
    xy = _cloud() * 40 + 200
    (bands, keep), = marks.crowd(pal, [(xy[:, 0], xy[:, 1], 3.0, pal.color('a1'))], (0.0, 0.0, 400.0, 400.0))
    # The core is a map of bands in the hue's shades, faint outside to strong inside, and its points give way to it
    shades = [pal.color(s) for s in pal.a[::-1]]
    assert len(bands) == marks.CROWD_BANDS and keep[:20000].mean() < 0.3
    assert bands[0][0] == pytest.approx(tuple(ramp(shades, 0.1)))
    assert bands[-1][0] == pytest.approx(tuple(ramp(shades, 0.9)))
    # Every outline closes on itself
    assert all(xs[0] == xs[-1] and ys[0] == ys[-1] for _, outlines in bands for xs, ys in outlines)
    far = np.hypot(*(xy[20000:] - 200).T) > 80
    assert far.sum() > 30 and keep[20000:][far].all()
    sparse = marks.crowd(pal, [(xy[20000:, 0], xy[20000:, 1], 3.0, pal.color('a1'))], (0.0, 0.0, 400.0, 400.0))
    assert sparse[0][0] is None and sparse[0][1].all()


def test_a_field_lies_under_the_points_in_every_format_and_density_false_keeps_them():
    xy = _cloud()
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'aura')
    ax.scatter(xy[:, 0], xy[:, 1], s = 12)
    ops = fig.display_list()
    bands = [i for i, op in enumerate(ops) if op.get('even_odd')]
    shapes = [i for i, op in enumerate(ops) if op['kind'] in ('path', 'markers') and op.get('mark')
              and not op.get('even_odd')]
    # Vector bands, each outlined by a hairline of the page, under the points
    assert len(bands) == marks.CROWD_BANDS and shapes and max(bands) < min(shapes)
    assert all(ops[i]['stroke_rgb'] == fig.pal.color('paper') for i in bands)
    plain, ax = vc.subplots(figsize = (4, 4), palette = 'aura')
    ax.scatter(xy[:, 0], xy[:, 1], s = 12, density = False)
    assert not any(op.get('even_odd') for op in plain.display_list())
    # Raster output thins the same points, and draws faster for it
    assert fig.to_png(dpi = 60) != plain.to_png(dpi = 60)


def test_a_color_scale_keeps_every_point():
    xy = _cloud()
    fig, ax = vc.subplots(figsize = (4, 4))
    ax.scatter(xy[:, 0], xy[:, 1], c = xy[:, 0], s = 12)
    assert not any(op.get('even_odd') for op in fig.display_list())


def _comfortaa():
    root = os.path.join(os.path.dirname(pdfwrite.__file__), 'fonts')
    pdfmetrics.registerFont(pdfwrite.TTFont('Comfortaa', os.path.join(root, 'Comfortaa-Regular.ttf')))
    return 'Comfortaa'


def test_tabular_figures_take_one_width_and_leave_tabular_fonts_alone():
    name = _comfortaa()
    cell = vc.text._figures(name)
    pieces, w, _ = vc.text._pieces('11', name, 10, 0.0, True)
    assert cell and w == pytest.approx(2 * cell * 10) and len(pieces) == 2
    assert vc.text._pieces('10', name, 10, 0.0, True)[1] == pytest.approx(w)
    try:
        pdfmetrics.registerFont(pdfwrite.TTFont('CenturyGothic', 'GOTHIC.TTF'))
    except FileNotFoundError:
        return
    # Century Gothic's figures are tabular already, so its numbers stay one piece
    assert vc.text._figures('CenturyGothic') is None
    assert len(vc.text._pieces('10.5', 'CenturyGothic', 10, 0.0, True)[0]) == 1


def test_numeric_y_labels_line_up_on_their_decimal_points():
    fig, ax = vc.subplots(figsize = (4, 3))
    ax.plot([0, 1, 2], [0, 1.5, 0.2])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ops = [op for op in fig.display_list() if op['kind'] == 'text']
    rows = {}
    for op in ops:
        rows.setdefault(round(op['ctm'][5] + op['y'], 2), []).append(op)
    points = []
    for label in ('0', '0.5', '1', '1.5'):
        row = next(r for r in rows.values() if ''.join(o['text'] for o in sorted(r, key = lambda o: o['x'])) == label)
        left, size = min(o['ctm'][4] + o['x'] for o in row), row[0]['size']
        points.append(left + width(label.split('.')[0], size, tabular = True))
    assert max(points) - min(points) < 1e-6


def test_text_lines_up_by_its_ink():
    c = pdfwrite.Canvas(BytesIO(), (200, 100))
    for ha in ('left', 'right'):
        c.ops = []
        vc.text.draw(c, 100, 50, 'Title', 20, (0, 0, 0), ha = ha)
        op = c.ops[0]
        name = vc.text.fonts()[0]
        lsb, rsb = vc.text._bearings(name, 'T')[0], vc.text._bearings(name, 'e')[1]
        start = op['ctm'][4] + op['x']
        expect = 100 - lsb * 20 if ha == 'left' else 100 + rsb * 20 - width('Title', 20)
        assert start == pytest.approx(expect)
    # Numbers set as a table's line up by their cells instead
    c.ops = []
    vc.text.draw(c, 100, 50, '15', 20, (0, 0, 0), ha = 'right', tabular = True)
    assert min(o['ctm'][4] + o['x'] for o in c.ops) == pytest.approx(100 - width('15', 20, tabular = True))


def test_apca_and_ciede2000_match_their_references():
    from viscose.palettes import apca, delta_e
    # APCA-W3's own examples, and Sharma, Wu, and Dalal's (2005) CIEDE2000 test pairs
    assert apca((0, 0, 0), (1, 1, 1)) == pytest.approx(106.04, abs = 0.05)
    assert apca((1, 1, 1), (0, 0, 0)) == pytest.approx(107.88, abs = 0.05)
    assert apca((0x88 / 255,) * 3, (1, 1, 1)) == pytest.approx(63.06, abs = 0.05)
    assert apca((1, 1, 1), (0x88 / 255,) * 3) == pytest.approx(68.54, abs = 0.05)
    pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425), ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
             ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0), ((50, 2.5, 0), (73, 25, -18), 27.1492),
             ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]
    assert delta_e([p[0] for p in pairs], [p[1] for p in pairs]) == pytest.approx([p[2] for p in pairs], abs = 1e-4)


def test_colors_of_different_hues_are_moved_apart_for_color_blind_readers():
    from viscose.palettes import CVD_MIN, apart
    rng = np.random.default_rng(5)
    fig, ax = vc.subplots(figsize = (4, 4), palette = 'aura')
    ax.scatter(rng.normal(size = 80), rng.normal(size = 80), hue = np.repeat(list('ABCD'), 20))
    fig.layout()
    colors = [a['rgb'] for a in ax.artists if a['kind'] == 'points']
    assert any(why == 'color vision' for _, _, why in fig.adjusted) and len(set(colors)) == 4
    # Each color's hue is its nearest palette shade's, as the check moves colors only within their hue
    from viscose.palettes import _hex, _oklab
    shades = [(i, fig.pal.color(s)) for i, h in enumerate((fig.pal.a, fig.pal.b)) for s in h]
    hue = lambda c: min(shades, key = lambda s: np.linalg.norm(_oklab(s[1]) - _oklab(c)))[0]
    assert all(apart(c, d) >= CVD_MIN for c in colors for d in colors if c != d and hue(c) != hue(d))
    # Each move is a small one, so that the figure keeps its look
    assert all(np.linalg.norm(_oklab(_hex(a)) - _oklab(_hex(b))) < 0.25 for a, b, _ in fig.adjusted)
    # The legend shows what the plot does, and laying out again changes nothing
    assert [e[1] for e in ax.legend_items] == colors
    before = list(fig.adjusted)
    fig.layout()
    assert fig.adjusted == before


def test_legibility_leaves_passing_colors_alone_and_can_be_switched_off():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'lumen')
    ax.plot([0, 1], [0, 1], color = 'a1')
    ax.plot([0, 1], [1, 0], color = 'yellow')
    ax.text(0.5, 0.5, 'pale', color = 'a3')
    fig.layout()
    found = {a['kind']: a['rgb'] for a in ax.artists}
    assert ax.artists[0]['rgb'] == fig.pal.color('a1') and found['text'] != fig.pal.color('a3')
    assert ax.artists[1]['rgb'] != (1.0, 1.0, 0.0)
    off, ax = vc.subplots(figsize = (4, 3), palette = 'lumen')
    off.legible = False
    ax.plot([0, 1], [1, 0], color = 'yellow')
    off.layout()
    assert ax.artists[0]['rgb'] == (1.0, 1.0, 0.0) and off.adjusted == []


def test_a_press_prints_what_it_can_and_keeps_the_hue_of_what_it_cannot():
    from viscose.palettes import REACH, _oklch, cielab, delta_e
    press = vc.Press()
    inside = np.array([get_palette('aura').color('a1')])
    assert delta_e(press.lab(press.separate(inside)), cielab(inside))[0] < REACH
    pale = get_palette('lumen').color('a3')
    printed = press.printed(np.array([pale]))[0]
    (L0, C0, h0), (L1, C1, h1) = _oklch(pale), _oklch(printed)
    # Out of the press's gamut, it gives up chroma, not hue or lightness
    assert C1 < C0 and abs(L1 - L0) < 0.03 and abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi) < 0.1
    ink = press.separate(np.array([[0.11, 0.1, 0.09], [1.0, 1.0, 1.0]]))
    # Near-black text takes mostly black ink, and white takes none
    assert ink[0, 3] > 0.6 and ink[1].max() < 1e-6 and (ink.sum(1) <= press.ink_limit + 1e-9).all()
    rows = press.gamut('lumen')
    assert rows[0][0] == 'a3' and rows[0][3] == max(r[3] for r in rows)


def test_the_default_press_writes_an_output_profile_that_reads_back():
    import struct
    icc = vc.Press().icc()
    assert icc[12:24] == b'prtrCMYKLab ' and icc[36:40] == b'acsp' and struct.unpack('>I', icc[:4])[0] == len(icc)
    count = struct.unpack('>I', icc[128:132])[0]
    tags = {sig for sig, _, _ in struct.iter_unpack('>4sII', icc[132:132 + 12 * count])}
    assert {b'desc', b'cprt', b'wtpt', b'A2B0', b'A2B1', b'B2A0', b'B2A1', b'gamt'} <= tags
    again = vc.Press(icc)
    gray = np.array([[0.5, 0.5, 0.5]])
    assert np.allclose(again.lab(vc.Press().separate(gray)), vc.Press().lab(vc.Press().separate(gray)))


def test_a_pdf_for_a_press_is_a_pdf_x_in_cmyk(tmp_path):
    import zlib
    if not hasattr(pdfwrite.Canvas(BytesIO()), 'pdfx'):
        pytest.skip('needs glycorender with CMYK output')
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'aura')
    fig.suptitle('Print')
    ax.bar(['a', 'b'], [1, 2], color = 'b2')
    ax.scatter(*_cloud(3000).T, s = 10)
    pdf = fig.to_pdf(press = True)
    assert pdf.startswith(b'%PDF-1.6') and b'DeviceRGB' not in pdf and b'/DeviceCMYK' in pdf
    for key in (b'/OutputIntents', b'/S /GTS_PDFX', b'(CGATS TR 001)', b'/DestOutputProfile', b'/TrimBox', b'/ID [',
                b'/Trapped /False', b'<pdfxid:GTS_PDFXVersion>PDF/X-4</pdfxid:GTS_PDFXVersion>', b'(Print)'):
        assert key in pdf
    streams = []
    for s in re.findall(rb'stream\n(.*?)\nendstream', pdf, re.S):
        try:
            streams.append(zlib.decompress(s))
        except zlib.error:
            pass
    content = [s for s in streams if b' cm' in s and b'BT' in s][0]
    assert re.search(rb' k\n', content) and not re.search(rb' rg\n', content) and not re.search(rb' RG\n', content)
    assert any(s[12:24] == b'prtrCMYKLab ' for s in streams)
    with pytest.raises(ValueError):
        fig.save(tmp_path / 'print.svg', press = True)
    proof = fig.to_png(dpi = 40, press = True)
    assert proof[:8] == b'\x89PNG\r\n\x1a\n' and proof != fig.to_png(dpi = 40)
