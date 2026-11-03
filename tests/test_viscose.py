import re
import sys
from io import BytesIO
from unittest.mock import patch
import numpy as np
import pytest
import viscose as vc
from viscose import marks
from viscose.axes import fmt, nice_ticks
from viscose.palettes import get_palette, ramp
from viscose.text import runs, width
from glycorender import pdfwrite, raster


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
    c = fig._canvas()
    x0, y0, w, h = ax.rect
    assert x0 > 0 and y0 > 0 and x0 + w <= fig.width and y0 + h <= fig.height
    assert w > 100 and h > 100


def test_clustermap_dendrograms_line_up_with_the_matrix():
    pytest.importorskip('scipy')
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
    op = fig._shadow_op(300.0)
    assert op['w'] <= 600 and op['h'] <= 600 and len(op['rgb']) == op['w'] * op['h'] * 3
    assert abs(op['ctm'][0] * 300.0 / 72.0 - op['w']) < 1e-6 and abs(op['ctm'][3] * 300.0 / 72.0 - op['h']) < 1e-6
    # The soft shadow reaches past the bar but fades out before the page's corner
    assert op['alpha'][0] == 0 and max(op['alpha']) > 0


def test_colliding_tick_labels_turn_and_then_thin():
    fig, ax = vc.subplots(figsize = (6, 4), palette = 'lumen')
    ax.bar([f'a long category {i}' for i in range(12)], range(12))
    fig._layout()
    assert ax._xrot == 90.0 and len(ax.xlabels) == 12
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
    pytest.importorskip('scipy')
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


def test_a_glycan_label_near_the_edge_keeps_its_side_and_grows_the_page():
    fig, ax = vc.subplots(figsize = (4, 3), palette = 'poster')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', (0.9, 0.5), offset = (5, 0), svg_label = True)
    x = float(re.search(r'translate\(([\d.]+) ', fig.to_svg()).group(1))
    assert abs(x - (float(ax._tx(0.9)) + 5)) < 0.01 and fig.width > 4 * 72


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
    op = fig._shadow_op(300.0)
    a, _, _, d, e, f = op['ctm']
    # On the output's own pixels, and smaller than the page
    sc = 300.0 / 72.0
    assert op['w'] < fig.width * sc and op['h'] < fig.height * sc
    for value, size in ((a * sc, op['w']), (d * sc, op['h'])):
        assert abs(value - size) < 1e-6
    assert abs(e * sc - round(e * sc)) < 1e-6 and abs(f * sc - round(f * sc)) < 1e-6
    assert len(op['alpha']) == op['w'] * op['h'] and len(op['rgb']) == 3 * op['w'] * op['h']


def test_shadow_does_not_wrap_around_the_page():
    # A mark at the very bottom of the page casts its shadow off the page, not back in at the top
    fig = vc.Figure(4, 4, palette = 'aura')
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0)).off()
    ax.bar([0, 1], [1.0, 1.0], width = 1.0)
    ax.set_ylim(0.0, 3.0)
    arr = pixels(fig, 100)
    assert arr[:10].min() == 255


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
