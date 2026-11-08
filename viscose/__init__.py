"""viscose: scientific plots that are pieces of art. Palettes bring their own special effects."""
from importlib.metadata import version, PackageNotFoundError
from . import marks, text
from .figure import Figure, clustermap, linkage, show_png, subplot_mosaic, subplots
from .axes import Axes
from .palettes import Palette, dark_mode, default_palette, palette, palettes, register_palette, set_palette, get_palette
try:
    __version__ = version('viscose')
except PackageNotFoundError:
    __version__ = '0.0.0'
__all__ = ['Axes', 'Figure', 'Palette', 'clustermap', 'dark_mode', 'default_palette', 'get_palette', 'linkage',
           'palette', 'palettes', 'register_palette', 'set_palette', 'show_png', 'subplot_mosaic', 'subplots']
