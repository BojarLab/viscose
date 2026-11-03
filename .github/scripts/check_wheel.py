#!/usr/bin/env python
"""Check a built distribution: the wheel holds the whole package, and draws from a clean venv away from the checkout.

Usage:
  python .github/scripts/check_wheel.py dist
"""
import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

SMOKE = """
import viscose as vc
assert 'site-packages' in vc.__file__ and vc.__version__ != '0.0.0', (vc.__file__, vc.__version__)
for name in vc.palettes:
    fig, ax = vc.subplots(palette = name)
    ax.plot([0, 1], [0, 1], label = 'line', marker = True)
    ax.bar([0, 1], [1, 2])
    ax.legend()
    assert fig.to_png(72)[:4] == b'\\x89PNG' and fig.to_pdf()[:4] == b'%PDF' and '<svg' in fig.to_svg(), name
print(f'viscose {vc.__version__} draws all {len(vc.palettes)} palettes from {vc.__file__}')
"""
dist = Path(sys.argv[1]).resolve()
wheels, sdists = list(dist.glob('viscose-*.whl')), list(dist.glob('viscose-*.tar.gz'))
if len(wheels) != 1 or len(sdists) != 1:
    sys.exit(f'Expected one wheel and one sdist in {dist}, found {wheels + sdists}')
with zipfile.ZipFile(wheels[0]) as z:
    names = z.namelist()
missing = [p.as_posix() for p in Path('viscose').rglob('*.py') if p.as_posix() not in names]
if missing:
    sys.exit(f'Not in the wheel: {missing}')
extra = {n.split('/')[0] for n in names if not n.split('/')[0].endswith('.dist-info')} - {'viscose'}
if extra:
    sys.exit(f'In the wheel, but not part of the package: {extra}')
if not wheels[0].name.endswith('-py3-none-any.whl') or not any(n.endswith('LICENSE') for n in names):
    sys.exit(f'{wheels[0].name} is not pure Python or has no license file')
with tempfile.TemporaryDirectory() as tmp:
    venv.create(Path(tmp) / 'venv', with_pip = True)
    python = str(Path(tmp) / 'venv' / ('Scripts' if os.name == 'nt' else 'bin') / 'python')
    subprocess.run([python, '-m', 'pip', 'install', '--quiet', str(wheels[0])], check = True, cwd = tmp)
    # -I, and a working directory elsewhere, so that the checkout cannot stand in for the wheel
    subprocess.run([python, '-I', '-c', SMOKE], check = True, cwd = tmp)
