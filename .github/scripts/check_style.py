#!/usr/bin/env python
"""viscose's coding standard from CLAUDE.md, on whole files. Derived from glycowork's check_style.py.

Rules enforced beyond the pycodestyle subset below:
  GW1  no blank line inside a function body, and unlike glycowork none around a nested def or class either
  GW2  keyword arguments are written with spaces around '=' (f(x = 1), not f(x=1))

glycowork's GW3 (parameters with a default are passed by name) is not one of viscose's rules.

slate imports this file and calls check_blank_lines, check_kwarg_spacing, and pycodestyle_findings by name, to show the
same findings as squiggles while typing, so nothing runs on import.

Usage:
  python .github/scripts/check_style.py viscose tests examples benchmarks .github/scripts
"""
import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

PYCODESTYLE_SELECT = 'E101,E111,E225,E231,E301,E302,E303,E305,E501,E702,E703,W191,W291,W292,W293,W391'
MAX_LINE_LENGTH = 120


def check_blank_lines(tree, lines, tokens, report):
    in_string, starts = set(), []
    for tok in tokens:
        kind = tokenize.tok_name[tok.type]
        if kind == 'STRING':
            in_string.update(range(tok.start[0], tok.end[0] + 1))
        # Python 3.12+ splits f-strings into tokens, so a blank line inside one starts no token
        elif kind == 'FSTRING_START':
            starts.append(tok.start[0])
        elif kind == 'FSTRING_END':
            in_string.update(range(starts.pop(), tok.end[0] + 1))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i in range(node.body[0].lineno, node.end_lineno):
                if not lines[i - 1].strip() and i not in in_string:
                    report(i, 'GW1', 'blank line inside function body')


def check_kwarg_spacing(tokens, report):
    stack, prev = [], None
    for tok in tokens:
        kind = tokenize.tok_name[tok.type]
        if kind == 'OP' and tok.string in '([{':
            stack.append(tok.string == '(' and prev is not None and (prev.type == tokenize.NAME or prev.string in ')]'))
        elif kind == 'OP' and tok.string in ')]}':
            stack.pop()
        elif kind == 'OP' and tok.string == '=' and stack and stack[-1]:
            if not tok.line[:tok.start[1]].endswith(' ') or not tok.line[tok.end[1]:].startswith(' '):
                report(tok.start[0], 'GW2', "keyword argument '=' needs spaces around it")
        if kind not in ('COMMENT', 'NL', 'NEWLINE', 'INDENT', 'DEDENT'):
            prev = tok


def pycodestyle_findings(source):
    """[(line, column, code, message)] for the pycodestyle subset, checked in-process on a string, for slate."""
    import pycodestyle
    found = []
    class Collect(pycodestyle.BaseReport):
        def error(self, line_number, offset, text, check):
            if super().error(line_number, offset, text, check):
                found.append((line_number, offset, text[:4], text[5:]))
    style = pycodestyle.StyleGuide(select = PYCODESTYLE_SELECT.split(','), max_line_length = MAX_LINE_LENGTH,
                                   reporter = Collect)
    pycodestyle.Checker(lines = io.StringIO(source).readlines(), options = style.options).check_all()
    return found


if __name__ == '__main__':
    files = sorted({f for p in map(Path, sys.argv[1:]) for f in (p.rglob('*.py') if p.is_dir() else [p])})
    findings = set()
    for f in files:
        source = f.read_text(encoding = 'utf-8')
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        report = lambda line, code, message: findings.add(f'{f}:{line}: {code} {message}')
        check_kwarg_spacing(tokens, report)
        check_blank_lines(ast.parse(source), source.splitlines(), tokens, report)
    proc = subprocess.run([sys.executable, '-m', 'pycodestyle', f'--max-line-length={MAX_LINE_LENGTH}',
                           f'--select={PYCODESTYLE_SELECT}', *map(str, files)], capture_output = True, text = True)
    # pycodestyle missing or crashed: failing loudly beats a silent pass without the E and W rules
    if proc.returncode and not proc.stdout:
        sys.exit(proc.stderr.strip())
    findings.update(proc.stdout.splitlines())
    for line in sorted(findings):
        print(line)
    if findings:
        sys.exit(f'\n{len(findings)} style violation(s). See Code style in CLAUDE.md.')
    print('Style check passed.')
