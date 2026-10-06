# -*- coding: utf-8 -*-
"""
Типографика текстов программы: в строковых литералах app.py,
function_visualizer.py и math_editor.py (кроме docstring) нет длинного
тире «—» и точки с запятой «;». Исключение помечается на той же строке
комментарием «typography: ok» (например, разделитель CSV).

Запуск:
    /usr/bin/python3.12 -m pytest tests/test_typography.py -q
"""
import io
import os
import tokenize

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("app.py", "function_visualizer.py", "math_editor.py")
FORBIDDEN = ("—", ";")          # длинное тире, точка с запятой


def _string_literals(path):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:
            continue
        body = tok.string.lstrip("rbufRBUF")
        if body.startswith(('"""', "'''")):
            continue                                   # docstring / комментарий
        line = lines[tok.start[0] - 1]
        if "typography: ok" in line:
            continue
        yield tok.start[0], tok.string


@pytest.mark.parametrize("name", FILES)
def test_no_em_dash_or_semicolon_in_ui_strings(name):
    bad = [(ln, s[:80]) for ln, s in _string_literals(os.path.join(ROOT, name))
           if any(ch in s for ch in FORBIDDEN)]
    assert bad == [], f"{name}: длинное тире или точка с запятой в строках {bad}"
