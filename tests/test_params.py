# -*- coding: utf-8 -*-
"""
Параметры-ползунки: буква в формуле (кроме x, y, e) — параметр, подставляется
точным числом при разборе; редактор принимает буквы как переменные.

Запуск:
    MPLBACKEND=Agg /usr/bin/python3.12 -m pytest tests/test_params.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import function_visualizer as fv  # noqa: E402
from math_editor import MathModel  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_params():
    fv.PARAMS = {}
    fv.clear_symbolic_cache()
    yield
    fv.PARAMS = {}
    fv.clear_symbolic_cache()


def test_find_parameters():
    assert fv.find_parameters(["a*x**2 + b", "sin(k*x)", "x**2+y**2=c", "", "x=2"]) == ["a", "b", "c", "k"]
    assert fv.find_parameters(["sin(x) + exp(x) + pi + E"]) == []          # известные имена — не параметры
    assert fv.find_parameters(["foo*x"]) == []                              # многобуквенное — не параметр
    assert fv.is_param_name("a") and fv.is_param_name("k") and not fv.is_param_name("x")
    assert not fv.is_param_name("e") and not fv.is_param_name("ab")


def test_param_substitution_and_cache():
    fv.PARAMS = {"a": "1.5"}
    expr, _raw, f = fv.build_numpy_func("a*x**2")
    assert float(f(2.0)) == pytest.approx(6.0)
    assert expr == fv.sympify("3*x**2/2")                                   # 1.5 → 3/2 (точно)
    fv.PARAMS = {"a": "2"}
    expr2, _r, f2 = fv.build_numpy_func("a*x**2")                           # другое значение — другой кэш
    assert float(f2(2.0)) == pytest.approx(8.0)
    assert expr2 != expr
    with pytest.raises(NameError):
        fv.PARAMS = {}
        fv.build_numpy_func("a*x**2")                                       # без значения — unknown name


def test_param_in_exact_labels():
    """С a = 2 корни x² − a точные: ±√2."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fv.PARAMS = {"a": "2"}
    fv.FUNCS = ["x**2 - a"]; fv.CURVE_COLORS = ["#000"]; fv.CURVE_WIDTHS = [1.5]; fv.CURVE_STYLES = ["-"]
    fv.FUNC_DOMAINS = [(float("-inf"), float("inf"))]
    fv.X_LIM_L, fv.X_LIM_R, fv.Y_LIM_B, fv.Y_LIM_T = -5, 5, -5, 5
    fv.FILL = []; fv.X_TAG = True; fv.SHOW_VALUES = True; fv.MATHTEXT_LABELS = False
    fig = Figure(figsize=(6, 6), dpi=100); FigureCanvasAgg(fig)
    fv.SYMBOLIC_MODE = 'compute'
    res = fv.plot_function(fig)
    texts = [t.get_text() for t in res['ax'].texts]
    assert any("√2" in t or "sqrt(2)" in t for t in texts), texts


def test_editor_accepts_parameter_letters():
    m = MathModel()
    for tok in ["a", "*", "x", "sq", "+", "k"]:
        m.insert_token(tok)
    assert m.to_sympy().replace(" ", "") in ("a*x**2+k", "a*x**(2)+k")
    m2 = MathModel.from_text("a*x^2 + b")
    assert "a" in m2.to_sympy() and "b" in m2.to_sympy()
    m3 = MathModel(); m3.insert_token("e")
    assert m3.to_sympy().strip() == "E"                                     # e остаётся константой
