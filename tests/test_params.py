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


# ── Разбор один раз, значения связываются при каждом вызове ──────────────

def _count_parse(monkeypatch):
    """Подменяет parse_expr счётчиком вызовов (build_* импортируют его из
    sympy.parsing.sympy_parser при каждом вызове)."""
    import sympy.parsing.sympy_parser as sp
    real = sp.parse_expr
    calls = []

    def counting(*a, **k):
        calls.append(a[0] if a else k.get("s"))
        return real(*a, **k)
    monkeypatch.setattr(sp, "parse_expr", counting)
    return calls


def test_numpy_func_parsed_once_across_param_changes(monkeypatch):
    from sympy import Mul, Pow
    x = fv._X_SYM
    calls = _count_parse(monkeypatch)                                       # sympify(str) тоже разбирает: сравниваем с символами
    fv.PARAMS = {"a": "1"}
    expr1, raw1, f1 = fv.build_numpy_func("a*x^2")
    assert len(calls) == 2                                                  # expr + expr_raw
    assert float(f1(2.0)) == pytest.approx(4.0)
    assert expr1 == x**2
    fv.PARAMS = {"a": "2"}
    expr2, raw2, f2 = fv.build_numpy_func("a*x^2")
    assert len(calls) == 2                                                  # второй вызов НЕ разбирает заново
    assert float(f2(2.0)) == pytest.approx(8.0)
    assert expr2 == 2 * x**2
    assert raw2 == Mul(2, Pow(x, 2), evaluate=False)
    np.testing.assert_allclose(f2(np.array([1.0, -3.0])), [2.0, 18.0])     # массив
    assert float(f1(2.0)) == pytest.approx(4.0)                             # старое замыкание держит старое значение
    fv.PARAMS = {"a": "0"}
    expr0, raw0, f0 = fv.build_numpy_func("a*x^2")
    assert len(calls) == 2
    assert expr0 == 0 and raw0 == Mul(0, Pow(x, 2), evaluate=False)         # структура raw сохранена
    assert float(f0(3.0)) == 0.0


def test_numpy_func_without_params_not_reparsed(monkeypatch):
    calls = _count_parse(monkeypatch)
    fv.PARAMS = {"a": "1"}
    e1, _r1, f1 = fv.build_numpy_func("sin(x) + x^2")
    assert len(calls) == 2
    fv.PARAMS = {"a": "2.5"}
    e2, _r2, f2 = fv.build_numpy_func("sin(x) + x^2")
    assert len(calls) == 2
    assert e1 == e2 and f1 is f2
    fv.PARAMS = {}
    e3, _r3, f3 = fv.build_numpy_func("sin(x) + x^2")                      # набор имён сменился, разбор не нужен
    assert len(calls) == 2 and f3 is f1
    assert float(f3(0.0)) == 0.0


def test_implicit_func_binds_current_values(monkeypatch):
    x, y = fv._X_SYM, fv._Y_SYM
    calls = _count_parse(monkeypatch)
    fv.PARAMS = {"a": "9"}
    H1, h1 = fv.build_implicit_func("x^2 + y^2", "a")
    n = len(calls)
    assert H1 == x**2 + y**2 - 9
    assert float(h1(3.0, 0.0)) == pytest.approx(0.0)
    fv.PARAMS = {"a": "4"}
    H2, h2 = fv.build_implicit_func("x^2 + y^2", "a")
    assert len(calls) == n                                                  # без повторного разбора
    assert H2 == x**2 + y**2 - 4
    assert float(h2(2.0, 0.0)) == pytest.approx(0.0)
    X, Y = np.meshgrid([0.0, 2.0], [0.0, 1.0])
    np.testing.assert_allclose(h2(X, Y), X**2 + Y**2 - 4)
    assert float(h1(3.0, 0.0)) == pytest.approx(0.0)                       # старое замыкание держит a = 9
    with pytest.raises(NameError):
        fv.PARAMS = {}
        fv.build_implicit_func("x^2 + y^2", "a")


def _draw_texts(funcs):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fv.FUNCS = list(funcs); fv.CURVE_COLORS = ["#000"] * len(funcs)
    fv.CURVE_WIDTHS = [1.5] * len(funcs); fv.CURVE_STYLES = ["-"] * len(funcs)
    fv.FUNC_DOMAINS = [(float("-inf"), float("inf"))] * len(funcs)
    fv.X_LIM_L, fv.X_LIM_R, fv.Y_LIM_B, fv.Y_LIM_T = -5, 5, -5, 5
    fv.FILL = []; fv.X_TAG = True; fv.SHOW_VALUES = True; fv.MATHTEXT_LABELS = False
    fig = Figure(figsize=(6, 6), dpi=100); FigureCanvasAgg(fig)
    fv.SYMBOLIC_MODE = 'compute'
    res = fv.plot_function(fig)
    return [t.get_text() for t in res['ax'].texts]


def test_param_exact_labels_roots_half():
    """a·x² − 1 при a = 4: корни ±1/2 подписаны точно; после сдвига ползунка
    (a = 1) подписи пересчитаны для нового значения."""
    fv.PARAMS = {"a": "4"}
    texts = _draw_texts(["a*x^2 - 1"])
    assert any("1/2" in t and "-" not in t for t in texts), texts
    assert any("-1/2" in t for t in texts), texts
    fv.PARAMS = {"a": "1"}
    texts2 = _draw_texts(["a*x^2 - 1"])
    assert not any("1/2" in t for t in texts2), texts2
    assert any(t.strip() in ("1", "-1") for t in texts2), texts2
