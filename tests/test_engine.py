# -*- coding: utf-8 -*-
"""
Тесты движка function_visualizer.py (backend Agg, дисплей не нужен).

Запуск:
    MPLBACKEND=Agg /usr/bin/python3.12 -m pytest tests/test_engine.py -q

Покрытие (контракт DESIGN.md, модуль B, п. 9):
  • пять базовых сцен (baseline_render.py): без исключений, errors == {},
    точные подписи среди ax.texts;
  • цикл cached/pending: первая отрисовка pending=True, ≤ 5 итераций
    run_jobs(take_pending_jobs()) + redraw → те же подписи, что в compute;
  • некорректный ввод ('x^', 'sin(') → errors[idx], остальные кривые рисуются;
  • пустые строки пропускаются, индексы заливки стабильны;
  • nroot/cbrt, векторный make_y_array, подписки на canvas, ANNOTATION_OFFSETS,
    штриховка трёх стилей, identify_value (отказ от «почти» значений), fmt_sym;
  • регрессия относительно исходного движка (коммит 2e2179d): те же точки.
Замеры времени печатаются по каждому случаю (видны и без -s).
"""
import importlib.util
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure                      # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.backend_bases import MouseEvent           # noqa: E402
from sympy import E, Float, Integer, Rational, Symbol, exp, log, pi, sqrt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import function_visualizer as fv  # noqa: E402

ORIG_COMMIT = "2e2179d"       # исходный движок (до рефакторинга) — для регрессии
X = fv._X_SYM

# ─────────────────────────────────────────────────────────────
#  Базовые сцены (scratchpad/baseline_render.py) и ожидаемые подписи
# ─────────────────────────────────────────────────────────────
BASELINE = {
    "sinc":     ["sin(x)/x"],
    "user":     ["x^2 + 5/x - sqrt(x^2+15)"],
    "trig":     ["sin(x)", "cos(x)"],
    "rational": ["(x^2-1)/(x-1)", "1/(x-2)"],
    "circle":   ["x^2+y^2=9", "x=2"],
}

# Полный набор подписей точек '(x, y)' для каждой сцены в окне [-5,5]².
EXPECTED_LABELS = {
    "sinc": {'(π, 0)', '(-π, 0)', '(0, 1)',
             '(4.49, -0.22)', '(-4.49, -0.22)'},            # экстремумы — десятичные
    "user": {'(1.42, 1.41)',                                  # экстремум остаётся десятичным
             '(-2.57, 0)'},
    "trig": {'(π/4, √2/2)', '(-3π/4, -√2/2)', '(5π/4, -√2/2)',   # sin ∩ cos
             '(0, 0)', '(π, 0)', '(-π, 0)',                       # нули sin
             '(π/2, 0)', '(-π/2, 0)', '(3π/2, 0)', '(-3π/2, 0)',  # нули cos
             '(π/2, 1)', '(-π/2, -1)', '(3π/2, -1)', '(-3π/2, 1)',  # экстремумы sin
             '(0, 1)', '(π, -1)', '(-π, -1)'},                     # экстремумы cos
    "rational": {'(1, 2)',                                    # устранимый разрыв
                 '(-1, 0)', '(0, 1)', '(0, -1/2)',
                 '((1+√13)/2, (3+√13)/2)', '((1-√13)/2, (3-√13)/2)'},
    "circle": {'(3, 0)', '(-3, 0)', '(0, 3)', '(0, -3)',
               '(2, 0)', '(2, √5)', '(2, -√5)'},
}
EXPECTED_CURVES = {"sinc": 1, "user": 1, "trig": 2, "rational": 2, "circle": 0}

TIMINGS = []


# ─────────────────────────────────────────────────────────────
#  Вспомогательные функции
# ─────────────────────────────────────────────────────────────
_SETTINGS = ('FUNCS', 'FUNC_DOMAINS', 'CURVE_WIDTHS', 'CURVE_STYLES', 'FILL',
             'X_LIM_L', 'X_LIM_R', 'Y_LIM_B', 'Y_LIM_T',
             'GRID', 'X_GRID', 'Y_GRID',
             'ASIMP', 'DISC', 'EXTR', 'X_TAG', 'Y_TAG', 'INTER', 'SHOW_VALUES',
             'X_HIDE', 'Y_HIDE', 'FONT_SIZE', 'SYMBOLIC_MODE', 'MATHTEXT_LABELS')


def _reset_engine():
    fv.FUNCS = []
    fv.FUNC_DOMAINS = []
    fv.CURVE_WIDTHS = []
    fv.CURVE_STYLES = []
    fv.FILL = []
    fv.X_LIM_L, fv.X_LIM_R, fv.Y_LIM_B, fv.Y_LIM_T = -5, 5, -5, 5
    fv.GRID = fv.X_GRID = fv.Y_GRID = 1
    fv.ASIMP = fv.DISC = fv.EXTR = fv.X_TAG = fv.Y_TAG = fv.INTER = fv.SHOW_VALUES = 1
    fv.X_HIDE = fv.Y_HIDE = 0
    fv.FONT_SIZE = 10
    fv.FREE_TEXTS.clear()
    fv.SYMBOLIC_MODE = 'compute'
    fv.MATHTEXT_LABELS = False      # тесты сравнивают unicode-подписи
    fv.clear_symbolic_cache()
    fv.reset_annotation_offsets()


@pytest.fixture(autouse=True)
def engine():
    """Чистое состояние модуля перед каждым тестом; восстановление после."""
    saved = {k: getattr(fv, k) for k in _SETTINGS}
    saved_free = list(fv.FREE_TEXTS)
    _reset_engine()
    yield fv
    _reset_engine()
    fv.FREE_TEXTS[:] = saved_free
    for k, v in saved.items():
        setattr(fv, k, v)


def draw(funcs, fig=None, **settings):
    """plot_function(fig) с замером времени; возвращает dict результата + 'fig', 'time'."""
    for k, v in settings.items():
        setattr(fv, k, v)
    fv.FUNCS = list(funcs)
    if fig is None:
        fig = Figure(figsize=(6, 6))
    t0 = time.perf_counter()
    res = fv.plot_function(fig)
    res['time'] = time.perf_counter() - t0
    res['fig'] = fig
    return res


def report_timing(capsys, name, seconds, extra=''):
    TIMINGS.append((name, seconds))
    with capsys.disabled():
        print(f"\n[timing] {name}: {seconds:.2f} s {extra}".rstrip())


def point_labels(ax):
    """Тексты подписей точек '(x, y)' (ax.texts содержит и аннотации)."""
    return {t.get_text() for t in ax.texts if t.get_text().startswith('(')}


def point_coords(ax, nd=2):
    """Координаты подписанных точек (округлённые) — для сравнения движков."""
    return {(round(float(t.xy[0]), nd), round(float(t.xy[1]), nd))
            for t in ax.texts if t.get_text().startswith('(') and hasattr(t, 'xy')}


def curve_lines(ax):
    """Кривые y=f(x) (6000 узлов сетки); штриховка/оси/маркеры короче."""
    return [ln for ln in ax.lines if len(ln.get_xdata()) > 2000]


def hatch_lines(ax):
    """Линии штриховки (стили 0/1): linewidth 0.6, без маркера."""
    return [ln for ln in ax.lines
            if abs(ln.get_linewidth() - 0.6) < 1e-9 and ln.get_marker() == 'None']


def hatch_dots(ax):
    """Точечная штриховка (стиль 2): маркер '.', linewidth 0."""
    return [ln for ln in ax.lines if ln.get_marker() == '.' and ln.get_linewidth() == 0]


def n_callbacks(fig):
    return sum(len(v) for v in fig.canvas.callbacks.callbacks.values())


def zero_verifier(expr):
    return lambda c: fv._is_zero_at(expr, X, c)


# ═════════════════════════════════════════════════════════════
#  1. Базовые сцены — режим compute
# ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", list(BASELINE))
def test_baseline_case(name, capsys):
    res = draw(BASELINE[name])
    ax = res['ax']
    report_timing(capsys, f"baseline {name}", res['time'])

    assert res['errors'] == {}
    assert res['pending'] is False
    assert ax in res['fig'].axes and len(res['fig'].axes) == 1
    assert point_labels(ax) == EXPECTED_LABELS[name]
    assert len(curve_lines(ax)) == EXPECTED_CURVES[name]
    assert res['time'] < 60

    if name == "rational":
        # вертикальная асимптота x=2 (пунктир) и «дырка» (1, 2) белым кружком
        assert any(list(ln.get_xdata()) == [2.0, 2.0] and ln.get_linestyle() == '--'
                   for ln in ax.lines)
        holes = [(float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
                 for ln in ax.lines if ln.get_markerfacecolor() == 'white']
        assert holes == [(1.0, 2.0)]
    if name == "circle":
        assert len(ax.collections) >= 1            # contour окружности
        assert any(list(ln.get_xdata()) == [2, 2] and ln.get_linewidth() == 1.8
                   for ln in ax.lines)             # вертикаль x = 2
    if name == "user":
        assert not any(ch in lbl for lbl in point_labels(ax) for ch in 'π√e')


@pytest.mark.parametrize("funcs, limits, expected", [
    (["x^2-2"],            (-5, 5), {'(√2, 0)', '(-√2, 0)', '(0, -2)'}),
    (["e^x-5e"],           (-5, 5), {'(1+ln(5), 0)'}),
    (["x=2pi", "x = e"],   (-1, 7), {'(2π, 0)', '(e, 0)'}),      # вертикали: точный текст
    (["x^2+y^2=9", "x=2"], (-5, 5), {'(2, √5)', '(2, -√5)'}),    # вертикаль ∩ окружность
])
def test_exact_labels(funcs, limits, expected, capsys):
    res = draw(funcs, X_LIM_L=limits[0], X_LIM_R=limits[1])
    report_timing(capsys, f"exact {funcs}", res['time'])
    assert res['errors'] == {}
    assert expected <= point_labels(res['ax'])


@pytest.mark.parametrize("funcs, expected, forbidden", [
    (["x-3.1416"],   {'(3.14, 0)'},                 'π'),
    (["x^2-2.0001"], {'(1.41, 0)', '(-1.41, 0)'},   '√'),
])
def test_near_miss_values_stay_decimal(funcs, expected, forbidden):
    """Значения, лишь близкие к π / √2, НЕ выдаются за точные."""
    res = draw(funcs)
    labels = point_labels(res['ax'])
    assert res['errors'] == {}
    assert expected <= labels
    assert not any(forbidden in lbl for lbl in labels)


# ═════════════════════════════════════════════════════════════
#  2. Регрессия относительно исходного движка
# ═════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def orig_engine(tmp_path_factory):
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "show", f"{ORIG_COMMIT}:function_visualizer.py"],
            capture_output=True, timeout=30, check=True).stdout
    except Exception as exc:          # нет git / коммита (неполный клон) — пропускаем
        pytest.skip(f"original engine unavailable: {exc}")
    path = tmp_path_factory.mktemp("orig") / "orig_engine.py"
    path.write_bytes(out)
    spec = importlib.util.spec_from_file_location("orig_engine_2e2179d", str(path))
    mod = importlib.util.module_from_spec(spec)
    family = list(fv.plt.rcParams['font.family'])
    spec.loader.exec_module(mod)
    mod.plt.show = lambda *a, **k: None   # standalone-движок открывал окно
    yield mod
    fv.plt.close('all')
    fv.plt.rcParams['font.family'] = family


@pytest.mark.parametrize("name", list(BASELINE))
def test_baseline_points_match_original_engine(name, orig_engine, capsys):
    """Те же подписанные точки (с точностью до 2 знаков), что у старого движка."""
    new = draw(BASELINE[name])
    orig = orig_engine
    orig.FUNCS = list(BASELINE[name])
    orig.FUNC_DOMAINS, orig.CURVE_WIDTHS, orig.CURVE_STYLES, orig.FILL = [], [], [], []
    orig.X_LIM_L, orig.X_LIM_R, orig.Y_LIM_B, orig.Y_LIM_T = -5, 5, -5, 5
    t0 = time.perf_counter()
    orig.plot_function()
    t_orig = time.perf_counter() - t0
    try:
        old_ax = orig.plt.gcf().axes[0]
        report_timing(capsys, f"original-engine {name}", t_orig,
                      f"(new engine {new['time']:.2f} s)")
        assert point_coords(new['ax']) == point_coords(old_ax)
        assert len(curve_lines(new['ax'])) == len(curve_lines(old_ax))
    finally:
        orig.plt.close('all')


# ═════════════════════════════════════════════════════════════
#  3. Кэш символьных вычислений и отложенные задания
# ═════════════════════════════════════════════════════════════

def test_sym_cached_semantics():
    calls = []

    def job():
        calls.append(1)
        return 42

    assert issubclass(fv.SymbolicPending, Exception)
    # compute: считается один раз, далее — из кэша
    assert fv.sym_cached(('t', 'a'), job) == 42
    assert fv.sym_cached(('t', 'a'), job) == 42
    assert calls.count(1) == 1

    # исключение тоже кэшируется (не пересчитывается при каждой перерисовке)
    def bad():
        calls.append('bad')
        raise NotImplementedError("no closed form")
    for _ in range(2):
        with pytest.raises(NotImplementedError):
            fv.sym_cached(('t', 'bad'), bad)
    assert calls.count('bad') == 1

    # cached: незакэшированный ключ откладывается, fn НЕ вызывается
    fv.SYMBOLIC_MODE = 'cached'
    assert fv.sym_cached(('t', 'a'), job) == 42          # попадание в кэш работает и тут
    with pytest.raises(fv.SymbolicPending):
        fv.sym_cached(('t', 'b'), job)
    with pytest.raises(fv.SymbolicPending):
        fv.sym_cached(('t', 'b'), job)                   # дубликат ключа
    assert calls.count(1) == 1
    jobs = fv.take_pending_jobs()
    assert [k for k, _ in jobs] == [('t', 'b')]          # дедупликация по ключу
    assert fv.take_pending_jobs() == []                  # очередь опустошена

    # run_jobs из рабочего потока → результат доступен в режиме cached
    th = threading.Thread(target=fv.run_jobs, args=(jobs,))
    th.start()
    th.join(30)
    assert not th.is_alive()
    assert fv.sym_cached(('t', 'b'), job) == 42
    assert calls.count(1) == 2
    assert fv.take_pending_jobs() == []

    fv.clear_symbolic_cache()
    with pytest.raises(fv.SymbolicPending):
        fv.sym_cached(('t', 'a'), job)


@pytest.mark.parametrize("name", list(BASELINE) + ["x2m2"])
def test_cached_pending_cycle(name, capsys):
    funcs = BASELINE.get(name, ["x^2-2"])
    ref = point_labels(draw(funcs)['ax'])
    ref_curves = len(curve_lines(draw(funcs)['ax']))

    fv.clear_symbolic_cache()
    fv.SYMBOLIC_MODE = 'cached'
    fig = Figure(figsize=(6, 6))
    res = draw(funcs, fig=fig)
    t_first = res['time']
    assert res['pending'] is True
    assert res['errors'] == {}
    assert len(curve_lines(res['ax'])) == ref_curves      # кривые есть сразу (численно)

    cycles = 0
    while res['pending'] and cycles < 5:
        jobs = fv.take_pending_jobs()
        assert jobs, "pending=True, но очередь заданий пуста"
        fv.run_jobs(jobs)
        res = draw(funcs, fig=fig)
        cycles += 1
    report_timing(capsys, f"cached {name}", t_first, f"(first draw; {cycles} cycle(s))")
    assert res['pending'] is False
    assert res['errors'] == {}
    assert point_labels(res['ax']) == ref
    assert fv.take_pending_jobs() == []
    assert len(fig.axes) == 1


def test_cached_draw_never_runs_heavy_sympy(monkeypatch):
    """В режиме cached поток GUI не ждёт sympy: solveset/limit/… не вызываются."""
    funcs = ["(x^2-1)/(x-1)", "sin(x)", "x^2+y^2=9"]
    ref = point_labels(draw(funcs)['ax'])

    fv.clear_symbolic_cache()
    fv.SYMBOLIC_MODE = 'cached'

    def boom(*a, **k):
        raise AssertionError("heavy sympy call in cached mode")
    for nm in ('solveset', 'singularities', 'limit', 'simplify', 'nsimplify'):
        monkeypatch.setattr(fv, nm, boom)
    fig = Figure(figsize=(6, 6))
    res = draw(funcs, fig=fig)
    assert res['pending'] is True and res['errors'] == {}
    assert len(curve_lines(res['ax'])) == 2
    monkeypatch.undo()

    # задания досчитываются в рабочем потоке, как в GUI
    cycles = 0
    while res['pending'] and cycles < 5:
        jobs = fv.take_pending_jobs()
        th = threading.Thread(target=fv.run_jobs, args=(jobs,))
        th.start()
        th.join(120)
        assert not th.is_alive()
        res = draw(funcs, fig=fig)
        cycles += 1
    assert res['pending'] is False
    assert point_labels(res['ax']) == ref
    assert '(1, 2)' in ref and '(π, 0)' in ref


# ═════════════════════════════════════════════════════════════
#  4. Ошибки ввода, пустые строки, заливка
# ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", ['compute', 'cached'])
def test_bad_inputs_reported_other_curves_still_drawn(mode):
    fv.SYMBOLIC_MODE = mode
    res = draw(["x^", "sin(", "x^2"])
    ax = res['ax']
    assert set(res['errors']) == {0, 1}
    for msg in res['errors'].values():
        assert isinstance(msg, str) and 0 < len(msg) <= 80
    curves = curve_lines(ax)
    assert len(curves) == 1
    assert curves[0].get_color() == fv.CURVE_COLORS[2]   # цвет по индексу строки
    assert '(0, 0)' in point_labels(ax)


def test_error_messages_for_other_bad_inputs():
    funcs = ["foo(x)", "x+y=zz", "x=abc", "x=x+1", "y=y", "log(x, 1)", "x"]
    res = draw(funcs)
    assert set(res['errors']) == {0, 1, 2, 3, 4, 5}
    assert 'foo' in res['errors'][0]
    assert 'zz' in res['errors'][1]                      # неявная кривая: ошибка не глотается
    assert all(isinstance(m, str) and m for m in res['errors'].values())
    assert len(curve_lines(res['ax'])) == 1


def test_empty_entries_skipped_with_stable_indices():
    funcs = ["", "x^2", None, "x"]
    res = draw(funcs)
    assert res['errors'] == {}
    assert [ln.get_color() for ln in curve_lines(res['ax'])] == \
        [fv.CURVE_COLORS[1], fv.CURVE_COLORS[3]]

    # заливка между индексами 1 и 3 (x² и x) — индексы не «сползают»
    res = draw(funcs, FILL=[(1, 3, -1, 1, 0)])
    assert res['errors'] == {}
    hl = hatch_lines(res['ax'])
    assert len(hl) >= 5
    for ln in hl:
        xs = np.asarray(ln.get_xdata(), float)
        ys = np.asarray(ln.get_ydata(), float)
        assert xs.min() >= -1 - 1e-9 and xs.max() <= 1 + 1e-9
        assert np.all(ys >= np.minimum(xs ** 2, xs) - 1e-9)
        assert np.all(ys <= np.maximum(xs ** 2, xs) + 1e-9)

    # заливка от пустой строки — тихо пропускается
    res = draw(funcs, FILL=[(0, 1, -1, 1, 0), (2, "x", -1, 1, 1)])
    assert res['errors'] == {}
    assert hatch_lines(res['ax']) == []


@pytest.mark.parametrize("style", [0, 1, 2])
def test_fill_styles_produce_hatching(style):
    res = draw(["x^2", "x"], FILL=[(0, 1, -1, 1, style)])
    ax = res['ax']
    assert res['errors'] == {}
    artists = hatch_dots(ax) if style == 2 else hatch_lines(ax)
    assert len(artists) >= 5
    if style == 2:
        assert hatch_lines(ax) == []
    else:
        assert hatch_dots(ax) == []
    tol = 1e-9 if style != 2 else 0.01    # точки стоят на решётке между узлами сетки
    n_pts = 0
    for ln in artists:
        xs = np.asarray(ln.get_xdata(), float)
        ys = np.asarray(ln.get_ydata(), float)
        n_pts += xs.size
        assert xs.min() >= -1 - 1e-9 and xs.max() <= 1 + 1e-9
        assert np.all(ys >= np.minimum(xs ** 2, xs) - tol)
        assert np.all(ys <= np.maximum(xs ** 2, xs) + tol)
    assert n_pts >= 50
    # границы заливки x=-1 и x=1
    borders = sorted(float(ln.get_xdata()[0]) for ln in ax.lines
                     if abs(ln.get_linewidth() - 1.2) < 1e-9)
    assert borders == [-1.0, 1.0]
    assert len(curve_lines(ax)) == 2


def test_fill_to_x_axis():
    res = draw(["x^2"], FILL=[(0, "x", -2, 2, 0)])
    hl = hatch_lines(res['ax'])
    assert len(hl) >= 5
    for ln in hl:
        xs = np.asarray(ln.get_xdata(), float)
        ys = np.asarray(ln.get_ydata(), float)
        assert np.all(ys >= -1e-9) and np.all(ys <= xs ** 2 + 1e-9)


# ═════════════════════════════════════════════════════════════
#  5. nroot / cbrt, make_y_array, кэш функций
# ═════════════════════════════════════════════════════════════

def test_nroot_real_cube_root():
    expr, _raw, f = fv.build_numpy_func("nroot(x, 3)")
    assert f(-8.0) == pytest.approx(-2.0)
    assert np.allclose(f(np.array([-8.0, 8.0, 27.0])), [-2.0, 2.0, 3.0])
    assert expr.subs(X, -8) == -2
    expr, _raw, f = fv.build_numpy_func("nroot(-8, 3)")
    assert expr == Integer(-2)
    assert float(f(1.0)) == pytest.approx(-2.0)
    _expr, _raw, f = fv.build_numpy_func("nroot(x, 2)")
    assert f(16.0) == pytest.approx(4.0)
    _expr, _raw, f = fv.build_numpy_func("root(x, 3)")
    assert f(-8.0) == pytest.approx(-2.0)
    res = draw(["nroot(x, 3)"])
    assert res['errors'] == {} and '(0, 0)' in point_labels(res['ax'])


def test_cbrt_alias_accepted():
    # cbrt — лямбда в local_dict; parse_expr(evaluate=False) передаёт ей evaluate=…
    _expr, _raw, f = fv.build_numpy_func("cbrt(x)")
    assert f(-8.0) == pytest.approx(-2.0)
    res = draw(["cbrt(x)"])
    assert res['errors'] == {}


@pytest.mark.parametrize("func_str", ["3", "abs(x)", "sin(x)/x"])
def test_make_y_array_vectorized_matches_loop(func_str):
    _expr, _raw, f = fv.build_numpy_func(func_str)
    xs = np.arange(-500, 501) / 100.0          # содержит ровно 0.0 (sin(x)/x → NaN)
    ys = fv.make_y_array(f, xs, [], -5, 5)
    loop = np.array([fv._safe_val(f, float(xi)) for xi in xs], dtype=float)
    assert ys.shape == xs.shape and ys.dtype == np.float64
    assert np.array_equal(np.isnan(ys), np.isnan(loop))
    assert np.allclose(ys, loop, rtol=1e-12, atol=0, equal_nan=True)
    if func_str == "sin(x)/x":
        assert np.isnan(ys[xs == 0.0]).all()
    else:
        assert not np.isnan(ys).any()


def test_make_y_array_masks_and_fallback():
    _expr, _raw, f = fv.build_numpy_func("1/x")
    xs = np.linspace(-5, 5, 6000)
    ys = fv.make_y_array(f, xs, [0.0], -5, 5)
    assert np.isnan(ys[np.abs(xs) < 0.05]).all()        # окрестность разрыва
    assert np.isnan(ys[np.abs(1 / xs) > 200]).all()       # выбросы > 20·y_span
    assert np.isfinite(ys[np.abs(xs) > 1]).all()

    # невекторизуемая функция (math.sin на массиве бросает) → поточечный запасной путь
    g = lambda t: math.sin(float(t))                     # noqa: E731
    ys = fv.make_y_array(g, xs, [], -5, 5)
    assert np.allclose(ys, np.sin(xs), rtol=1e-12, atol=0)


def test_build_numpy_func_is_memoized():
    a = fv.build_numpy_func("x^2+1")
    b = fv.build_numpy_func("x^2+1")
    assert a[2] is b[2]
    fv.clear_symbolic_cache()
    c = fv.build_numpy_func("x^2+1")
    assert c[2] is not a[2]
    with pytest.raises(Exception):
        fv.build_numpy_func("x^")                      # ошибки разбора не кэшируются


# ═════════════════════════════════════════════════════════════
#  6. Подписки на canvas и положение подписей
# ═════════════════════════════════════════════════════════════

def test_canvas_callbacks_do_not_grow_over_redraws():
    fig = Figure(figsize=(6, 6))
    counts, drags = [], []
    for _ in range(6):
        res = draw(["sin(x)", "cos(x)"], fig=fig)
        counts.append(n_callbacks(fig))
        drags.append(len(fv._ACTIVE_DRAGGABLES))
        assert len(fig.axes) == 1
    assert counts[-1] == counts[0], counts
    assert drags[-1] == drags[0] == len(point_labels(res['ax']))
    assert all(d.press is None for d in fv._ACTIVE_DRAGGABLES)


def test_annotation_offsets_applied_and_persist():
    fig = Figure(figsize=(6, 6))
    res = draw(["x^2-2"], fig=fig)
    ann = next(t for t in res['ax'].texts if t.get_text() == '(√2, 0)')
    home = tuple(map(float, ann.get_position()))

    key = ('(√2, 0)', round(math.sqrt(2), 3), 0.0)
    fv.ANNOTATION_OFFSETS[key] = (30.0, -20.0)          # пункты относительно home
    res = draw(["x^2-2"], fig=fig)
    ann = next(t for t in res['ax'].texts if t.get_text() == '(√2, 0)')
    moved = tuple(map(float, ann.get_position()))
    dx, dy = moved[0] - home[0], moved[1] - home[1]
    unit = 10.0 / (6 * 72)             # ≈ единиц данных на пункт при фигуре 6"
    assert 0.5 * 30 * unit < dx < 2.0 * 30 * unit
    assert -2.0 * 20 * unit < dy < -0.5 * 20 * unit
    # другие подписи не трогаем
    other = next(t for t in res['ax'].texts if t.get_text() == '(-√2, 0)')
    assert float(other.get_position()[1]) == pytest.approx(home[1])

    res = draw(["x^2-2"], fig=fig)                        # повторная перерисовка — не прыгает
    ann = next(t for t in res['ax'].texts if t.get_text() == '(√2, 0)')
    assert tuple(map(float, ann.get_position())) == pytest.approx(moved)

    fv.reset_annotation_offsets()
    assert fv.ANNOTATION_OFFSETS == {}
    res = draw(["x^2-2"], fig=fig)
    ann = next(t for t in res['ax'].texts if t.get_text() == '(√2, 0)')
    assert tuple(map(float, ann.get_position())) == pytest.approx(home)


def test_drag_release_stores_offset():
    """Перетаскивание подписи мышью (синтетические события) запоминается."""
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    res = draw(["x^2-2"], fig=fig)
    ax = res['ax']
    canvas.draw()
    ann = next(t for t in ax.texts if t.get_text() == '(√2, 0)')
    before = tuple(map(float, ann.get_position()))
    bb = ann.get_window_extent(canvas.get_renderer())
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    assert ax.contains_point((cx, cy))

    def fire(kind, x, y):
        canvas.callbacks.process(kind, MouseEvent(kind, canvas, x, y, button=1))
    fire('button_press_event', cx, cy)
    fire('motion_notify_event', cx + 36, cy - 36)
    fire('button_release_event', cx + 36, cy - 36)

    key = ('(√2, 0)', round(math.sqrt(2), 3), 0.0)
    assert key in fv.ANNOTATION_OFFSETS
    dx, dy = fv.ANNOTATION_OFFSETS[key]
    assert dx > 0 and dy < 0
    after = tuple(map(float, ann.get_position()))
    assert after[0] > before[0] and after[1] < before[1]

    res = draw(["x^2-2"], fig=fig)                        # после перерисовки — там же
    ann2 = next(t for t in res['ax'].texts if t.get_text() == '(√2, 0)')
    assert tuple(map(float, ann2.get_position())) == pytest.approx(after, abs=1e-9)
    other = next(t for t in res['ax'].texts if t.get_text() == '(-√2, 0)')
    assert float(other.get_position()[1]) == pytest.approx(before[1])


# ═════════════════════════════════════════════════════════════
#  7. identify_value / exact_candidates / fmt_sym
# ═════════════════════════════════════════════════════════════

def test_identify_value_accepts_true_constants():
    assert fv.identify_value(math.pi) == pi
    assert fv.identify_value(math.sqrt(2)) == sqrt(2)
    assert fv.identify_value(0.5) == Rational(1, 2)
    assert fv.identify_value(0.0) == 0
    assert fv.identify_value(float('nan')) is None
    e = fv.build_numpy_func("x^2-2")[0]
    assert fv.identify_value(math.sqrt(2), verify=zero_verifier(e)) == sqrt(2)
    assert fv.identify_value(math.pi, verify=lambda c: False) is None   # verify — ворота


def test_identify_value_rejects_near_misses():
    e1 = fv.build_numpy_func("x-3.1416")[0]
    assert fv.identify_value(3.1416, verify=zero_verifier(e1)) is None
    assert fv.identify_value(3.1416) is None
    e2 = fv.build_numpy_func("x^2-2.0001")[0]
    v = math.sqrt(2.0001)
    assert fv.identify_value(v, verify=zero_verifier(e2)) is None
    assert fv.identify_value(v) is None
    # кандидат принимается только при |c − v| ≤ 1e-6·max(1,|v|)
    assert fv.identify_value(1.41421356, candidates=[sqrt(2)]) == sqrt(2)
    assert fv.identify_value(1.42, candidates=[sqrt(2)]) is None
    assert fv.identify_value(1.42, candidates=[(1.4142135623730951, sqrt(2))]) is None


def test_exact_candidates():
    e = fv.build_numpy_func("x^2-2")[0]
    cands = fv.exact_candidates(e, X, 'roots', -5, 5)
    assert sorted(c for c, _ in cands) == pytest.approx([-math.sqrt(2), math.sqrt(2)])
    assert {str(s) for _, s in cands} == {'-sqrt(2)', 'sqrt(2)'}
    s = fv.build_numpy_func("sin(x)")[0]
    roots = fv.exact_candidates(s, X, 'roots', -5, 5)          # ImageSet над ℤ → окно
    assert {str(c) for _, c in roots} == {'-pi', '0', 'pi'}
    crit = fv.exact_candidates(s, X, 'critical', -5, 5)
    assert {str(c) for _, c in crit} == {'-3*pi/2', '-pi/2', 'pi/2', '3*pi/2'}
    fv.SYMBOLIC_MODE = 'cached'
    fv.clear_symbolic_cache()
    with pytest.raises(fv.SymbolicPending):
        fv.exact_candidates(e, X, 'roots', -5, 5)


@pytest.mark.parametrize("expr, text", [
    (pi / 2, 'π/2'), (2 * pi / 3, '2π/3'), (-pi / 2, '-π/2'), (pi, 'π'),
    (sqrt(2) / 2, '√2/2'), (-sqrt(3) / 2, '-√3/2'), (2 * sqrt(3), '2√3'),
    (Rational(1, 3), '1/3'), (Rational(-1, 2), '-1/2'), (Integer(3), '3'), (Integer(0), '0'),
    (E, 'e'), (2 * E, '2e'), (exp(2), 'e²'), (E ** 2, 'e²'),
    (1 + sqrt(2), '1+√2'), ((1 + sqrt(13)) / 2, '(1+√13)/2'),
    (log(2), 'ln(2)'), (1 + log(5), '1+ln(5)'),
])
def test_fmt_sym(expr, text):
    assert fv.fmt_sym(expr) == text


@pytest.mark.parametrize("expr", [Float(1.5), Symbol('x'), 1.0 * sqrt(2), pi ** pi, 'x'])
def test_fmt_sym_returns_none_for_non_exact(expr):
    assert fv.fmt_sym(expr) is None


def test_fmt_exact_or_falls_back_to_decimal():
    assert fv.fmt_exact_or(math.sqrt(2), sqrt(2)) == '√2'
    assert fv.fmt_exact_or(1.42, sqrt(2), dec=lambda v: f'{v:.2f}') == '1.42'   # не согласуется
    assert fv.fmt_exact_or(1.5, None, dec=lambda v: f'{v:.2f}') == '1.50'


# ═════════════════════════════════════════════════════════════
#  8. Прочее: standalone, домен, сводка времени
# ═════════════════════════════════════════════════════════════

def test_standalone_plot_function(monkeypatch):
    shown = []
    monkeypatch.setattr(fv.plt, 'show', lambda *a, **k: shown.append(1))
    fv.FUNCS = ["x^2-2"]
    try:
        res = fv.plot_function()
        assert shown == [1]
        assert res['errors'] == {} and res['pending'] is False
        assert {'(√2, 0)', '(-√2, 0)', '(0, -2)'} <= point_labels(res['ax'])
    finally:
        fv.plt.close('all')


def test_function_domain_limits_curve():
    res = draw(["x^2"], FUNC_DOMAINS=[(0, 2)])
    assert res['errors'] == {}
    ln = curve_lines(res['ax'])[0]
    xs = np.asarray(ln.get_xdata(), float)
    ys = np.asarray(ln.get_ydata(), float)
    assert not np.isfinite(ys[(xs < 0) | (xs > 2)]).any()
    assert np.isfinite(ys[(xs > 0.01) & (xs < 1.99)]).all()


def test_zz_timing_summary(capsys):
    """Сводка замеров (последний тест модуля)."""
    with capsys.disabled():
        print("\n[timing] summary:")
        for name, s in TIMINGS:
            print(f"    {name:48s} {s:6.2f} s")
    assert TIMINGS


def test_fmt_sym_log_quotient_as_log_base():
    import sympy as sp
    assert fv.fmt_sym(sp.log(3) / sp.log(2)) == "log₂(3)"
    assert fv.fmt_sym(sp.Rational(1, 2) + sp.log(3) / sp.log(2)) == "1/2+log₂(3)"
    assert fv.fmt_sym(sp.log(5)) == "ln(5)"


def test_fmt_sym_tex_forms():
    assert fv.fmt_sym_tex(sqrt(5)) == r"\sqrt{5}"
    assert fv.fmt_sym_tex((sqrt(21) - 1) / 2) == r"\frac{\sqrt{21}-1}{2}"
    assert fv.fmt_sym_tex(pi / 2) == r"\frac{\pi}{2}"
    assert fv.fmt_sym_tex(-pi / 2) == r"-\frac{\pi}{2}"
    assert fv.fmt_sym_tex(2 * pi / 3) == r"\frac{2\pi}{3}"
    assert fv.fmt_sym_tex(Rational(-1, 2)) == r"-\frac{1}{2}"
    assert fv.fmt_sym_tex(sqrt(2) / 2) == r"\frac{\sqrt{2}}{2}"
    assert fv.fmt_sym_tex(log(3) / log(2)) == r"\log_{2}(3)"
    assert fv.fmt_sym_tex(exp(2)) == r"e^{2}"
    assert fv.fmt_sym_tex(Integer(2) ** Rational(1, 3)) == r"\sqrt[3]{2}"
    assert fv.fmt_num(math.pi / 2, tex=True) == r"\frac{\pi}{2}"
    assert fv.fmt_num(-4, tex=True) == "-4"


def test_mathtext_labels_render():
    """В режиме mathtext подписи с корнями/дробями — $…$, десятичные — обычный текст с минусом; всё рисуется."""
    fv.MATHTEXT_LABELS = True
    res = draw(["x^2-5", "sin(x)", "cos(x)", "x^2 + 5/x - sqrt(x^2+15)"])
    assert res['errors'] == {}
    texts = [t.get_text() for t in res['ax'].texts]
    assert any(t == r"$(\sqrt{5},\ 0)$" for t in texts), texts
    assert any(t == r"$(\frac{\pi}{4},\ \frac{\sqrt{2}}{2})$" for t in texts), texts
    assert any(t == "(1.42, 1.41)" for t in texts), texts          # десятичные — без $
    assert any(t == "\u22124" for t in texts), texts                 # деление -4 с настоящим минусом
    assert "$x$" in texts and "$y$" in texts
    res['fig'].canvas.draw()                                      # mathtext разбирается без ошибок
    assert fv.math_label(r"\bad{") == r"\bad{"                      # некорректный фрагмент → обычный текст


def test_font_presets_apply():
    import matplotlib.pyplot as plt
    for name in ("schola", "times", "latex", "serif", "sans", "century"):
        fv.apply_font_preset(name)
        assert fv.GRAPH_FONT == name
        assert plt.rcParams['mathtext.fontset'] in ("stix", "cm", "dejavuserif", "dejavusans", "custom")
    fv.apply_font_preset("nonsense")
    assert fv.GRAPH_FONT == "schola"           # неизвестное имя → пресет по умолчанию
    fv.apply_font_preset("schola")


def test_effective_limits_extend_to_canvas():
    fv.X_LIM_L, fv.X_LIM_R, fv.Y_LIM_B, fv.Y_LIM_T = -5, 5, -5, 5
    assert fv.effective_limits(600, 600) == (-5, 5, -5, 5)
    xl, xr, yb, yt = fv.effective_limits(1200, 600)        # вдвое шире — x расширяется вдвое
    assert (xl, xr) == pytest.approx((-10, 10)) and (yb, yt) == (-5, 5)
    xl, xr, yb, yt = fv.effective_limits(600, 900)         # выше — расширяется y
    assert (xl, xr) == (-5, 5) and (yb, yt) == pytest.approx((-7.5, 7.5))
    # при построении на широкой фигуре пределы осей расширены (по области
    # построения = фигура минус поля PLOT_MARGIN_PX), а глобалы не тронуты
    fig = Figure(figsize=(12, 6), dpi=100)
    FigureCanvasAgg(fig)
    res = draw(["sin(x)"], fig=fig)
    m = fv.PLOT_MARGIN_PX
    exp = fv.effective_limits(1200 - 2 * m, 600 - 2 * m)
    assert res['ax'].get_xlim() == pytest.approx(exp[:2])
    assert res['ax'].get_ylim() == pytest.approx((-5, 5))
    assert (fv.X_LIM_L, fv.X_LIM_R) == (-5, 5)
    # кривая и подписи заполняют расширенную область
    xs = res['ax'].lines[0].get_xdata()
    assert len(xs) > 100 and xs.min() < -9 and xs.max() > 9 or any(
        len(l.get_xdata()) > 100 and l.get_xdata().min() < -9 for l in res['ax'].lines)


def test_plot_margins_and_axis_labels():
    """Поля 20 px вокруг области построения; «x» справа от стрелки оси X,
    «y» над стрелкой оси Y — в полях; подписи осей не перетаскиваются."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    fv.AXIS_LABELS['x'].update({'dx': 7.0, 'dy': -3.0})       # «старый» сдвиг игнорируется
    res = draw(["x^2"], fig=fig)
    ax = res['ax']
    m = fv.PLOT_MARGIN_PX
    pos = ax.get_position()
    assert pos.x0 * 800 == pytest.approx(m, abs=0.6) and (1 - pos.x1) * 800 == pytest.approx(m, abs=0.6)
    assert pos.y0 * 600 == pytest.approx(m, abs=0.6) and (1 - pos.y1) * 600 == pytest.approx(m, abs=0.6)
    assert fv.plot_box_px(fig) == (m, m, 800 - m, 600 - m)
    fig.canvas.draw()
    mgr = fv._active_axis_label_manager
    lx, ly = mgr.artists['x'], mgr.artists['y']
    ax_bb = ax.get_window_extent()
    bx = lx.get_window_extent(); by = ly.get_window_extent()
    assert bx.x0 >= ax_bb.x1 - 0.5 and bx.x1 <= 800 + 0.5          # «x» в правом поле
    assert by.y0 >= ax_bb.y1 - 0.5 and by.y1 <= 600 + 0.5          # «y» в верхнем поле
    assert fv.AXIS_LABELS['x']['dx'] == 0.0 and fv.AXIS_LABELS['x']['dy'] == 0.0
    # перетаскивание отключено: on_press левой кнопкой ничего не захватывает
    class Ev:
        x, y, button, guiEvent = bx.x0 + 1, bx.y0 + 1, 1, None
    mgr.on_press(Ev()); assert mgr.drag is None
    Ev.x += 30; mgr.on_motion(Ev())
    assert fv.AXIS_LABELS['x']['dx'] == 0.0
    # кривая не выходит за область построения: все точки внутри пределов осей
    xs = ax.lines[0].get_xdata()
    assert xs.min() >= ax.get_xlim()[0] - 1e-9 and xs.max() <= ax.get_xlim()[1] + 1e-9


def test_edge_ticks_quarter_step_rule():
    """Линии сетки и деления у краёв: рисуются, если отстоят от края не меньше чем
    на четверть шага; ближе - ни линии, ни подписи (одно правило для обоих)."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    fv.EXTEND_TO_CANVAS = False
    try:
        res = draw(["x^2"], fig=fig, X_LIM_L=-6.967, X_LIM_R=5.843, Y_LIM_B=-4.34, Y_LIM_T=5.66,
                   X_GRID=1, Y_GRID=1)
        ax = res['ax']
        xg = sorted(ax.xaxis.get_majorticklocs()); yg = sorted(ax.yaxis.get_majorticklocs())
        assert -6 in xg and 5 in xg and 5 in yg and -4 in yg
        texts = {t.get_text().replace('$', '').replace('−', '-') for t in ax.texts}
        assert {'-6', '5', '-4'} <= texts, texts
        # Ближе четверти шага к краю: ни линии сетки, ни подписи
        res = draw(["x^2"], fig=fig, X_LIM_L=-5.0, X_LIM_R=5.2, Y_LIM_B=-5, Y_LIM_T=5, X_GRID=1, Y_GRID=1)
        ax = res['ax']
        xg = sorted(ax.xaxis.get_majorticklocs()); yg = sorted(ax.yaxis.get_majorticklocs())
        assert 5 not in xg and -5 not in xg and 5 not in yg and -5 not in yg
        assert not [t for t in ax.texts if t.get_text().replace('$', '') == '5']
        # Ровно четверть шага и больше - рисуется
        res = draw(["x^2"], fig=fig, X_LIM_L=-5.0, X_LIM_R=5.3, Y_LIM_B=-5.25, Y_LIM_T=5, X_GRID=1, Y_GRID=1)
        ax = res['ax']
        assert 5 in ax.xaxis.get_majorticklocs() and -5 in ax.yaxis.get_majorticklocs()
    finally:
        fv.EXTEND_TO_CANVAS = True


def test_fill_area_label():
    """Число площади у заливки: точное «S = 1/3» для ∫₀¹ x² dx, численное при смене знака, без галочки — нет."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    fv.SYMBOLIC_MODE = 'compute'
    # f0 = x², заливка до оси X на [0, 1], площадь показывать
    res = draw(["x^2"], fig=fig, FILL=[(0, "x", 0.0, 1.0, 0, True, 0.02, True, "0", "1")])
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels and ("1/3" in labels[0]), labels
    # разность меняет знак (x на [-1, 1]) — только численно: ∫|x| = 1
    res = draw(["x"], fig=fig, FILL=[(0, "x", -1.0, 1.0, 0, True, 0.02, True, "-1", "1")])
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels and "≈" in labels[0] and "1" in labels[0], labels
    # между двумя функциями: x² и x на [0, 1] → 1/6
    res = draw(["x^2", "x"], fig=fig, FILL=[(0, 1, 0.0, 1.0, 0, True, 0.02, True, "0", "1")])
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels and "1/6" in labels[0], labels
    # галочка снята — подписи нет, но площадь посчитана (для строки заливки)
    res = draw(["x^2"], fig=fig, FILL=[(0, "x", 0.0, 1.0, 0, True, 0.02, False, "0", "1")])
    assert not [t for t in res['ax'].texts if t.get_text().startswith("S")]
    area, exact = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(1 / 3, abs=1e-4) and str(exact) == "1/3"
    assert fv.area_text_plain(area, exact) == "S = 1/3"
    assert fv.area_text_plain(2.3456, None) == "S ≈ 2.346"


def test_exact_readability_and_mixed_numbers():
    """Читаемые точные формы остаются, громоздкие → десятичные (2 знака); дроби > 1 — смешанные числа."""
    from sympy import Rational, sqrt, pi, log
    R = fv.exact_is_readable
    assert R(Rational(20, 3)) and R(sqrt(2) / 2) and R((1 + sqrt(13)) / 2) and R(3 * pi / 2) and R(2 * sqrt(5) / 5)
    assert R(log(3) / log(2)) and R(1 + sqrt(2)) and R(Rational(125, 3))
    assert not R(Rational(141, 50)) and not R(Rational(-141, 83)) and not R(Rational(18795, 11))
    assert not R(51 * sqrt(2) / 2) and not R(5813 * sqrt(3) / 2) and not R(sqrt(105))
    assert fv.fmt_sym(Rational(20, 3)) == "6 2/3" and fv.fmt_sym(Rational(-20, 3)) == "-6 2/3"
    assert fv.fmt_sym_tex(Rational(20, 3)) == "6\\,\\frac{2}{3}"
    assert fv.fmt_sym(Rational(1, 2)) == "1/2"
    dec2 = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    assert fv.fmt_exact_or(2.82, Rational(141, 50), dec=dec2) == "2.82"
    assert fv.fmt_exact_or(-36.0624, -51 * sqrt(2) / 2, dec=dec2) == "-36.06"
    assert fv.fmt_exact_or(20 / 3, Rational(20, 3), dec=dec2) == "6 2/3"
    assert fv.fmt_exact_or(18795 / 11, Rational(18795, 11)) == "1708.64"        # без «кратных π» по грубому допуску
    # в подписях точек: y = 83x/50 + 141/50 → (0, 2.82) и (−1.7, 0)
    fv.SYMBOLIC_MODE = 'compute'
    res = draw(["83*x/50 + 141/50"], X_LIM_L=-5, X_LIM_R=5, Y_LIM_B=-5, Y_LIM_T=5, X_TAG=True, Y_TAG=True, SHOW_VALUES=True)
    texts = [t.get_text().replace("$", "").replace("−", "-") for t in res['ax'].texts]
    assert any("2.82" in t for t in texts) and any("-1.7" in t for t in texts), texts
    assert not any("141" in t for t in texts), texts


def test_hidden_function_keeps_fill():
    """Скрытая глазком функция: кривой и подписей нет, заливка под ней остаётся, пересечения не считаются."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    try:
        fv.CURVE_HIDDEN = {0}
        res = draw(["x^2", "x"], fig=fig, FILL=[(0, "x", 0.0, 1.0, 0, True, 0.02, True, "0", "1")], INTER=True)
        ax = res['ax']
        long_lines = [l for l in ax.lines if len(l.get_xdata()) > 1000]
        assert len(long_lines) == 1                                 # только f1 = x нарисована
        assert any(t.get_text().startswith(("S", "$S")) for t in ax.texts)   # площадь под скрытой x² есть
        assert fv.LAST_CURVES[0].get('hidden') and fv.LAST_CURVES[0]['kind'] == 'func'
        assert not any("(1, 1)" in t.get_text() for t in ax.texts)  # пересечение со скрытой не подписано
    finally:
        fv.CURVE_HIDDEN = set()


def test_grid_far_from_origin():
    """Сетка и деления строятся на любом расстоянии от начала координат (баг: пропадали дальше ±12)."""
    for (xl, xr, yb, yt) in ((100, 110, 200, 210), (-18, -8, 6.5, 16.5), (1e6, 1e6 + 10, -5, 5)):
        res = draw(["sin(x)"], X_LIM_L=xl, X_LIM_R=xr, Y_LIM_B=yb, Y_LIM_T=yt, X_GRID=1, Y_GRID=1)
        xs = sorted(res['ax'].xaxis.get_majorticklocs())
        ys = sorted(res['ax'].yaxis.get_majorticklocs())
        assert len(xs) >= 8 and xs[0] >= xl and xs[-1] <= xr, (xl, xs)
        assert len(ys) >= 8 and ys[0] >= yb and ys[-1] <= yt, (yb, ys)
        tick_marks = [l for l in res['ax'].lines if l.get_marker() in ('|', '_')]
        assert len(tick_marks) >= 16          # ~9 делений по каждой оси
