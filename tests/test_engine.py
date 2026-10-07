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
             'X_HIDE', 'Y_HIDE', 'AXES_HIDDEN', 'FONT_SIZE', 'SYMBOLIC_MODE', 'MATHTEXT_LABELS')


def _reset_engine():
    fv.FUNCS = []
    fv.FUNC_DOMAINS = []
    fv.CURVE_WIDTHS = []
    fv.CURVE_STYLES = []
    fv.FILL = []
    fv.X_LIM_L, fv.X_LIM_R, fv.Y_LIM_B, fv.Y_LIM_T = -5, 5, -5, 5
    fv.GRID = fv.X_GRID = fv.Y_GRID = 1
    fv.ASIMP = fv.DISC = fv.EXTR = fv.X_TAG = fv.Y_TAG = fv.INTER = fv.SHOW_VALUES = 1
    fv.X_HIDE = fv.Y_HIDE = fv.AXES_HIDDEN = 0
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
    """Отрезки штриховки (стили 0/1): одна LineCollection на область, linewidth 0.6.
    Возвращает список массивов (N, 2)."""
    from matplotlib.collections import LineCollection
    segs = []
    for c in ax.collections:
        if isinstance(c, LineCollection) and abs(float(c.get_linewidths()[0]) - 0.6) < 1e-9:
            segs.extend(np.asarray(s, float) for s in c.get_segments())
    return segs


def hatch_dots(ax):
    """Точечная штриховка (стиль 2): маркер '.', linewidth 0."""
    return [ln for ln in ax.lines if ln.get_marker() == '.' and ln.get_linewidth() == 0]


def region_points(ax):
    """Все точки штриховки (линии и точки) как массив (N, 2)."""
    pts = [s for s in hatch_lines(ax)]
    pts += [np.column_stack((ln.get_xdata(), ln.get_ydata())) for ln in hatch_dots(ax)]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


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

    # область между x² (индекс 1) и x (индекс 3) - пустые строки индексов не сдвигают
    res = draw(funcs, FILL=[fv.region_fill(0.5, 0.4)])
    assert res['errors'] == {}
    pts = region_points(res['ax'])
    assert len(pts) >= 50
    xs, ys = pts[:, 0], pts[:, 1]
    assert xs.min() >= -0.05 and xs.max() <= 1.05
    assert np.all(ys >= xs ** 2 - 0.05) and np.all(ys <= xs + 0.05)
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(1 / 6, abs=2e-4) and str(exact) == "1/6" and not cut

    # точка без области (пустая запись) - тихо пропускается
    res = draw(funcs, FILL=[{'x': None, 'y': None}, fv.region_fill(100.0, 0.0)])
    assert res['errors'] == {}
    assert hatch_lines(res['ax']) == []
    assert fv.LAST_FILL_AREAS == {0: None, 1: None}


@pytest.mark.parametrize("style", [0, 1, 2])
def test_region_styles_produce_hatching(style):
    """Область над x² под y = 2 справа от оси Y: штриховка 45°/135° (одна
    LineCollection) или точки (одна линия-маркер), все точки внутри области."""
    res = draw(["x^2", "y=2"], FILL=[fv.region_fill(0.5, 1.0, style)])
    ax = res['ax']
    assert res['errors'] == {}
    if style == 2:
        assert hatch_lines(ax) == [] and len(hatch_dots(ax)) == 1
    else:
        assert hatch_dots(ax) == [] and len(hatch_lines(ax)) >= 5
    pts = region_points(ax)
    assert len(pts) >= 50
    xs, ys = pts[:, 0], pts[:, 1]
    tol = 0.05
    assert xs.min() >= -tol and xs.max() <= math.sqrt(2) + tol
    assert np.all(ys >= xs ** 2 - tol) and np.all(ys <= 2 + tol)
    assert len(curve_lines(ax)) == 2
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(4 * math.sqrt(2) / 3, abs=3e-4) and not cut


def test_region_boundaries_axes_and_window():
    """Границы области: кривые, оси и края окна. Под x² справа от оси Y с x = 1 -
    S = 1/3 точно; без вертикали область упирается в край окна (cut), подписи «S =» нет."""
    fv.SYMBOLIC_MODE = 'compute'
    res = draw(["x^2", "x=1"], FILL=[fv.region_fill(0.5, 0.1)])
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(1 / 3, abs=2e-4) and str(exact) == "1/3" and not cut
    pts = region_points(res['ax'])
    assert pts[:, 0].min() >= -0.05 and pts[:, 0].max() <= 1.05 and pts[:, 1].min() >= -0.05
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels == ["S = 1/3"]
    # слева от оси Y - зеркальная область, та же площадь (ось Y - граница)
    res = draw(["x^2", "x=-1"], FILL=[fv.region_fill(-0.5, 0.1)])
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(1 / 3, abs=2e-4) and str(exact) == "1/3"
    # без вертикали: область до правого края окна
    res = draw(["x^2"], FILL=[fv.region_fill(0.5, 0.1)])
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert cut and exact is None and area > 1.0
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels and labels[0].startswith("S ≈")
    # точка точно на оси - берётся ближайшая свободная ячейка; далеко за окном - области нет
    res = draw(["x^2", "x=1"], FILL=[fv.region_fill(0.0, 0.5), fv.region_fill(50.0, 0.0)])
    assert fv.LAST_FILL_AREAS[0] is not None and fv.LAST_FILL_AREAS[1] is None


def test_region_implicit_and_vertical_boundaries():
    """Неявные кривые как границы: сегмент круга над y = 2 делится осью Y пополам;
    верхняя половина области между x = y² и x = 4 - численно 16/3 с точностью 1e-3."""
    fv.SYMBOLIC_MODE = 'compute'
    res = draw(["x^2+y^2=9", "y=2"], FILL=[fv.region_fill(1.0, 2.5), fv.region_fill(-1.0, 2.5, 1)])
    assert res['errors'] == {}
    half = (9 * math.acos(2 / 3) - 2 * math.sqrt(5)) / 2
    for i in (0, 1):
        area, exact, cut = fv.LAST_FILL_AREAS[i]
        assert area == pytest.approx(half, rel=1e-3) and not cut
    pts = region_points(res['ax'])
    assert np.all(pts[:, 1] >= 2 - 0.05) and np.all(pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 9.2)
    res = draw(["x=y^2", "x=4"], FILL=[fv.region_fill(2.0, 0.5, 2)])
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(16 / 3, rel=1e-3) and not cut
    pts = region_points(res['ax'])
    assert np.all(pts[:, 1] >= -0.05) and np.all(pts[:, 0] <= 4.05) and np.all(pts[:, 1] ** 2 <= pts[:, 0] + 0.1)


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


def test_region_at_for_hover_highlight():
    """region_at(x, y): область под курсором по стенам последнего построения - та же маска
    для точек одной области, другая для соседней, None вне окна; после нового построения
    старые маски не используются."""
    res = draw(["x^2", "x=1"])
    assert fv._REGION_CONTEXT and not fv._REGION_CONTEXT['cache']      # стены строятся лениво
    m1, extent = fv.region_at(0.5, 0.1)
    assert extent == (-5, 5, -5, 5) and m1.dtype == bool and 0 < m1.sum() < m1.size / 50
    m2, _ = fv.region_at(0.7, 0.2)
    assert m2 is m1                                                     # та же область - из кэша
    m3, _ = fv.region_at(2.0, 3.0)
    assert m3 is not m1 and m3.sum() > m1.sum() * 10 and not (m3 & m1).any()
    assert fv.region_at(50.0, 0.0) is None and fv.region_at(float('nan'), 0.0) is None
    # точка точно на оси Y - ближайшая свободная ячейка
    assert fv.region_at(0.0, 2.0) is not None
    # штриховка по маске из region_at совпадает с областью заливки того же построения
    draw(["x^2", "x=1"], FILL=[fv.region_fill(0.5, 0.1)])
    m4, _ = fv.region_at(0.5, 0.1)
    assert np.array_equal(m4, fv._LAST_REGION['mask'])
    fv.FILL = []
    draw([""])
    assert fv.region_at(0.5, 0.1)[0].sum() > m3.sum()                   # без кривых - вся четверть окна


def test_hidden_axes():
    """AXES_HIDDEN: нет стрелок осей, делений и подписей x/y, оси не ограничивают области:
    под y = 1 над x² с осями - половина (точка у оси Y), без осей - вся область, S = 1 1/3."""
    fv.SYMBOLIC_MODE = 'compute'
    res = draw(["x^2", "y=1"], FILL=[fv.region_fill(0.0, 0.5)])
    ax = res['ax']
    arrows = [a for a in ax.texts if getattr(a, 'arrow_patch', None) is not None and a.get_text() == ""]
    assert len(arrows) == 2 and any(t.get_text() == "x" for t in ax.texts)
    assert any(t.get_text() == "3" for t in ax.texts)                 # деления
    assert fv.LAST_FILL_AREAS[0][0] == pytest.approx(2 / 3, abs=2e-3)   # ось Y делит область
    res = draw(["x^2", "y=1"], FILL=[fv.region_fill(0.0, 0.5)], AXES_HIDDEN=1)
    ax = res['ax']
    arrows = [a for a in ax.texts if getattr(a, 'arrow_patch', None) is not None and a.get_text() == ""]
    assert arrows == [] and not any(t.get_text() == "3" for t in ax.texts)
    labels = fv._active_axis_label_manager.artists
    assert not labels['x'].get_visible() and not labels['y'].get_visible()
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(4 / 3, abs=2e-3) and str(exact) == "4/3" and not cut
    assert fv.region_at(0.0, 0.5)[0].sum() == pytest.approx(fv._LAST_REGION['mask'].sum())
    assert len(curve_lines(ax)) == 2                                    # кривые и сетка остались
    assert ax.xaxis.get_gridlines()[0].get_visible()


def test_region_area_label():
    """Число площади области: точная форма для «школьных» областей (синус над осью:
    S = 2, 1/x между x = 1 и x = 2: ln 2, √x до x = 4: 5 1/3), без галочки подписи нет,
    но площадь посчитана (для строки области)."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    fv.SYMBOLIC_MODE = 'compute'
    res = draw(["sin(x)"], fig=fig, FILL=[fv.region_fill(1.5, 0.3)])
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels == ["S = 2"], labels
    res = draw(["1/x", "x=1", "x=2"], fig=fig, FILL=[fv.region_fill(1.5, 0.3)])
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(math.log(2), abs=2e-4) and str(exact) == "log(2)"
    res = draw(["sqrt(x)", "x=4"], fig=fig, FILL=[fv.region_fill(2.0, 0.5)])
    labels = [t.get_text() for t in res['ax'].texts if t.get_text().startswith("S")]
    assert labels == ["S = 5 1/3"], labels
    # галочка снята - подписи нет, но площадь посчитана
    res = draw(["x^2", "x=1"], fig=fig, FILL=[fv.region_fill(0.5, 0.1, 0, 0.02, False)])
    assert not [t for t in res['ax'].texts if t.get_text().startswith("S")]
    area, exact, cut = fv.LAST_FILL_AREAS[0]
    assert area == pytest.approx(1 / 3, abs=2e-4) and str(exact) == "1/3"
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


def test_hidden_curves_stay_region_boundaries():
    """Скрытая глазком кривая (функция, вертикаль, неявная): не рисуется, в пересечениях
    не участвует, но остаётся границей области - штриховка и площадь не меняются."""
    fig = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(fig)
    fv.SYMBOLIC_MODE = 'compute'
    try:
        fv.CURVE_HIDDEN = {0}
        res = draw(["x^2", "x"], fig=fig, FILL=[fv.region_fill(0.5, 0.4)], INTER=True)
        ax = res['ax']
        assert len(curve_lines(ax)) == 1                            # только f1 = x нарисована
        area, exact, cut = fv.LAST_FILL_AREAS[0]
        assert str(exact) == "1/6"                                  # x² скрыта, но ограничивает область
        assert fv.LAST_CURVES[0].get('hidden') and fv.LAST_CURVES[0]['kind'] == 'func'
        assert not any("(1, 1)" in t.get_text() for t in ax.texts)  # пересечение со скрытой не подписано
        fv.CURVE_HIDDEN = {2}
        res = draw(["x^2", "x", "x=1"], fig=fig, FILL=[fv.region_fill(0.5, 0.1)])
        assert fv.LAST_CURVES[2].get('hidden') and fv.LAST_CURVES[2]['kind'] == 'vline'
        assert not [l for l in res['ax'].lines if list(l.get_xdata()) == [1.0, 1.0]]
        assert str(fv.LAST_FILL_AREAS[0][1]) == "1/3"
        fv.CURVE_HIDDEN = {0}
        res = draw(["x^2+y^2=9", "y=2"], fig=fig, FILL=[fv.region_fill(1.0, 2.5)])
        assert fv.LAST_CURVES[0].get('hidden') and fv.LAST_CURVES[0]['kind'] == 'implicit'
        assert not res['ax'].collections or all(getattr(c, 'get_linewidths', lambda: [0.6])()[0] == 0.6
                                                for c in res['ax'].collections)
        half = (9 * math.acos(2 / 3) - 2 * math.sqrt(5)) / 2
        assert fv.LAST_FILL_AREAS[0][0] == pytest.approx(half, rel=1e-3)
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


# ──────────────────────────────────────────────
#  Производительность: три точечных ускорения движка
# ──────────────────────────────────────────────

def _nan_edges_loop(mask):
    """Прежняя (цикловая) реализация набора nan_edges из find_extrema_numerical."""
    edges = set()
    for i in range(1, len(mask)):
        if mask[i] != mask[i - 1]:
            edges.add(i - 1)
            edges.add(i)
    return edges


def test_nan_edge_indices_matches_loop():
    rng = np.random.default_rng(12345)
    for n in (0, 1, 2, 3, 7, 6000):
        for p in (0.0, 0.05, 0.5, 0.95, 1.0):
            mask = rng.random(n) < p
            assert fv._nan_edge_indices(mask) == _nan_edges_loop(mask)
    # ручные случаи: NaN-зона в начале, в конце и внутри
    assert fv._nan_edge_indices([False, True, True]) == {0, 1}
    assert fv._nan_edge_indices([True, True, False]) == {1, 2}
    assert fv._nan_edge_indices([True, False, True]) == {0, 1, 2}
    assert fv._nan_edge_indices([True, True, True]) == set()


@pytest.mark.parametrize("func_str, disc, expected", [
    ("sqrt(4-x^2)", [], [(0.0, 2.0)]),     # NaN при |x|>2, максимум (0, 2)
    ("1/x", [0.0], []),                    # разрыв в 0, экстремумов нет
    ("log(x)", [], []),                    # NaN при x<=0, монотонна
])
def test_extrema_with_nan_gaps(func_str, disc, expected):
    _expr, _raw, f = fv.build_numpy_func(func_str)
    xs = np.linspace(-5, 5, 6001)          # нечётное число узлов: 0.0 на сетке
    ys = fv.make_y_array(f, xs, disc, -5, 5)
    finite = np.isfinite(ys)
    assert finite.any() and not finite.all()
    edges = sorted(fv._nan_edge_indices(finite))
    assert edges, "у функции должны быть границы NaN-областей"
    edge_xs = xs[edges]
    extrema = fv.find_extrema_numerical(f, xs, ys, -5, 5, disc)
    assert len(extrema) == len(expected)
    for (ex, ey), (px, py) in zip(sorted(extrema), sorted(expected)):
        assert abs(ex - px) < 1e-6 and abs(ey - py) < 1e-6
    # ни один экстремум не стоит вплотную к границе NaN-области
    # (артефакт nan_to_num: крайний конечный узел выглядит локальным минимумом)
    for ex, _ey in extrema:
        assert np.min(np.abs(edge_xs - ex)) > 0.05


@pytest.mark.parametrize("text, expected", [
    # быстрый путь (обычные десятичные числа)
    ("1", 1.0), ("1.5", 1.5), ("-2", -2.0), ("+3", 3.0), (".5", 0.5), ("5.", 5.0),
    ("1e3", 1000.0), ("1E-2", 0.01), ("-1.5e+2", -150.0), ("-7E+1", -70.0),
    (" 2.5 ", 2.5), ("\t-3\n", -3.0), ("00012", 12.0), ("007.5", 7.5),
    ("1e400", float("inf")), ("-1e400", float("-inf")), ("1e-400", 0.0),
    ("12345678901234567890", 1.2345678901234567e+19),
    # прежний путь (sympy / float): семантика не меняется
    ("pi", math.pi), ("π", math.pi), ("2pi", 2 * math.pi), ("pi/2", math.pi / 2),
    ("2*pi", 2 * math.pi), ("e", math.e), ("2e", 2 * math.e), ("1e", math.e),
    ("e1", math.e), ("1/2", 0.5), ("3+2", 5.0), ("sqrt(2)", math.sqrt(2)),
    ("- 2", -2.0), ("1 000", 0.0), ("1..2", 0.2), ("1e3.5", 500.0),
    ("0x10", 16.0), ("1_000", 1000.0), ("٣", 3.0), ("1.5.5", 0.75),
    ("inf", float("inf")), ("+inf", float("inf")), ("-inf", float("-inf")),
    ("+∞", float("inf")), ("∞", float("inf")), ("-∞", float("-inf")),
])
def test_parse_number_table(text, expected):
    got = fv.parse_number(text)
    assert type(got) is float
    assert got == expected


def test_parse_number_nan_and_signed_zero():
    assert math.isnan(fv.parse_number("nan"))
    # «-0» через sympy давал +0.0, быстрый путь обязан вернуть то же
    for t in ("-0", "-0.0", "-.0", "-0e0", "0", "+0"):
        v = fv.parse_number(t)
        assert v == 0.0 and math.copysign(1.0, v) == 1.0, t


@pytest.mark.parametrize("text", ["−5", "−1.5", "−1e3", "−inf", "−", "abc", "1,5",
                                  "1-", "1e+", "", "   "])
def test_parse_number_garbage_raises(text):
    with pytest.raises(ValueError):
        fv.parse_number(text)


def test_parse_number_default():
    assert fv.parse_number("", default=7.5) == 7.5
    assert fv.parse_number(None, default=-1.0) == -1.0
    assert fv.parse_number("   ", default=float("inf")) == float("inf")
    with pytest.raises(ValueError):
        fv.parse_number(None)


def _render_multiscript_text():
    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.text(0.1, 0.7, "Привет мир, функция", fontsize=14)
    ax.text(0.1, 0.4, "שלום עולם", fontsize=14)
    ax.text(0.1, 0.1, "y = x² + $\\sqrt{2}$", fontsize=14)
    ax.set_title("Заголовок")
    canvas.draw()
    return np.array(canvas.buffer_rgba()).copy()


def test_font_lookup_memo_installed_once_and_pixel_identical():
    from matplotlib import font_manager
    fm = font_manager.fontManager
    if not hasattr(type(fm), '_find_fonts_by_props'):
        pytest.skip("в этой версии matplotlib нет FontManager._find_fonts_by_props")
    assert fv._FONT_LOOKUP_MEMO_INSTALLED
    assert '_find_fonts_by_props' in vars(fm)
    memo = vars(fm)['_find_fonts_by_props']
    # повторный вызов ничего не переустанавливает
    fv.apply_font_preset(fv.GRAPH_FONT)
    assert vars(fm)['_find_fonts_by_props'] is memo

    fv._FONT_LOOKUP_CACHE.clear()
    with_memo = _render_multiscript_text()
    assert fv._FONT_LOOKUP_CACHE, "кэш должен наполниться при отрисовке"
    with_memo_again = _render_multiscript_text()
    del fm._find_fonts_by_props          # временно снимаем мемоизацию
    try:
        without_memo = _render_multiscript_text()
    finally:
        fm._find_fonts_by_props = memo
    assert np.array_equal(with_memo, with_memo_again)
    assert np.array_equal(with_memo, without_memo)

    # fallback-семейства (иврит) действительно участвуют в отрисовке
    fig = Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.text(0.1, 0.7, "Привет мир, функция", fontsize=14)
    ax.text(0.1, 0.1, "y = x² + $\\sqrt{2}$", fontsize=14)
    ax.set_title("Заголовок")
    canvas.draw()
    assert not np.array_equal(with_memo, np.array(canvas.buffer_rgba()))


def test_font_lookup_memo_respects_preset_change():
    from matplotlib import font_manager
    if '_find_fonts_by_props' not in vars(font_manager.fontManager):
        pytest.skip("мемоизация не установлена")
    base = fv.GRAPH_FONT
    try:
        fv.apply_font_preset(base)
        a = _render_multiscript_text()
        fv.apply_font_preset('sans')
        b = _render_multiscript_text()
    finally:
        fv.apply_font_preset(base)
    c = _render_multiscript_text()
    assert not np.array_equal(a, b)      # другой пресет: ключ кэша другой
    assert np.array_equal(a, c)          # возврат к пресету: тот же результат


def test_region_raster_helpers_match_dense_references():
    """Растровые помощники заливки считают в окне маски, а не на полном растре, с тем же
    результатом: _runs_along_rows против поточечного перебора, _region_center против
    плотной версии на всём растре (сдвиг маски сдвигает центр ровно на столько же),
    _mask_window - прямоугольник маски с запасом, обрезанный краями растра."""
    rng = np.random.default_rng(7)
    ny, nx = 61, 83
    for p in (0.1, 0.5, 0.9):
        a = rng.random((ny, nx)) < p
        r, s, e = fv._runs_along_rows(a)
        ref = []
        for i in range(ny):
            j = 0
            while j < nx:
                if a[i, j]:
                    k = j
                    while k < nx and a[i, k]:
                        k += 1
                    ref.append((i, j, k))
                    j = k
                else:
                    j += 1
        assert list(zip(r.tolist(), s.tolist(), e.tolist())) == ref
    assert all(v.size == 0 for v in fv._runs_along_rows(np.zeros((3, 4), dtype=bool)))

    def dense_center(mask):
        """Прежняя реализация: четыре прохода accumulate по всему растру."""
        ny, nx = mask.shape
        blocked = ~mask
        jj = np.broadcast_to(np.arange(nx)[None, :], mask.shape)
        ii = np.broadcast_to(np.arange(ny)[:, None], mask.shape)
        left = jj - np.maximum.accumulate(np.where(blocked, jj, -1), axis=1)
        right = np.minimum.accumulate(np.where(blocked, jj, nx)[:, ::-1], axis=1)[:, ::-1] - jj
        down = ii - np.maximum.accumulate(np.where(blocked, ii, -1), axis=0)
        up = np.minimum.accumulate(np.where(blocked, ii, ny)[::-1, :], axis=0)[::-1, :] - ii
        score = np.minimum(np.minimum(left, right), np.minimum(up, down)).astype(float)
        score[blocked] = 0.0
        best = float(score.max())
        if best <= 0:
            return None
        cand = np.argwhere(score >= best * 0.85)
        ci, cj = np.argwhere(mask).mean(axis=0)
        d = (cand[:, 0] - ci) ** 2 + (cand[:, 1] - cj) ** 2
        i, j = cand[int(np.argmin(d))]
        return int(i), int(j)

    yy, xx = np.mgrid[0:ny, 0:nx]
    shapes = [((yy - 20) ** 2 + (xx - 30) ** 2 <= 90) | ((yy >= 18) & (yy <= 22) & (xx >= 30) & (xx <= 60)),
              (yy <= 2) | (xx >= nx - 2),                                     # у края растра
              rng.random((ny, nx)) < 0.7,
              np.zeros((ny, nx), dtype=bool)]
    shapes[3][40, 50] = True                                                  # одна ячейка
    for mask in shapes:
        assert fv._region_center(mask) == dense_center(mask)
    assert fv._region_center(np.zeros((ny, nx), dtype=bool)) is None
    blob = shapes[0]
    c0 = fv._region_center(blob)
    shifted = np.zeros((ny + 15, nx + 9), dtype=bool)
    shifted[15:, 9:] = blob
    assert fv._region_center(shifted) == (c0[0] + 15, c0[1] + 9)

    m = np.zeros((ny, nx), dtype=bool)
    m[10:13, 20:25] = True
    assert fv._mask_window(m, 1) == (slice(9, 14), slice(19, 26))
    assert fv._mask_window(m, 0) == (slice(10, 13), slice(20, 25))
    m[:] = False
    m[0:2, nx - 3:nx] = True
    assert fv._mask_window(m, 1) == (slice(0, 3), slice(nx - 4, nx))
    assert fv._mask_window(np.zeros((ny, nx), dtype=bool)) is None
