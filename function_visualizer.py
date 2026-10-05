"""
Умный визуализатор функций
Зависимости: matplotlib, numpy, sympy, scipy
Установка: pip install matplotlib numpy sympy scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from sympy import (
    sympify, symbols, lambdify, diff, limit, oo, nan, zoo,
    im, re, singularities, solve, pi, I, Rational, Float
)
from scipy.signal import argrelextrema
from scipy.optimize import brentq

# ──────────────────────────────────────────────
#  ПАРАМЕТРЫ — меняй только эту секцию
# ──────────────────────────────────────────────

FUNCS  = [
    "sin(x) / x",
    # "x**2",       # добавь ещё функции сюда
]

# Область определения для каждой функции: список (from, to) той же длины,
# что FUNCS. По умолчанию (или если короче) — (-inf, inf), т.е. вся область.
# Функция строится только на пересечении [from, to] с view window.
FUNC_DOMAINS = []

GRID   = 1    # 1 — показывать сетку, 0 — нет
X_GRID = 1    # шаг сетки по оси X
Y_GRID = 1    # шаг сетки по оси Y

X_LIM_L = -5  # левый предел по X
X_LIM_R =  5  # правый предел по X
Y_LIM_B =  -5  # нижний предел по Y
Y_LIM_T =   5  # верхний предел по Y

ASIMP  = 1    # 1 — показывать асимптоты (гориз. и верт.),  0 — нет
DISC   = 1    # 1 — показывать устранимые разрывы (holes),   0 — нет
EXTR   = 1    # 1 — показывать экстремумы,             0 — нет
X_TAG  = 1    # 1 — показывать пересечения с осью X,   0 — нет
Y_TAG  = 1    # 1 — показывать пересечения с осью Y,   0 — нет
INTER  = 1    # 1 — показывать пересечения между функциями, 0 — нет
SHOW_VALUES = 1  # 1 — показывать координаты точек,    0 — только точки без подписей

X_HIDE = 0    # 1 — скрыть числа делений на оси X,     0 — показывать
Y_HIDE = 0    # 1 — скрыть числа делений на оси Y,     0 — показывать

FONT_SIZE = 10  # размер текста на графике (точки/подписи/деления)

# Заливка площади
# Формат: (f1, f2, x_от, x_до, стиль)
#
# f1  — индекс первой функции (из FUNCS)
# f2  — индекс второй функции (из FUNCS), или "x" чтобы считать до оси X
# x_от, x_до — диапазон по X
# стиль — 0: штриховка 45°,  1: штриховка 135°,  2: точки
#
# Примеры:
#   (0, "x", -3, 3, 0)   — площадь между функцией 0 и осью X, штриховка 45°
#   (0, 1, -3, 3, 1)     — площадь между функцией 0 и функцией 1, штриховка 135°
FILL = [
    # (0, "x", -3, 3, 0),
]

# ──────────────────────────────────────────────
#  КОД — не трогай ниже этой строки
# ──────────────────────────────────────────────

def _preprocess_func_str(s):
    """
    Преобразует удобные пользователю записи логарифмов в форму sympy:
      log_2(x)   -> log(x, 2)
      log_10(x)  -> log(x, 10)
      log2(x)    -> log(x, 2)
      log10(x)   -> log(x, 10)
      lg(x)      -> log(x, 10)   (десятичный логарифм)
    Основание может быть числом или выражением (log_e, log_pi и т.п.).
    ln(x) и обычный log(x) (натуральный) не трогаются.
    """
    import re
    if not s:
        return s

    # log_<base>(...)  где base — число/идентификатор (e, pi, 2, 10, ...)
    # Заменяем на log(<...>, <base>). Берём сбалансированные скобки простым
    # нежадным подходом для одного уровня вложенности аргумента.
    def repl_underscore(m):
        base = m.group(1)
        return f'log__BASE__{base}__('

    # сначала пометим log_<base>(  -> спец-маркер с захватом аргумента ниже
    s = re.sub(r'log_([A-Za-z0-9.]+)\s*\(', repl_underscore, s)

    # log2( / log10(  -> тоже маркер
    s = re.sub(r'log2\s*\(',  'log__BASE__2__(',  s)
    s = re.sub(r'log10\s*\(', 'log__BASE__10__(', s)
    # lg( -> десятичный
    s = re.sub(r'(?<![A-Za-z])lg\s*\(', 'log__BASE__10__(', s)

    # Теперь раскрываем маркеры log__BASE__<b>__( ... ) -> log( ... , <b> )
    # с корректным учётом вложенных скобок.
    out = []
    i = 0
    marker = 'log__BASE__'
    while i < len(s):
        j = s.find(marker, i)
        if j == -1:
            out.append(s[i:])
            break
        out.append(s[i:j])
        # читаем основание до '__('
        k = j + len(marker)
        end_base = s.find('__(', k)
        base = s[k:end_base]
        # находим парную закрывающую скобку для '(' в позиции end_base+2
        open_paren = end_base + 2
        depth = 1
        p = open_paren + 1
        while p < len(s) and depth > 0:
            if s[p] == '(':
                depth += 1
            elif s[p] == ')':
                depth -= 1
            p += 1
        inner = s[open_paren + 1:p - 1]
        out.append(f'log(({inner}), {base})')
        i = p
    return ''.join(out)


def _parse_equation_input(s):
    """
    Классифицирует строку из поля функции:
      • 'y = ln(x)'          -> ('func', 'ln(x)')      обычная функция y=f(x)
      • 'ln(x)'              -> ('func', 'ln(x)')       без префикса — тоже функция
      • 'x = e'  / 'x = 2pi' -> ('vline', <число>)      вертикальная линия x=c
      • 'y = 3'              -> ('func', '3')            горизонталь как обычная функция
      • 'x**2+y**2=25'       -> ('implicit', ('x**2+y**2', '25'))   неявная кривая
      • 'y**2 = x'           -> ('implicit', ('y**2', 'x'))         (в правой части есть x → неявная)
    Правило: 'y=<выражение без y>' — обычная функция; 'x=<константа без x,y>' —
    вертикаль; всё остальное с '=' — неявная кривая F(x,y)=G(x,y).
    """
    if s is None:
        return ('func', '')
    raw = str(s).strip()

    import re
    def _has_var(expr, var):
        # есть ли переменная var (x или y) как отдельный идентификатор
        return re.search(r'(?<![A-Za-z_])' + var + r'(?![A-Za-z_])', expr) is not None

    if '=' not in raw:
        # Нет '=' — подразумеваем 'y = <выражение>'. Если в выражении есть y
        # (например 'cos(y)'), это уже неявное уравнение y = f(...,y);
        # иначе — обычная функция y=f(x).
        if _has_var(raw.lower(), 'y'):
            return ('implicit', ('y', raw))
        return ('func', raw)

    left, _, right = raw.partition('=')
    left_n = left.strip().lower()
    right = right.strip()

    # y = f(x): слева ровно 'y', справа НЕТ y -> обычная функция
    if left_n == 'y' and not _has_var(right.lower(), 'y'):
        return ('func', right)

    # x = c: слева ровно 'x', справа константа (нет ни x, ни y) -> вертикаль
    if left_n == 'x' and not _has_var(right.lower(), 'x') and not _has_var(right.lower(), 'y'):
        try:
            c = parse_number(right)
            return ('vline', c)
        except Exception:
            return ('vline', None)

    # Всё остальное с '=' — неявная кривая F(x,y) = G(x,y)
    return ('implicit', (left.strip(), right))


def build_numpy_func(func_str):
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)
    from sympy import E as _E, pi as _pi
    func_str = _preprocess_func_str(func_str)
    x = symbols('x')

    # Те же преобразования, что и в parse_number: неявное умножение (2x, ex,
    # 2pi) и ^ как степень. local_dict делает 'e' числом Эйлера, 'pi' — π,
    # 'x' — переменной. Без этого 'e' считался свободным символом и
    # 'e*x-e' не вычислялся (пустой график + долгое раздумье).
    transformations = (standard_transformations +
                       (implicit_multiplication_application, convert_xor))
    local_dict = {'e': _E, 'pi': _pi, 'x': x}

    expr_raw = parse_expr(func_str, evaluate=False,
                          transformations=transformations, local_dict=local_dict)
    expr     = parse_expr(func_str, transformations=transformations,
                          local_dict=local_dict)
    f = lambdify(x, expr, modules=[
        {'sqrt': np.sqrt, 'log': np.log, 'ln': np.log,
         'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
         'tan': np.tan, 'abs': np.abs, 'pi': np.pi,
         'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan},
        'numpy'
    ])
    return expr, expr_raw, f


def build_implicit_func(lhs_str, rhs_str):
    """
    Строит numpy-функцию H(x, y) = F(x, y) - G(x, y) для неявной кривой
    F(x,y) = G(x,y). Кривая — это множество точек, где H = 0, поэтому её
    можно нарисовать как линию уровня 0 (contour). Поддерживает pi/e,
    неявное умножение, ^ как степень — так же, как build_numpy_func.
    Возвращает callable H(X, Y), работающий с массивами (meshgrid).
    """
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)
    from sympy import E as _E, pi as _pi
    x, y = symbols('x y')

    transformations = (standard_transformations +
                       (implicit_multiplication_application, convert_xor))
    local_dict = {'e': _E, 'pi': _pi, 'x': x, 'y': y}

    lhs = parse_expr(_preprocess_func_str(lhs_str),
                     transformations=transformations, local_dict=local_dict)
    rhs = parse_expr(_preprocess_func_str(rhs_str),
                     transformations=transformations, local_dict=local_dict)
    H = lhs - rhs

    modules = [
        {'sqrt': np.sqrt, 'log': np.log, 'ln': np.log,
         'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
         'tan': np.tan, 'abs': np.abs, 'pi': np.pi,
         'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan},
        'numpy'
    ]
    h = lambdify((x, y), H, modules=modules)
    return H, h


def _scan_roots_1d(g, lo, hi, n=2000, xtol=1e-9):
    """
    Находит корни функции одной переменной g на [lo, hi]: сканирует мелкую
    сетку, ищет смену знака между соседними узлами и уточняет корень через
    brentq. Возвращает отсортированный список уникальных корней.
    Используется для пересечений неявной кривой с осями: g(t)=H(t,0) для оси
    X и g(t)=H(0,t) для оси Y.
    """
    ts = np.linspace(lo, hi, n)
    with np.errstate(all='ignore'):
        vals = np.array([_safe_scalar(g, t) for t in ts], dtype=float)
    roots = []

    def _add(r):
        if not any(abs(r - e) < 1e-6 for e in roots):
            roots.append(r)

    for i in range(len(ts) - 1):
        a, b = ts[i], ts[i + 1]
        fa, fb = vals[i], vals[i + 1]
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa == 0.0:
            _add(a)
        if fa * fb < 0:
            try:
                r = brentq(g, a, b, xtol=xtol)
                if np.isfinite(r):
                    _add(r)
            except Exception:
                pass
    # правый конец
    if len(vals) and vals[-1] == 0.0:
        _add(ts[-1])
    return sorted(roots)


def _safe_scalar(g, t):
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            v = g(t)
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _dedup_points(pts, tol=1e-3):
    """Убирает почти совпадающие точки (x,y)."""
    out = []
    for p in pts:
        if not any(abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol for q in out):
            out.append(p)
    return out


def intersect_func_implicit(f, h, x_lo, x_hi, y_lo, y_hi):
    """
    Пересечения явной функции y=f(x) и неявной кривой H(x,y)=0:
    ищем корни одномерной g(x) = H(x, f(x)) на [x_lo, x_hi], затем оставляем
    те точки, чья y=f(x) попадает в окно.
    """
    def g(t):
        yv = _safe_scalar(f, t)
        if not np.isfinite(yv):
            return np.nan
        return _safe_scalar(lambda _: h(t, yv), 0.0)
    pts = []
    for xr in _scan_roots_1d(g, x_lo, x_hi):
        yv = _safe_scalar(f, xr)
        if np.isfinite(yv) and y_lo - 1e-9 <= yv <= y_hi + 1e-9:
            pts.append((xr, yv))
    return _dedup_points(pts)


def intersect_implicit_implicit(h1, h2, x_lo, x_hi, y_lo, y_hi, n=240):
    """
    Пересечения двух неявных кривых H1(x,y)=0 и H2(x,y)=0.
    Идея: на сетке считаем знаки H1 и H2; в каждой ячейке, где ОБЕ функции
    меняют знак (то есть через ячейку проходят обе кривые), берём центр
    ячейки как стартовое приближение и уточняем 2D-методом Ньютона по
    системе {H1=0, H2=0}. Затем дедуп. Ловит любое число точек (в т.ч.
    2 пересечения двух окружностей).
    """
    xs = np.linspace(x_lo, x_hi, n)
    ys = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all='ignore'):
        Z1 = np.asarray(h1(X, Y), dtype=float)
        Z2 = np.asarray(h2(X, Y), dtype=float)

    def sign_change_cell(Z, i, j):
        a, b, c, d = Z[i, j], Z[i, j+1], Z[i+1, j], Z[i+1, j+1]
        vals = [v for v in (a, b, c, d) if np.isfinite(v)]
        if len(vals) < 2:
            return False
        return (min(vals) <= 0 <= max(vals))

    def newton(x0, y0):
        # 2D Newton на систему H1=H2=0 с численным якобианом
        x, y = x0, y0
        eps = 1e-6
        for _ in range(40):
            f1 = _safe_scalar(lambda _: h1(x, y), 0.0)
            f2 = _safe_scalar(lambda _: h2(x, y), 0.0)
            if not (np.isfinite(f1) and np.isfinite(f2)):
                return None
            if abs(f1) < 1e-11 and abs(f2) < 1e-11:
                return (x, y)
            j11 = (_safe_scalar(lambda _: h1(x+eps, y), 0.0) - f1) / eps
            j12 = (_safe_scalar(lambda _: h1(x, y+eps), 0.0) - f1) / eps
            j21 = (_safe_scalar(lambda _: h2(x+eps, y), 0.0) - f2) / eps
            j22 = (_safe_scalar(lambda _: h2(x, y+eps), 0.0) - f2) / eps
            det = j11*j22 - j12*j21
            if abs(det) < 1e-14:
                return None
            dx = ( j22*f1 - j12*f2) / det
            dy = (-j21*f1 + j11*f2) / det
            x -= dx; y -= dy
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                break
        f1 = _safe_scalar(lambda _: h1(x, y), 0.0)
        f2 = _safe_scalar(lambda _: h2(x, y), 0.0)
        if np.isfinite(f1) and np.isfinite(f2) and abs(f1) < 1e-6 and abs(f2) < 1e-6:
            return (x, y)
        return None

    seeds = []
    for i in range(n-1):
        for j in range(n-1):
            if sign_change_cell(Z1, i, j) and sign_change_cell(Z2, i, j):
                seeds.append(((xs[j]+xs[j+1])/2.0, (ys[i]+ys[i+1])/2.0))

    pts = []
    for (sx, sy) in seeds:
        r = newton(sx, sy)
        if r is None:
            continue
        rx, ry = r
        if x_lo - 1e-9 <= rx <= x_hi + 1e-9 and y_lo - 1e-9 <= ry <= y_hi + 1e-9:
            pts.append((rx, ry))
    return _dedup_points(pts, tol=1e-2)


def _get_x(expr_sympy):
    free = expr_sympy.free_symbols
    return next((s for s in free if str(s) == 'x'), symbols('x'))


def _safe_val(f, x):
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            v = float(f(x))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def find_discontinuities_numerical(f, expr_sympy, expr_raw, x_lim_l, x_lim_r, n=10000):
    disc = []
    x_sym = _get_x(expr_raw) if expr_raw.free_symbols else _get_x(expr_sympy)

    for expr_candidate in [expr_raw, expr_sympy]:
        try:
            sp_pts = list(singularities(expr_candidate, x_sym))
            for pt in sp_pts:
                if im(pt) == 0:
                    px = float(re(pt))
                    if x_lim_l <= px <= x_lim_r:
                        if not any(abs(px - d) < 0.01 for d in disc):
                            disc.append(round(px, 8))
        except Exception:
            pass
        if disc:
            break

    if not disc:
        xs = np.linspace(x_lim_l, x_lim_r, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            ys = np.array([_safe_val(f, xi) for xi in xs], dtype=float)

        bad = ~np.isfinite(ys)
        transitions = np.where(np.diff(bad.astype(int)) != 0)[0]
        for idx in transitions:
            xc = (xs[idx] + xs[idx + 1]) / 2
            probes = [_safe_val(f, xc + d) for d in [-1e-5, 0, 1e-5]]
            if not all(np.isfinite(v) for v in probes):
                if not any(abs(xc - d) < 0.05 for d in disc):
                    disc.append(round(xc, 6))

        dy = np.abs(np.diff(ys))
        threshold = max(50.0, 20 * np.nanstd(dy))
        jumps = np.where(dy > threshold)[0]
        for idx in jumps:
            xc = (xs[idx] + xs[idx + 1]) / 2
            vl = abs(_safe_val(f, xc - 1e-4))
            vr = abs(_safe_val(f, xc + 1e-4))
            if (vl > 20 or vr > 20) and not any(abs(xc - d) < 0.05 for d in disc):
                disc.append(round(xc, 6))

    disc.sort()
    return disc


def classify_discontinuities(f, expr_sympy, disc_pts, y_lim_b, y_lim_t):
    x_sym = _get_x(expr_sympy)
    vasymps = []
    removable = []

    for dp in disc_pts:
        lim_val = None
        try:
            lv_p = limit(expr_sympy, x_sym, dp, '+')
            lv_m = limit(expr_sympy, x_sym, dp, '-')
            if lv_p in (oo, -oo, zoo) or lv_m in (oo, -oo, zoo):
                vasymps.append(dp)
                continue
            if lv_p == lv_m and lv_p.is_real:
                lim_val = float(lv_p)
        except Exception:
            pass

        if lim_val is None:
            eps_vals = [_safe_val(f, dp + d) for d in [1e-6, -1e-6, 1e-5, -1e-5]]
            eps_vals = [v for v in eps_vals if np.isfinite(v)]
            if not eps_vals:
                vasymps.append(dp)
                continue
            if max(abs(v) for v in eps_vals) > 200:
                vasymps.append(dp)
                continue
            lim_val = round(sum(eps_vals) / len(eps_vals), 6)

        if lim_val is not None and y_lim_b <= lim_val <= y_lim_t:
            removable.append((dp, lim_val))
        else:
            vasymps.append(dp)

    return vasymps, removable


def find_horizontal_asymptotes(f, expr_sympy, x_lim_l, x_lim_r, y_lim_b, y_lim_t):
    hasymps = []
    x_sym = _get_x(expr_sympy)

    for direction in [oo, -oo]:
        val = None
        try:
            lv = limit(expr_sympy, x_sym, direction)
            if lv not in (oo, -oo, zoo, nan) and lv.is_real:
                val = float(lv)
        except Exception:
            pass

        if val is None:
            x_span = x_lim_r - x_lim_l
            sign = 1 if direction == oo else -1
            far_points = [sign * x_span * k for k in [100, 1000, 10000, 100000]]
            probes = [_safe_val(f, xp) for xp in far_points]
            probes = [v for v in probes if np.isfinite(v)]
            if len(probes) >= 3:
                diffs = [abs(probes[i+1] - probes[i]) for i in range(len(probes)-1)]
                if all(d < 1e-4 for d in diffs):
                    val = probes[-1]

        if val is not None:
            val = round(val, 8)
            if y_lim_b <= val <= y_lim_t and val not in hasymps:
                hasymps.append(val)

    return hasymps


def detect_const_context(func_str):
    """
    Определяет, какие «красивые» константы уместны для данной функции:
      • π — если есть тригонометрия (sin/cos/tan/asin/acos/atan/...) ИЛИ
            пользователь сам явно написал pi/π в выражении.
      • e — если есть экспонента/логарифм (exp/log/ln/lg) ИЛИ пользователь
            сам явно написал e в выражении.
    Возвращает (allow_pi, allow_e). Это не даёт, например, точке пересечения
    чисто экспоненциальной функции e^x-5e «прилипнуть» к 5π/6.
    """
    if not func_str:
        return (True, True)
    s = str(func_str).lower()
    import re
    trig = re.search(r'\b(a?sin|a?cos|a?tan|sinh|cosh|tanh|cot|sec|csc)\b', s) is not None
    has_pi_literal = ('pi' in s) or ('π' in s)
    explog = re.search(r'\b(exp|ln|lg)\b', s) is not None or 'log' in s
    # одиночная буква e как литерал (не часть слова): 'e', '2e', 'e^x' и т.п.
    has_e_literal = re.search(r'(?<![A-Za-z])e(?![A-Za-z])', s) is not None
    allow_pi = trig or has_pi_literal
    allow_e  = explog or has_e_literal
    # Чисто алгебраическая функция (без тригонометрии, exp/log и без явных
    # pi/e) — ни π, ни e не имеют здесь смысла, поэтому отключаем оба.
    # Так точки вроде вершины параболы при x≈1.6 и пересечения e^x-5e не
    # «прилипают» к π/e. (Целочисленная привязка работает всегда.)
    return (allow_pi, allow_e)


def snap_to_nice(val, tol_pct=0.02, allow_pi=True, allow_e=True):
    if val == 0:
        return 0.0

    PI = math.pi
    E  = math.e
    PI_E_TOL = 0.004   # 0.4 % — узкий допуск для долей π и кратных e

    # Собираем ВСЕХ подходящих кандидатов (целое, доля π, кратное e), затем
    # выбираем ближайшего. Важно именно сравнивать их между собой, а НЕ брать
    # первого сработавшего: у дальних экстремумов (5π/2≈7.854, 7π/2≈10.996…)
    # ближайшее целое попадает в 2%-коридор и раньше перехватывало π-долю —
    # отсюда «8» вместо «5π/2». Теперь побеждает тот, кто реально ближе.
    candidates = []  # (относительная_ошибка, точное_значение)

    # Целое
    nearest_int = round(val)
    if nearest_int != 0:
        rel = abs(val - nearest_int) / abs(nearest_int)
        if rel <= tol_pct:
            candidates.append((rel, float(nearest_int)))

    # Доли π: k·π/den, den ∈ {1,2,3,4,6}
    if allow_pi:
        for den in (1, 2, 3, 4, 6):
            num = round(val / PI * den)
            if num == 0:
                continue
            exact = num / den * PI
            if abs(exact) > 1e-12:
                rel = abs(val - exact) / abs(exact)
                if rel <= PI_E_TOL:
                    candidates.append((rel, exact))

    # Кратные e
    if allow_e:
        n_e = round(val / E)
        if n_e != 0:
            exact_e = n_e * E
            rel = abs(val - exact_e) / abs(exact_e)
            if rel <= PI_E_TOL:
                candidates.append((rel, exact_e))

    if candidates:
        # ближайший по относительной ошибке
        return min(candidates, key=lambda c: c[0])[1]
    return val


def fmt_num(val, tol=1e-4):
    import math
    from math import gcd
    PI = math.pi
    E  = math.e

    # Кратные π с «хорошими» знаменателями: 1, 2, 3, 4, 6.
    # Проверяем их от большего знаменателя к меньшему и берём первое
    # достаточно точное совпадение в наиболее простой форме.
    ratio_pi = val / PI
    best = None
    for den in (1, 2, 3, 4, 6):
        num = round(ratio_pi * den)
        if num == 0:
            continue
        approx = (num / den) * PI
        if abs(val - approx) / (abs(val) + 1e-12) < tol:
            g = gcd(abs(num), den)
            sn, sd = num // g, den // g
            # Кандидат найден; сохраняем самый простой (наименьший знаменатель)
            if best is None or sd < best[1]:
                best = (sn, sd)
    if best is not None:
        num, den = best
        sign = '-' if num < 0 else ''
        coef = '' if abs(num) == 1 else str(abs(num))
        if den == 1:
            return f'{sign}{coef}π'
        else:
            return f'{sign}{coef}π/{den}'

    ratio_e = val / E
    n_e = round(ratio_e)
    if abs(n_e) > 0 and abs(val - n_e * E) / (abs(val) + 1e-12) < tol:
        if n_e ==  1: return  'e'
        if n_e == -1: return '-e'
        return f'{n_e}e'

    return f'{val:g}'


def parse_number(s, default=None):
    """
    Парсит строку с числом ИЛИ математическим выражением в float.
    Поддерживает: pi/π, e, exp(1), 2*pi, pi/2, sqrt(2), 3+2, -inf, inf и т.п.
    Это позволяет вводить «pi», «2pi», «pi/4» в любые числовые поля
    (сетка, домен, окно просмотра, диапазон заливки), а не только в функции.

    При пустой строке возвращает default (если задан), иначе бросает ValueError.
    """
    if s is None:
        if default is not None:
            return default
        raise ValueError("empty")
    t = str(s).strip().lower()
    if t == "":
        if default is not None:
            return default
        raise ValueError("empty")

    # Бесконечности — отдельно (sympy их тоже понимает, но так быстрее и явнее)
    t_clean = t.replace(" ", "")
    if t_clean in ("inf", "+inf", "infinity", "+infinity", "∞", "+∞"):
        return float("inf")
    if t_clean in ("-inf", "-infinity", "-∞"):
        return float("-inf")

    # Нормализуем удобные пользователю обозначения
    t = t.replace("π", "pi").replace("∞", "oo")
    # «2pi», «3e», «2pi/3» — неявное умножение разрешит sympy (implicit_multiplication)
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        transformations = (standard_transformations +
                           (implicit_multiplication_application, convert_xor))
        # E — символ Эйлера в sympy; 'e' пользователя приводим к нему.
        # Делаем это через локальный словарь, чтобы 'e' не считался символом.
        from sympy import E as _E, pi as _pi
        expr = parse_expr(t, transformations=transformations,
                          local_dict={'e': _E, 'pi': _pi})
        val = float(expr.evalf())
        if not np.isfinite(val):
            # выражения вроде oo
            if expr.is_infinite:
                return float("inf") if expr.is_positive else float("-inf")
        return val
    except Exception:
        # Последняя попытка — обычный float
        return float(t_clean)


def prettify_math_text(text):
    """
    Заменяет в пользовательском тексте 'pi' -> 'π' и одиночное 'e' -> 'e'
    (символ как есть), а также 'inf' -> '∞', чтобы подписи вроде
    '(pi, 2pi)' отображались как '(π, 2π)'.

    Аккуратно: заменяет 'pi' только как отдельный математический токен
    (не внутри слов типа 'epic', 'spin'), опираясь на границы из
    не-буквенных символов. 'e' само по себе НЕ трогаем как букву (слишком
    часто встречается в обычном тексте); конвертируем только связки вида
    '2e', '3e' (число + e), где это явно число Эйлера.
    """
    import re
    if not text:
        return text

    # pi / π как отдельный токен (с возможным числовым коэффициентом перед ним
    # мы не трогаем — само 'pi' превратится в 'π', а '2pi' станет '2π').
    # Граница слева: начало строки или не-буква; справа: конец или не-буква.
    text = re.sub(r'(?<![A-Za-z])pi(?![A-Za-z])', 'π', text)

    # inf -> ∞ (как отдельный токен)
    text = re.sub(r'(?<![A-Za-z])inf(?![A-Za-z])', '∞', text)

    # 'число + e' как число Эйлера: '2e' -> '2e' (оставляем e-символ),
    # но только если это не часть слова и не похоже на экспоненту в духе
    # '1e5'. Требуем, чтобы после e не шла цифра.
    # Здесь по сути ничего не меняем визуально (e и так 'e'), поэтому
    # оставляем как есть — намеренно не трогаем 'e', чтобы не ломать слова.
    return text


def find_extrema_numerical(f, x_vals, y_vals, y_lim_b, y_lim_t, disc_pts_x,
                           allow_pi=True, allow_e=True):
    finite_mask = np.isfinite(y_vals)
    yf = np.where(finite_mask, y_vals, np.nan)

    # Адаптивный order: пропорционален доле конечных точек
    n_finite = int(np.sum(finite_mask))
    n_total  = len(y_vals)
    density  = n_finite / max(n_total, 1)
    order    = max(3, int(10 * density))

    maxima_idx = argrelextrema(np.nan_to_num(yf, nan=-1e18), np.greater, order=order)[0]
    minima_idx = argrelextrema(np.nan_to_num(yf, nan=+1e18), np.less,    order=order)[0]

    # Порог near_disc: 1% от диапазона X, но не более 0.1
    # (был 0.3 — слишком много, отсекал экстремумы рядом с разрывом)
    x_span = x_vals[-1] - x_vals[0]
    disc_tol = min(0.1, x_span * 0.01)

    def near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_pts_x)

    # Исключаем точки у границ NaN-областей (артефакты nan_to_num)
    nan_edges = set()
    for i in range(1, len(finite_mask)):
        if finite_mask[i] != finite_mask[i-1]:
            nan_edges.add(i-1)
            nan_edges.add(i)

    extrema = []
    seen = []
    for idx in list(maxima_idx) + list(minima_idx):
        if idx in nan_edges:
            continue
        ex, ey = x_vals[idx], y_vals[idx]
        if not np.isfinite(ey):
            continue
        if not (y_lim_b <= ey <= y_lim_t):
            continue
        if near_disc(ex):
            continue
        if any(abs(ex - s) < 0.2 for s in seen):
            continue
        seen.append(ex)
        ex = snap_to_nice(round(ex, 4), allow_pi=allow_pi, allow_e=allow_e)
        ey = snap_to_nice(round(ey, 4), allow_pi=allow_pi, allow_e=allow_e)
        extrema.append((ex, ey))
    return extrema


def find_x_intercepts(f, x_vals, y_vals, disc_pts, allow_pi=True, allow_e=True):
    zeros = []
    seen = []
    disc_x = [d[0] if isinstance(d, tuple) else d for d in disc_pts]
    x_span = x_vals[-1] - x_vals[0]
    disc_tol = min(0.1, x_span * 0.01)

    def is_near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_x)

    for i in range(len(y_vals) - 1):
        y0, y1 = y_vals[i], y_vals[i + 1]
        x0, x1 = x_vals[i], x_vals[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 * y1 < 0:
            try:
                xz = brentq(f, x0, x1, xtol=1e-8)
                if not is_near_disc(xz) and not any(abs(xz - s) < 0.1 for s in seen):
                    seen.append(xz)
                    zeros.append(snap_to_nice(round(xz, 4),
                                              allow_pi=allow_pi, allow_e=allow_e))
            except Exception:
                pass

    return sorted(zeros)


def find_y_intercept(f, disc_pts, y_lim_b, y_lim_t):
    if not any(abs(d) < 1e-9 for d in disc_pts):
        val = _safe_val(f, 0)
        if np.isfinite(val) and y_lim_b <= val <= y_lim_t:
            return round(val, 6)
    probes = [_safe_val(f, eps) for eps in [1e-7, -1e-7, 1e-6, -1e-6]]
    probes = [v for v in probes if np.isfinite(v)]
    if probes and max(probes) - min(probes) < 1e-4:
        val = sum(probes) / len(probes)
        if y_lim_b <= val <= y_lim_t:
            return round(val, 6)
    return None


def make_y_array(f, x_vals, disc_pts, y_lim_b, y_lim_t, tol=0.05):
    """Вычисляет y; маскирует окрестность разрывов и выбросы."""
    y_span = y_lim_t - y_lim_b
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        ys = np.array([_safe_val(f, xi) for xi in x_vals], dtype=float)
    for dp in disc_pts:
        ys[np.abs(x_vals - dp) < tol] = np.nan
    ys[np.abs(ys) > y_span * 20] = np.nan
    return ys


def find_intersections(f1, f2, x_vals, y1_vals, y2_vals, disc_pts_x):
    """
    Находит точки пересечения двух функций:
    1. Знакосмены (y1-y2) → Брент (обычные пересечения)
    2. Локальные минимумы |y1-y2| близкие к 0 → касательные точки
    """
    diff = y1_vals - y2_vals
    points = []
    seen = []

    disc_tol = min(0.1, (x_vals[-1] - x_vals[0]) * 0.01)

    def near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_pts_x)

    def add_point(xz):
        if near_disc(xz):
            return
        if any(abs(xz - s) < 0.05 for s in seen):
            return
        yz = _safe_val(f1, xz)
        if np.isfinite(yz):
            seen.append(xz)
            points.append((round(xz, 4), round(yz, 4)))

    def diff_f(x):
        v1 = _safe_val(f1, x)
        v2 = _safe_val(f2, x)
        return v1 - v2 if (np.isfinite(v1) and np.isfinite(v2)) else float('nan')

    abs_diff = np.abs(diff)

    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        x0, x1 = x_vals[i], x_vals[i + 1]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue

        # Случай 1: знакосмена → обычное пересечение
        if d0 * d1 < 0:
            try:
                xz = brentq(diff_f, x0, x1, xtol=1e-8)
                add_point(xz)
            except Exception:
                pass

    # Случай 2: локальный минимум |diff| → касательная точка
    # Ищем точки где |diff| минимально и очень мало
    from scipy.signal import argrelextrema
    finite_mask = np.isfinite(abs_diff)
    if np.sum(finite_mask) > 3:
        # Адаптивный порог: 0.1% от максимального |diff|
        threshold = max(np.nanmax(abs_diff) * 0.001, 1e-6)
        local_mins = argrelextrema(
            np.where(finite_mask, abs_diff, np.inf), np.less, order=3)[0]
        for idx in local_mins:
            if abs_diff[idx] < threshold:
                # Уточняем минимум через minimize_scalar
                try:
                    from scipy.optimize import minimize_scalar
                    i0 = max(0, idx - 5)
                    i1 = min(len(x_vals) - 1, idx + 5)
                    res = minimize_scalar(
                        lambda x: abs(diff_f(x)),
                        bounds=(x_vals[i0], x_vals[i1]),
                        method='bounded'
                    )
                    if res.fun < threshold:
                        add_point(res.x)
                except Exception:
                    add_point(x_vals[idx])

    return points
    y_span = y_lim_t - y_lim_b
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        ys = np.array([_safe_val(f, xi) for xi in x_vals], dtype=float)
    for dp in disc_pts:
        ys[np.abs(x_vals - dp) < tol] = np.nan
    ys[np.abs(ys) > y_span * 20] = np.nan
    return ys


# ══════════════════════════════════════════════════════════════
#  DRAGGABLE ANNOTATIONS
# ══════════════════════════════════════════════════════════════

class DraggableAnnotation:
    """Позволяет перетаскивать текстбокс аннотации мышью."""

    # Общий «замок» захвата на время одного события press: когда несколько
    # подписей лежат друг на друге (например (1,0) и (1.1,0)), без этого
    # КАЖДАЯ из них захватывала бы один и тот же клик и они двигались бы
    # вместе. Здесь первый объект, чей бокс содержит курсор, «забирает»
    # это событие себе, а остальные при том же press видят, что оно уже
    # занято, и не реагируют. Ключ — сам объект события (у каждого клика
    # он свой), чтобы замок автоматически «сбрасывался» на следующем клике.
    _press_claimed_by = {}   # id(event) -> DraggableAnnotation

    def __init__(self, annotation):
        self.ann   = annotation
        self.press = None
        self.fig   = annotation.figure
        self.ann.set_picker(True)
        self.cidpress   = self.fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ann.axes:
            return
        # Если этот же клик уже захвачен другой подписью — не реагируем,
        # чтобы перетаскивалась ровно одна точка, а не все под курсором.
        if id(event) in DraggableAnnotation._press_claimed_by:
            return
        # Проверяем попадание курсора в бокс аннотации
        contains, _ = self.ann.contains(event)
        if not contains:
            return
        # Забираем это событие себе (замок до следующего клика)
        DraggableAnnotation._press_claimed_by[id(event)] = self
        # Подчищаем старые записи, чтобы словарь не рос бесконечно
        if len(DraggableAnnotation._press_claimed_by) > 8:
            for k in list(DraggableAnnotation._press_claimed_by)[:-1]:
                DraggableAnnotation._press_claimed_by.pop(k, None)
        # Запоминаем начальное положение текста (в координатах данных)
        self.press = (event.xdata, event.ydata,
                      self.ann.get_position())  # (mouse_x, mouse_y, ann_xy)

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ann.axes:
            return
        mx0, my0, (ax0, ay0) = self.press
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - mx0
        dy = event.ydata - my0
        # Переводим аннотацию в абсолютные координаты данных
        self.ann.xyann = (ax0 + dx, ay0 + dy)
        self.ann.set_position((ax0 + dx, ay0 + dy))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.fig.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════
#  СВОБОДНЫЕ ПОДПИСИ НА ГРАФИКЕ (добавляются прямо на холсте)
# ══════════════════════════════════════════════════════════════

# Персистентный список свободных подписей — хранится на уровне модуля,
# чтобы переживать повторные построения графика (Plot Graph), как и
# _ACTIVE_DRAGGABLES. Каждый элемент — dict с координатами в данных,
# текстом, поворотом и стилем.
FREE_TEXTS = []

# Держим ссылку на текущий менеджер тут же, на уровне модуля — иначе
# объект (и его подписки на события мыши через mpl_connect) может быть
# собран сборщиком мусора, ровно как было с DraggableAnnotation раньше.
_active_free_text_manager = None


class FreeTextManager:
    """
    Интерактивные подписи прямо на графике:
      • двойной клик / правый клик по пустому месту — добавить подпись
      • левый клик + перетаскивание по существующей подписи — переместить
      • двойной клик по существующей подписи — изменить текст
      • колесо мыши над подписью — повернуть
      • правый клик по подписи — контекстное меню (изменить / повернуть
        в исходное положение / удалить)
    """

    ROTATE_STEP = 10  # градусов за один щелчок колеса

    def __init__(self, fig, ax, font_size=11):
        self.fig = fig
        self.ax = ax
        self.font_size = font_size
        self.entries = []   # [(record_dict, Text_artist), ...]
        self.drag = None

        # Фича: при каждом построении графика ВСЕ подписи получают текущий
        # Label size (font_size). Любой ранее заданный вручную размер
        # сбрасывается — пользователь может снова поменять его после.
        for record in FREE_TEXTS:
            record['fontsize'] = font_size
            self.entries.append((record, self._create_artist(record)))

        self.cid_press   = fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_motion  = fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_scroll  = fig.canvas.mpl_connect('scroll_event',         self.on_scroll)

    # ── Отрисовка ──────────────────────────────────────────
    def _create_artist(self, record):
        return self.ax.text(
            record['x'], record['y'], prettify_math_text(record['text']),
            color=record.get('color', '#000000'),
            fontsize=record.get('fontsize', self.font_size),
            rotation=record.get('rotation', 0),
            ha='center', va='center',
            picker=True,
            zorder=12,
        )

    def _find_hit(self, event):
        for record, artist in self.entries:
            contains, _ = artist.contains(event)
            if contains:
                return record, artist
        return None

    def _tk_root(self):
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    # ── Мышь ────────────────────────────────────────────────
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        hit = self._find_hit(event)

        if event.button == 3:   # правая кнопка — контекстное меню
            self.drag = None
            self._show_context_menu(event, hit)
            return

        if event.button != 1:
            return

        if hit is not None:
            record, artist = hit
            if event.dblclick:
                self.drag = None
                self._edit_text(record, artist)
                return
            self.drag = (record, artist, event.xdata, event.ydata,
                         record['x'], record['y'])
        elif event.dblclick:
            self._add_text_at(event.xdata, event.ydata)

    def on_motion(self, event):
        if self.drag is None or event.inaxes != self.ax:
            return
        record, artist, mx0, my0, x0, y0 = self.drag
        if event.xdata is None or event.ydata is None:
            return
        new_x = x0 + (event.xdata - mx0)
        new_y = y0 + (event.ydata - my0)
        artist.set_position((new_x, new_y))
        record['x'], record['y'] = new_x, new_y
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.drag = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        hit = self._find_hit(event)
        if hit is None:
            return
        record, artist = hit
        step = self.ROTATE_STEP if event.button == 'up' else -self.ROTATE_STEP
        new_rotation = (record.get('rotation', 0) + step) % 360
        artist.set_rotation(new_rotation)
        record['rotation'] = new_rotation
        self.fig.canvas.draw_idle()

    # ── Добавление / редактирование / удаление ──────────────
    def _add_text_at(self, x, y):
        from tkinter import simpledialog
        text = simpledialog.askstring("New label", "Label text:",
                                       parent=self._tk_root())
        if not text:
            return
        record = {'text': text, 'x': x, 'y': y, 'rotation': 0,
                  'color': '#000000', 'fontsize': self.font_size}
        FREE_TEXTS.append(record)
        self.entries.append((record, self._create_artist(record)))
        self.fig.canvas.draw_idle()

    def _edit_text(self, record, artist):
        from tkinter import simpledialog
        new_text = simpledialog.askstring(
            "Edit label", "Label text:",
            initialvalue=record['text'], parent=self._tk_root())
        if new_text is None:
            return
        if new_text == "":
            self._delete(record, artist)
            return
        record['text'] = new_text
        artist.set_text(prettify_math_text(new_text))
        self.fig.canvas.draw_idle()

    def _delete(self, record, artist):
        try:
            artist.remove()
        except Exception:
            pass
        self.entries = [(r, a) for r, a in self.entries if r is not record]
        if record in FREE_TEXTS:
            FREE_TEXTS.remove(record)
        self.fig.canvas.draw_idle()

    def _reset_rotation(self, record, artist):
        record['rotation'] = 0
        artist.set_rotation(0)
        self.fig.canvas.draw_idle()

    def _set_color(self, record, artist):
        from tkinter import colorchooser
        initial = record.get('color', '#000000')
        result = colorchooser.askcolor(
            color=initial, title="Label color", parent=self._tk_root())
        if result is None or result[1] is None:
            return
        record['color'] = result[1]
        artist.set_color(result[1])
        self.fig.canvas.draw_idle()

    def _set_size(self, record, artist, size):
        record['fontsize'] = size
        artist.set_fontsize(size)
        self.fig.canvas.draw_idle()

    def _build_size_submenu(self, parent_menu, record, artist):
        import tkinter as tk
        sub = tk.Menu(parent_menu, tearoff=0)
        for size in (8, 10, 12, 14, 16, 20, 24, 28):
            sub.add_command(
                label=str(size),
                command=lambda s=size: self._set_size(record, artist, s))
        return sub

    def _show_context_menu(self, event, hit):
        import tkinter as tk
        root = self._tk_root()
        if root is None or event.guiEvent is None:
            return
        menu = tk.Menu(root, tearoff=0)
        if hit is not None:
            record, artist = hit
            menu.add_command(label="Edit text",
                              command=lambda: self._edit_text(record, artist))
            menu.add_command(label="Text color...",
                              command=lambda: self._set_color(record, artist))
            menu.add_cascade(label="Text size",
                              menu=self._build_size_submenu(menu, record, artist))
            menu.add_command(label="Reset rotation",
                              command=lambda: self._reset_rotation(record, artist))
            menu.add_separator()
            menu.add_command(label="Delete",
                              command=lambda: self._delete(record, artist))
        else:
            if event.xdata is None or event.ydata is None:
                return
            menu.add_command(
                label="Add label here",
                command=lambda: self._add_text_at(event.xdata, event.ydata))
        try:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()


# ══════════════════════════════════════════════════════════════
#  ПОДПИСИ ОСЕЙ "x" / "y" — тоже интерактивные
# ══════════════════════════════════════════════════════════════

# Персистентные пользовательские настройки подписей осей: смещение от
# "домашнего" положения (в пунктах), цвет и размер. Переживают повторное
# построение графика. dx/dy — это пользовательский сдвиг поверх стандартного
# положения; домашнее положение (само по себе) при этом не меняется.
AXIS_LABELS = {
    'x': {'dx': 0.0, 'dy': 0.0, 'color': None, 'fontsize': None},
    'y': {'dx': 0.0, 'dy': 0.0, 'color': None, 'fontsize': None},
}

_active_axis_label_manager = None


class AxisLabelManager:
    """
    Делает подписи осей "x" и "y" интерактивными:
      • левый клик + перетаскивание — переместить подпись
      • правый клик — контекстное меню: цвет, размер, сбросить положение
    Удалять/добавлять эти подписи нельзя (в отличие от свободных).
    "Домашнее" положение фиксировано; Reset position возвращает к нему.
    """

    def __init__(self, fig, ax, artists, home_offsets, default_fontsize):
        # artists:      {'x': Text, 'y': Text}
        # home_offsets: {'x': (dx_pts, dy_pts), 'y': (...)} — стандартное
        #               смещение в пунктах от точки привязки (xy).
        self.fig = fig
        self.ax = ax
        self.artists = artists
        self.home = home_offsets
        self.default_fontsize = default_fontsize
        self.drag = None

        # Применяем сохранённые пользовательские настройки.
        # ВАЖНО (фича "размер от Label size"): размер при каждом построении
        # сбрасывается на текущий Label size — артист уже создан с этим
        # размером, поэтому просто очищаем сохранённый пользовательский
        # размер. Цвет и смещение (позиция) при этом сохраняются.
        for key, art in artists.items():
            st = AXIS_LABELS[key]
            st['fontsize'] = None
            if st['color']:
                art.set_color(st['color'])
            self._apply_offset(key)
            art.set_picker(True)

        self.cid_press   = fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_motion  = fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)

    def _apply_offset(self, key):
        # Итоговое смещение = домашнее + пользовательский сдвиг (в пунктах)
        from matplotlib.transforms import ScaledTranslation
        hx, hy = self.home[key]
        st = AXIS_LABELS[key]
        self.artists[key].set_transform(
            self.ax.transData +
            ScaledTranslation((hx + st['dx']) / 72.0,
                              (hy + st['dy']) / 72.0,
                              self.fig.dpi_scale_trans))

    def _hit(self, event):
        for key, art in self.artists.items():
            contains, _ = art.contains(event)
            if contains:
                return key
        return None

    def _tk_root(self):
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    def on_press(self, event):
        # ВАЖНО: НЕ требуем event.inaxes == self.ax. Подписи "x"/"y" стоят
        # у самых краёв осей и смещены наружу, поэтому физически находятся
        # ЗА прямоугольником осей — клик по ним приходит с inaxes=None.
        # contains() ниже работает в экранных координатах и не зависит от
        # границ осей, поэтому хит-тестим артисты напрямую.
        if event.x is None or event.y is None:
            return
        key = self._hit(event)
        if key is None:
            return
        if event.button == 3:
            self.drag = None
            self._menu(event, key)
            return
        if event.button == 1:
            self.drag = (key, event.x, event.y,
                         AXIS_LABELS[key]['dx'], AXIS_LABELS[key]['dy'])

    def on_motion(self, event):
        if self.drag is None:
            return
        key, px0, py0, dx0, dy0 = self.drag
        if event.x is None or event.y is None:
            return
        # Перемещение мыши в пикселях -> в пункты (1 пункт = dpi/72 пикселей)
        scale = 72.0 / self.fig.dpi
        AXIS_LABELS[key]['dx'] = dx0 + (event.x - px0) * scale
        AXIS_LABELS[key]['dy'] = dy0 + (event.y - py0) * scale
        self._apply_offset(key)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.drag = None

    # ── Контекстное меню ───────────────────────────────────
    def _set_color(self, key):
        from tkinter import colorchooser
        art = self.artists[key]
        result = colorchooser.askcolor(
            color=art.get_color(), title=f'"{key}" label color',
            parent=self._tk_root())
        if result is None or result[1] is None:
            return
        AXIS_LABELS[key]['color'] = result[1]
        art.set_color(result[1])
        self.fig.canvas.draw_idle()

    def _set_size(self, key, size):
        AXIS_LABELS[key]['fontsize'] = size
        self.artists[key].set_fontsize(size)
        self.fig.canvas.draw_idle()

    def _reset_position(self, key):
        AXIS_LABELS[key]['dx'] = 0.0
        AXIS_LABELS[key]['dy'] = 0.0
        self._apply_offset(key)
        self.fig.canvas.draw_idle()

    def _menu(self, event, key):
        import tkinter as tk
        root = self._tk_root()
        if root is None or event.guiEvent is None:
            return
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label=f'"{key}" label color...',
                          command=lambda: self._set_color(key))
        sub = tk.Menu(menu, tearoff=0)
        for size in (8, 10, 12, 14, 16, 20, 24, 28):
            sub.add_command(label=str(size),
                            command=lambda s=size: self._set_size(key, s))
        menu.add_cascade(label="Label size", menu=sub)
        menu.add_separator()
        menu.add_command(label="Reset position",
                          command=lambda: self._reset_position(key))
        try:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()


# Цвета для кривых (по порядку функций в FUNCS)
CURVE_COLORS = [
    "#000000",  # чёрный
    "#e05c2a",  # оранжевый
    "#2a7ae0",  # синий
    "#2ab850",  # зелёный
    "#9b2ae0",  # фиолетовый
    "#e0b02a",  # жёлтый
    "#e02a6a",  # малиновый
    "#2acce0",  # голубой
]

CURVE_WIDTHS = []   # ширина линий (пусто = 1.8 для всех)
CURVE_STYLES = []   # тип линий: "-", "--", ":" (пусто = "-" для всех)

# Держим ссылки на текущие DraggableAnnotation на уровне МОДУЛЯ, а не внутри
# plot_function(). matplotlib подписывает bound-методы (self.on_press и т.д.)
# через weakref — если на сам объект DraggableAnnotation не остаётся ни одной
# сильной ссылки, сборщик мусора может удалить его, и подписка на события
# тихо умирает. Раньше единственной сильной ссылкой был локальный список
# внутри plot_function(), который "жил" только пока эта функция не вернулась
# (а она блокируется в plt.show() лишь при первом построении графика за сессию —
# при повторном построении plt.show() может вернуться сразу, и список тут же
# терялся вместе со всеми подписками на drag).
_ACTIVE_DRAGGABLES = []

def plot_function():
    global CURVE_WIDTHS, CURVE_STYLES, CURVE_COLORS
    x_span = X_LIM_R - X_LIM_L
    y_span = Y_LIM_T - Y_LIM_B

    ASYM_COLOR   = "#555555"
    GRID_COLOR   = "#cccccc"
    LABEL_COLOR  = "#000000"

    _scale_geom = math.sqrt(x_span * y_span)  # оставляем для совместимости
    _FS       = max(4, min(24, FONT_SIZE))     # зажимаем в разумные пределы
    AXIS_FS   = max(6, int(_FS * 1.1))
    TICK_FS   = max(4, int(_FS * 0.9))

    plt.rcParams['font.family'] = 'Calibri'

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    ax.set_xlim(X_LIM_L, X_LIM_R)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

    # Квадратная сетка при ЛЮБОМ соотношении окна: 'equal' делает одну
    # единицу по X и одну по Y одинаковой длины на экране, поэтому клетки
    # сетки всегда квадратные (при равных шагах X_GRID/Y_GRID).
    # adjustable='box' ужимает рамку осей внутри квадратной фигуры под это
    # соотношение, само окно (фигура) остаётся квадратным.
    ax.set_aspect('equal', adjustable='box')

    # ── Сетка ───────────────────────────────────
    def make_ticks_for_grid(lim_lo, lim_hi, step):
        n_max = int(abs(lim_hi - lim_lo) / step) + 2
        return sorted({round(k * step, 10) for k in range(-n_max, n_max + 1)
                       if lim_lo <= k * step <= lim_hi})

    if GRID:
        x_grid_vals = [v for v in make_ticks_for_grid(X_LIM_L, X_LIM_R, X_GRID)
                       if X_LIM_L + X_GRID - 1e-9 <= v <= X_LIM_R - X_GRID + 1e-9]
        y_grid_vals = [v for v in make_ticks_for_grid(Y_LIM_B, Y_LIM_T, Y_GRID)
                       if Y_LIM_B + Y_GRID - 1e-9 <= v <= Y_LIM_T - Y_GRID + 1e-9]
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_grid_vals))
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_grid_vals))
        ax.grid(True, which='major', color=GRID_COLOR, linewidth=0.6,
                linestyle='-', zorder=1)
    else:
        ax.grid(False)

    # Позиции осей
    x_axis_y = max(Y_LIM_B, min(Y_LIM_T, 0.0))
    y_axis_x = max(X_LIM_L, min(X_LIM_R, 0.0))

    # ── Оси со стрелками ────────────────────────
    ax.annotate("", xy=(X_LIM_R, x_axis_y), xytext=(X_LIM_L, x_axis_y),
                arrowprops=dict(arrowstyle='->', color=LABEL_COLOR,
                                lw=1.2, mutation_scale=12), zorder=4)
    ax.annotate("", xy=(y_axis_x, Y_LIM_T), xytext=(y_axis_x, Y_LIM_B),
                arrowprops=dict(arrowstyle='->', color=LABEL_COLOR,
                                lw=1.2, mutation_scale=12), zorder=4)

    _axis_label_x = ax.text(X_LIM_R, x_axis_y, "x",
                ha='left', va='center', fontsize=AXIS_FS, color=LABEL_COLOR,
                clip_on=False, zorder=6)
    _axis_label_y = ax.text(y_axis_x, Y_LIM_T, "y",
                ha='center', va='bottom', fontsize=AXIS_FS, color=LABEL_COLOR,
                clip_on=False, zorder=6)
    # Домашние смещения (в пунктах) — то, что раньше было в xytext.
    _axis_home_offsets = {'x': (6.0, 0.0), 'y': (0.0, 6.0)}

    # ── Деления на осях ─────────────────────────
    def make_ticks(lim_lo, lim_hi, step):
        n_max = int(abs(lim_hi) / step + abs(lim_lo) / step) + 2
        vals = []
        for k in range(-n_max, n_max + 1):
            v = round(k * step, 10)
            if (lim_lo + step - 1e-9 <= v <= lim_hi - step + 1e-9
                    and abs(v) > 1e-9):
                vals.append(v)
        return sorted(set(vals))

    for v in make_ticks(X_LIM_L, X_LIM_R, X_GRID):
        if not X_HIDE:
            ax.plot(v, x_axis_y, '|', color=LABEL_COLOR, markersize=4, markeredgewidth=0.8, zorder=5)
            ax.annotate(fmt_num(v), xy=(v, x_axis_y),
                        xytext=(0, -6), textcoords='offset points',
                        ha='center', va='top', fontsize=TICK_FS, color=LABEL_COLOR)

    for v in make_ticks(Y_LIM_B, Y_LIM_T, Y_GRID):
        if not Y_HIDE:
            ax.plot(y_axis_x, v, '_', color=LABEL_COLOR, markersize=4, markeredgewidth=0.8, zorder=5)
            ax.annotate(fmt_num(v), xy=(y_axis_x, v),
                        xytext=(-6, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=TICK_FS, color=LABEL_COLOR)

    # ── Вспомогательные функции подписи ──────────
    def fmt2(v):
        if abs(v) < 1e-12:
            return '0'
        # fmt_num уже распознаёт кратные π (знаменатели 1,2,3,4,6) и e.
        # Если оно вернуло форму с π или e — используем её.
        s = fmt_num(v)
        if 'π' in s or 'e' in s:
            return s
        s = f'{v:.2f}'
        return s.rstrip('0').rstrip('.')

    draggables = _ACTIVE_DRAGGABLES
    draggables.clear()   # отпускаем подписи предыдущего графика

    # Множество уже подписанных точек (округлённые координаты), чтобы одна и
    # та же точка не подписывалась дважды — например (1, 0) у (x-1)(x+2) и у
    # (x-1). Сбрасывается на каждое построение. Ключ — округлённые (x, y).
    _annotated_points = set()
    _marked_points = set()

    def _point_key(px, py):
        return (round(float(px), 3), round(float(py), 3))

    def mark_point(px, py, color, **kw):
        """Рисует кружок-маркер точки, но только если такой ещё не рисовали."""
        key = _point_key(px, py)
        if key in _marked_points:
            return False
        _marked_points.add(key)
        opts = dict(markersize=4, zorder=9)
        opts.update(kw)
        ax.plot(px, py, 'o', color=color, **opts)
        return True

    def annotate_point(px, py, label, above, color, side=None):
        # Дедупликация: если такая точка уже подписана — пропускаем.
        key = _point_key(px, py)
        if key in _annotated_points:
            return None
        _annotated_points.add(key)

        near_y = abs(px - y_axis_x) < x_span * 0.12
        BBOX   = dict(boxstyle='round,pad=0.3', fc='none',
                      ec='none', alpha=0.95)

        # Переводим offset points → единицы данных для xytext
        # чтобы drag работал в системе координат данных
        pt_to_x = x_span / (fig.get_size_inches()[0] * 72)
        pt_to_y = y_span / (fig.get_size_inches()[1] * 72)

        if side is not None:
            dx_pt = 8 if side == 'right' else -8
            ha    = 'left' if side == 'right' else 'right'
            txt_x = px + dx_pt * pt_to_x
            txt_y = py + 6 * pt_to_y
        else:
            dy_pt = 8 if above else -8
            va    = 'bottom' if above else 'top'
            txt_x = px
            txt_y = py + dy_pt * pt_to_y

        ha  = ha  if side is not None else 'center'
        va  = 'bottom' if (side is not None or above) else 'top'

        ann = ax.annotate(
            label,
            xy=(px, py),
            xytext=(txt_x, txt_y),
            xycoords='data',
            textcoords='data',
            ha=ha, va=va,
            fontsize=_FS, color=color,
            bbox=BBOX,
            annotation_clip=False,
            zorder=10,
        )
        draggables.append(DraggableAnnotation(ann))
        return ann

    def choose_side(px, py, f):
        # Зондируем на фиксированное расстояние в данных: 15% x_span
        probe = x_span * 0.15
        pts_r = [_safe_val(f, px + probe * k) for k in [0.3, 0.6, 1.0]]
        pts_l = [_safe_val(f, px - probe * k) for k in [0.3, 0.6, 1.0]]
        dist_r = np.nanmean([abs(v - py) for v in pts_r if np.isfinite(v)] or [0])
        dist_l = np.nanmean([abs(v - py) for v in pts_l if np.isfinite(v)] or [0])
        return 'right' if dist_r >= dist_l else 'left'

    ON_Y_AXIS_TOL = x_span * 0.05
    x_vals = np.linspace(X_LIM_L, X_LIM_R, 6000)

    # ══════════════════════════════════════════════
    #  ЦИКЛ ПО ФУНКЦИЯМ
    # ══════════════════════════════════════════════
    all_vasymps  = set()
    all_hasymps  = set()
    func_data    = []   # [(f, y_vals, color), ...] — для заливки после цикла
    # Параллельный список метаданных кривой (та же длина/порядок, что func_data)
    # для расчёта пересечений между разными типами кривых. Каждый элемент —
    # dict: {'kind','color', и колбэки}. kind ∈ 'func' | 'implicit' | 'vline'.
    curve_meta   = []

    for func_idx, func_str in enumerate(FUNCS):
        color = CURVE_COLORS[func_idx % len(CURVE_COLORS)]

        kind, payload = _parse_equation_input(func_str)
        lw = CURVE_WIDTHS[func_idx] if func_idx < len(CURVE_WIDTHS) else 1.8
        ls = CURVE_STYLES[func_idx] if func_idx < len(CURVE_STYLES) else "-"

        # ── Вертикальная линия  x = c ───────────────────────────
        if kind == 'vline':
            cx = payload
            if cx is not None and np.isfinite(cx):
                # Учитываем домен функции (если задан) как ограничение по Y:
                # домен для вертикали трактуем как диапазон Y, на котором
                # её рисовать. По умолчанию — весь видимый Y.
                y_lo_line, y_hi_line = Y_LIM_B, Y_LIM_T
                if func_idx < len(FUNC_DOMAINS):
                    d_from, d_to = FUNC_DOMAINS[func_idx]
                    if d_from != float("-inf"):
                        y_lo_line = max(y_lo_line, d_from)
                    if d_to != float("inf"):
                        y_hi_line = min(y_hi_line, d_to)
                if X_LIM_L <= cx <= X_LIM_R and y_lo_line < y_hi_line:
                    ax.plot([cx, cx], [y_lo_line, y_hi_line],
                            color=color, linewidth=lw, linestyle=ls, zorder=5)
                    # Подпись точки пересечения с осью X: (c, 0)
                    if X_TAG and Y_LIM_B <= 0 <= Y_LIM_T:
                        vp, ve = detect_const_context(func_str)
                        cx_s = snap_to_nice(round(cx, 4), allow_pi=vp, allow_e=ve)
                        if SHOW_VALUES:
                            annotate_point(cx_s, 0.0, f'({fmt2(cx_s)}, 0)',
                                           above=True, color=color)
                        mark_point(cx_s, 0, color, markersize=5, zorder=8)
            # placeholder, чтобы индексы заливки (f1/f2) совпадали с FUNCS
            func_data.append((None, None, color))
            curve_meta.append({'kind': 'vline', 'color': color,
                               'x': cx if (payload is not None) else None})
            continue

        # ── Неявная кривая  F(x, y) = G(x, y) ───────────────────
        if kind == 'implicit':
            lhs_str, rhs_str = payload
            h = None
            try:
                _H_sym, h = build_implicit_func(lhs_str, rhs_str)
                # Сетка по видимому окну. Плотность подобрана как компромисс
                # гладкость/скорость; contour сам интерполирует линию уровня 0.
                N = 600
                xs = np.linspace(X_LIM_L, X_LIM_R, N)
                ys = np.linspace(Y_LIM_B, Y_LIM_T, N)
                X, Y = np.meshgrid(xs, ys)
                with np.errstate(all='ignore'):
                    Z = h(X, Y)
                    Z = np.asarray(Z, dtype=float)
                # Рисуем линию уровня 0 => кривую F−G=0
                # linestyles matplotlib ожидает 'solid'/'dashed'/'dotted'
                ls_map = {'-': 'solid', '--': 'dashed', ':': 'dotted', '-.': 'dashdot'}
                ax.contour(X, Y, Z, levels=[0], colors=[color],
                           linewidths=lw, linestyles=[ls_map.get(ls, 'solid')],
                           zorder=5)

                # ── Пересечения неявной кривой с осями ──────────
                # Ось X: корни H(x, 0)=0 -> точки (x, 0)
                # Ось Y: корни H(0, y)=0 -> точки (0, y)
                ip, ie = detect_const_context(lhs_str + ' ' + rhs_str)
                if X_TAG and (Y_LIM_B <= 0 <= Y_LIM_T):
                    gx = lambda t: h(t, 0.0)
                    for xr in _scan_roots_1d(gx, X_LIM_L, X_LIM_R):
                        xs_ = snap_to_nice(round(xr, 4), allow_pi=ip, allow_e=ie)
                        mark_point(xs_, 0.0, color, markersize=5, zorder=8)
                        if SHOW_VALUES:
                            annotate_point(xs_, 0.0, f'({fmt2(xs_)}, 0)',
                                           above=True, color=color)
                if Y_TAG and (X_LIM_L <= 0 <= X_LIM_R):
                    gy = lambda t: h(0.0, t)
                    for yr in _scan_roots_1d(gy, Y_LIM_B, Y_LIM_T):
                        ys_ = snap_to_nice(round(yr, 4), allow_pi=ip, allow_e=ie)
                        mark_point(0.0, ys_, color, markersize=5, zorder=8)
                        if SHOW_VALUES:
                            annotate_point(0.0, ys_, f'(0, {fmt2(ys_)})',
                                           above=True, color=color, side='right')
            except Exception:
                # Некорректное выражение — тихо пропускаем эту кривую,
                # не роняя весь график.
                h = None
            func_data.append((None, None, color))
            curve_meta.append({'kind': 'implicit', 'color': color, 'h': h})
            continue

        # Обычная функция (для 'y = f(x)' префикс уже снят)
        func_str = payload
        expr, expr_raw, f = build_numpy_func(func_str)
        allow_pi, allow_e = detect_const_context(func_str)

        all_disc   = find_discontinuities_numerical(f, expr, expr_raw, X_LIM_L, X_LIM_R) if (DISC or ASIMP) else []
        vasymps, removable = classify_discontinuities(f, expr, all_disc, Y_LIM_B, Y_LIM_T)
        hasymps    = find_horizontal_asymptotes(f, expr, X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T) if ASIMP else []

        disc_pts_x = [d[0] if isinstance(d, tuple) else d for d in all_disc]

        # Разрыв в самой линии графика делаем только там, где он математически
        # обязателен — на вертикальных асимптотах (функция там не определена).
        # На устранимых разрывах (holes) разрыв линии рисуем ТОЛЬКО если включено
        # отображение Holes — иначе галочка Asymptotes не должна влиять на holes.
        removable_x = [r[0] for r in removable]
        mask_pts_x  = list(vasymps) + (removable_x if DISC else [])
        y_vals      = make_y_array(f, x_vals, mask_pts_x, Y_LIM_B, Y_LIM_T)

        # Ограничение области определения (domain) этой функции: вне
        # [d_from, d_to] делаем y = NaN, поэтому кривая там не рисуется,
        # а экстремумы/нули (они берутся из y_vals) автоматически тоже
        # не считаются за пределами домена. View window при этом не меняется.
        if func_idx < len(FUNC_DOMAINS):
            d_from, d_to = FUNC_DOMAINS[func_idx]
            if d_from != float("-inf") or d_to != float("inf"):
                outside = (x_vals < d_from) | (x_vals > d_to)
                y_vals = y_vals.copy()
                y_vals[outside] = np.nan

        extrema      = find_extrema_numerical(f, x_vals, y_vals, Y_LIM_B, Y_LIM_T, disc_pts_x,
                                              allow_pi=allow_pi, allow_e=allow_e) if EXTR else []
        x_intercepts = find_x_intercepts(f, x_vals, y_vals, disc_pts_x,
                                         allow_pi=allow_pi, allow_e=allow_e) if X_TAG else []
        y_intercept  = find_y_intercept(f, disc_pts_x, Y_LIM_B, Y_LIM_T) if Y_TAG else None
        # y-пересечение (x=0) скрываем, если 0 вне домена функции
        if y_intercept is not None and func_idx < len(FUNC_DOMAINS):
            d_from, d_to = FUNC_DOMAINS[func_idx]
            if not (d_from <= 0 <= d_to):
                y_intercept = None

        # Асимптоты — рисуем только новые (чтобы не дублировать)
        if ASIMP:
            for ya in hasymps:
                if ya not in all_hasymps:
                    ax.axhline(ya, color=ASYM_COLOR, linewidth=1.0, linestyle='--', zorder=3)
                    all_hasymps.add(ya)
            for xa in vasymps:
                if xa not in all_vasymps:
                    ax.axvline(xa, color=ASYM_COLOR, linewidth=1.0, linestyle='--', zorder=3)
                    all_vasymps.add(xa)

        # ── Кривая ──────────────────────────────
        ax.plot(x_vals, y_vals, color=color, linewidth=lw, linestyle=ls, zorder=5)

        # Сохраняем для заливки после цикла
        func_data.append((f, y_vals, color))
        _dom = FUNC_DOMAINS[func_idx] if func_idx < len(FUNC_DOMAINS) else (float('-inf'), float('inf'))
        curve_meta.append({'kind': 'func', 'color': color, 'f': f, 'domain': _dom})

        # ── Точки разрыва ───────────────────────
        if DISC:
            for (dp, ylim) in removable:
                ax.plot(dp, ylim, 'o', color=color, markersize=5,
                        markerfacecolor='white', markeredgewidth=1.5, zorder=8)
                if SHOW_VALUES:
                    if abs(dp - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(dp, ylim, f"({fmt2(dp)}, {fmt2(ylim)})",
                                       above=True, color=color, side=choose_side(dp, ylim, f))
                    else:
                        annotate_point(dp, ylim, f"({fmt2(dp)}, {fmt2(ylim)})",
                                       above=(ylim >= x_axis_y), color=color)

        # ── Экстремумы ───────────────────────────
        if EXTR:
            for ex, ey in extrema:
                mark_point(ex, ey, color)
                if SHOW_VALUES:
                    if abs(ex - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(ex, ey, f"({fmt2(ex)}, {fmt2(ey)})",
                                       above=True, color=color, side=choose_side(ex, ey, f))
                    else:
                        annotate_point(ex, ey, f"({fmt2(ex)}, {fmt2(ey)})",
                                       above=(ey >= x_axis_y), color=color)

        # ── Пересечения с осью X ─────────────────
        if X_TAG:
            for xi in x_intercepts:
                mark_point(xi, x_axis_y, color)
                if SHOW_VALUES:
                    if abs(xi - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(xi, x_axis_y, f"({fmt2(xi)}, 0)",
                                       above=True, color=color, side=choose_side(xi, x_axis_y, f))
                    else:
                        annotate_point(xi, x_axis_y, f"({fmt2(xi)}, 0)",
                                       above=True, color=color)

        # ── Пересечение с осью Y ─────────────────
        removable_x = [r[0] for r in removable]
        is_removable_at_0 = any(abs(d - y_axis_x) < 1e-9 for d in removable_x)
        if Y_TAG and y_intercept is not None and not is_removable_at_0:
            mark_point(y_axis_x, y_intercept, color)
            if SHOW_VALUES:
                annotate_point(y_axis_x, y_intercept, f"(0, {fmt2(y_intercept)})",
                               above=True, color=color,
                               side=choose_side(y_axis_x, y_intercept, f))

    # ══════════════════════════════════════════════
    #  ПЕРЕСЕЧЕНИЯ МЕЖДУ ФУНКЦИЯМИ
    # ══════════════════════════════════════════════
    if INTER and len(curve_meta) >= 2:
        def blend(c1, c2):
            c1 = c1.lstrip('#'); c2 = c2.lstrip('#')
            r = (int(c1[0:2],16) + int(c2[0:2],16)) // 2
            g = (int(c1[2:4],16) + int(c2[2:4],16)) // 2
            b = (int(c1[4:6],16) + int(c2[4:6],16)) // 2
            return f'#{r:02x}{g:02x}{b:02x}'

        def emit(ix, iy, pt_color, probe_f=None):
            if not (Y_LIM_B <= iy <= Y_LIM_T and X_LIM_L <= ix <= X_LIM_R):
                return
            mark_point(ix, iy, pt_color, markersize=5, zorder=11)
            if SHOW_VALUES:
                above = iy >= x_axis_y
                if probe_f is not None and abs(ix - y_axis_x) < ON_Y_AXIS_TOL:
                    annotate_point(ix, iy, f"({fmt2(ix)}, {fmt2(iy)})",
                                   above=above, color=pt_color,
                                   side=choose_side(ix, iy, probe_f))
                else:
                    annotate_point(ix, iy, f"({fmt2(ix)}, {fmt2(iy)})",
                                   above=above, color=pt_color)

        for i in range(len(curve_meta)):
            for j in range(i + 1, len(curve_meta)):
                mi, mj = curve_meta[i], curve_meta[j]
                ki, kj = mi['kind'], mj['kind']
                pt_color = blend(mi['color'], mj['color'])

                # ── функция × функция (как раньше) ──
                if ki == 'func' and kj == 'func':
                    yi = func_data[i][1]; yj = func_data[j][1]
                    for ix, iy in find_intersections(mi['f'], mj['f'], x_vals, yi, yj, disc_pts_x):
                        emit(ix, iy, pt_color, probe_f=mi['f'])

                # ── функция × неявная ──
                elif {ki, kj} == {'func', 'implicit'}:
                    fm = mi if ki == 'func' else mj
                    im = mi if ki == 'implicit' else mj
                    if im.get('h') is None:
                        continue
                    for ix, iy in intersect_func_implicit(fm['f'], im['h'],
                                                          X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T):
                        emit(ix, iy, pt_color, probe_f=fm['f'])

                # ── неявная × неявная (в т.ч. две окружности) ──
                elif ki == 'implicit' and kj == 'implicit':
                    if mi.get('h') is None or mj.get('h') is None:
                        continue
                    for ix, iy in intersect_implicit_implicit(mi['h'], mj['h'],
                                                              X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T):
                        emit(ix, iy, pt_color)

                # ── вертикаль × (функция или неявная) ──
                elif 'vline' in (ki, kj):
                    vm = mi if ki == 'vline' else mj
                    om = mi if ki != 'vline' else mj
                    cx = vm.get('x')
                    if cx is None or not (X_LIM_L <= cx <= X_LIM_R):
                        continue
                    if om['kind'] == 'func':
                        yv = _safe_scalar(om['f'], cx)
                        if np.isfinite(yv):
                            emit(cx, yv, pt_color, probe_f=om['f'])
                    elif om['kind'] == 'implicit' and om.get('h') is not None:
                        # на вертикали x=cx ищем корни H(cx, y)=0
                        for yr in _scan_roots_1d(lambda t: om['h'](cx, t), Y_LIM_B, Y_LIM_T):
                            emit(cx, yr, pt_color)

    def lighten_color(hex_color, factor=0.45):
        hex_color = hex_color.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def draw_fill_lines(ax, xf, y1_fill, y2_fill, valid, fill_style, color,
                        x_lim_l, x_lim_r, y_lim_b, y_lim_t, density=0.01):
        x_span = x_lim_r - x_lim_l
        y_span = y_lim_t - y_lim_b

        step = min(x_span, y_span) * density
        lw   = 0.6

        # Стиль штриховки:
        #   0 — 45°,  сплошная диагональная линия
        #   1 — 135°, сплошная диагональная линия
        #   2 — "точки" — та же диагональная решётка (семейство параллельных
        #       линий + та же логика клиппинга по границе), что и у 45°,
        #       но вместо непрерывной линии на каждой диагонали ставятся
        #       отдельные точки на ФИКСИРОВАННОЙ глобальной сетке с шагом,
        #       зависящим от density. Сетка не зависит от формы границы —
        #       поэтому точки не "облипают" контур функции — и расстояние
        #       между любыми соседними точками (что вдоль диагонали, что
        #       между соседними диагоналями) всегда одинаковое.
        sign = -1 if fill_style == 1 else 1   # стили 0 и 2 — одно семейство 45°

        y_lo = np.minimum(y1_fill, y2_fill)
        y_hi = np.maximum(y1_fill, y2_fill)

        # Диапазон c должен покрыть все углы видимой области
        corners_c = [
            y_lim_b - sign * x_lim_l,
            y_lim_b - sign * x_lim_r,
            y_lim_t - sign * x_lim_l,
            y_lim_t - sign * x_lim_r,
        ]
        c_min = min(corners_c) - step
        c_max = max(corners_c) + step
        c_vals = np.arange(c_min, c_max, step * np.sqrt(2))

        if fill_style == 2:
            # Шаг по x, который при движении вдоль линии с |наклон|=1
            # даёт ровно евклидово расстояние `step` между точками.
            dot_dx = max(step / np.sqrt(2), 1e-9)
            # Привязываем фазу решётки к глобальной системе координат
            # (x_lim_l), а не к границам конкретной заливки — так точки
            # не "прыгают" при изменении from/to.
            n0 = int(np.ceil((xf[0] - x_lim_l) / dot_dx))
            x_start = x_lim_l + n0 * dot_dx
            xs_lat = np.arange(x_start, xf[-1] + dot_dx * 0.5, dot_dx)
            # Оставляем только узлы решётки, строго попадающие в диапазон x
            # самой заливки. Без этого крайний узел мог оказаться чуть за
            # пределами [xf[0], xf[-1]] (особенно при малой плотности —
            # большом dot_dx), а np.clip ниже "притягивал" его индекс к
            # крайнему столбцу xf, и точка проходила проверку границ заливки,
            # хотя по x была уже вне окраса — отсюда точки за рамкой.
            xs_lat = xs_lat[(xs_lat >= xf[0]) & (xs_lat <= xf[-1])]
            if len(xs_lat) == 0:
                return
            idx = np.clip(np.searchsorted(xf, xs_lat), 0, len(xf) - 1)

            for c in c_vals:
                y_lat = sign * xs_lat + c
                ok = (valid[idx] &
                      (y_lat >= y_lo[idx]) & (y_lat <= y_hi[idx]))
                if np.any(ok):
                    ax.plot(xs_lat[ok], y_lat[ok], '.', color=color,
                            markersize=2.2, linewidth=0, zorder=4)
            return

        # 45° (style 0) и 135° (style 1) — сплошные параллельные диагональные линии
        for c in c_vals:
            # Линия: y = sign*x + c
            y_line = sign * xf + c

            # Клипируем линию по области заливки [y_lo, y_hi]
            y_clipped = np.where(
                valid & (y_line >= y_lo) & (y_line <= y_hi),
                y_line, np.nan
            )

            # Разбиваем на непрерывные куски и рисуем каждый
            seg_x, seg_y = [], []
            for xi, yi in zip(xf, y_clipped):
                if np.isfinite(yi):
                    seg_x.append(xi)
                    seg_y.append(yi)
                else:
                    if len(seg_x) > 1:
                        ax.plot(seg_x, seg_y, color=color,
                                linewidth=lw, zorder=4)
                    seg_x, seg_y = [], []
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, color=color,
                        linewidth=lw, zorder=4)

    for fill_entry in FILL:
        # Формат: (f1, f2, x_от, x_до, стиль, границы, плотность)
        f1_idx, f2, fx_from, fx_to, fill_style = fill_entry[:5]
        show_borders = fill_entry[5] if len(fill_entry) > 5 else True
        density      = fill_entry[6] if len(fill_entry) > 6 else 0.01

        if f1_idx >= len(func_data):
            continue

        f1_func, y1_all, color1 = func_data[f1_idx]
        # Заливка относительно вертикальной линии не имеет смысла — пропускаем.
        if f1_func is None:
            continue

        fx_from = max(fx_from, X_LIM_L)
        fx_to   = min(fx_to,   X_LIM_R)
        if fx_from >= fx_to:
            continue

        mask = (x_vals >= fx_from) & (x_vals <= fx_to)
        xf   = x_vals[mask]

        y1_fill = np.array([_safe_val(f1_func, x) for x in xf])
        y1_fill = np.clip(y1_fill, Y_LIM_B, Y_LIM_T)

        if isinstance(f2, str) or f2 is None:
            y2_fill = np.zeros_like(y1_fill)
            fill_color = lighten_color(color1)
        else:
            if f2 >= len(func_data):
                continue
            f2_func = func_data[f2][0]
            if f2_func is None:
                continue
            y2_fill = np.array([_safe_val(f2_func, x) for x in xf])
            y2_fill = np.clip(y2_fill, Y_LIM_B, Y_LIM_T)
            fill_color = lighten_color(color1)

        valid = np.isfinite(y1_fill) & np.isfinite(y2_fill)

        # Рисуем штриховку реальными линиями
        draw_fill_lines(ax, xf, y1_fill, y2_fill, valid, fill_style,
                        fill_color, X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T,
                        density=density)

        # Вертикальные линии-границы по краям
        if show_borders:
            for xb in [fx_from, fx_to]:
                y1b = _safe_val(f1_func, xb)
                y1b = float(np.clip(y1b, Y_LIM_B, Y_LIM_T)) if np.isfinite(y1b) else None
                if isinstance(f2, str) or f2 is None:
                    y2b = 0.0
                else:
                    y2b = _safe_val(func_data[f2][0], xb)
                    y2b = float(np.clip(y2b, Y_LIM_B, Y_LIM_T)) if np.isfinite(y2b) else None
                if y1b is not None and y2b is not None:
                    ax.plot([xb, xb], [y1b, y2b], color=fill_color,
                            linewidth=1.2, zorder=5)

    global _active_free_text_manager
    _active_free_text_manager = FreeTextManager(fig, ax, font_size=_FS)

    global _active_axis_label_manager
    _active_axis_label_manager = AxisLabelManager(
        fig, ax,
        artists={'x': _axis_label_x, 'y': _axis_label_y},
        home_offsets=_axis_home_offsets,
        default_fontsize=AXIS_FS)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_function()
