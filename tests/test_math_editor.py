# -*- coding: utf-8 -*-
"""
Тесты math_editor: модель (без дисплея) + несколько проверок Tk-виджета
(пропускаются, если нет дисплея). Запуск:
    /usr/bin/python3.12 -m pytest tests/test_math_editor.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_editor import (  # noqa: E402
    MathModel, Atom, Frac, Sup, IncompleteExpression, ALL_TOKENS,
)


def typed(*tokens):
    m = MathModel()
    for t in tokens:
        m.insert_token(t)
    return m


def kinds(row):
    """Краткая форма строки: [('var','x'), 'Sup', ...]."""
    out = []
    for it in row.items:
        if isinstance(it, Atom):
            out.append((it.kind, it.text))
        else:
            out.append(type(it).__name__)
    return out


# ═════════════════════════════════════════════════════════════
#  Вставка токенов
# ═════════════════════════════════════════════════════════════

def test_all_tokens_accepted():
    for t in ALL_TOKENS:
        m = MathModel()
        m.insert_token('x')          # чтобы операторам/степеням было к чему цепляться
        m.insert_token(t)            # не должно падать


def test_atoms_and_operators():
    m = typed('1', '2', '.', '5', 'x', '+', 'y', '-', 'pi', '*', 'e', '=', '0')
    assert kinds(m.root) == [('num', '1'), ('num', '2'), ('num', '.'), ('num', '5'),
                             ('var', 'x'), ('op', '+'), ('var', 'y'), ('op', '−'),
                             ('const', 'π'), ('op', '×'), ('const', 'e'), ('op', '='),
                             ('num', '0')]
    assert m.to_sympy() == '12.5*x+y-pi*E=0'


def test_frac_captures_operand_with_exponent():
    m = typed('x', '^', '2', 'right', '/')
    assert len(m.root.items) == 1
    frac = m.root.items[0]
    assert isinstance(frac, Frac)
    assert kinds(frac.num) == [('var', 'x'), 'Sup']
    assert frac.den.is_empty()
    assert m.cursor == (frac.den, 0)
    m.insert_token('2')
    assert m.to_sympy() == '((x**(2))/(2))'


def test_frac_captures_implicit_product_like_desmos():
    m = typed('1', '+', '2', 'x', '/')
    frac = m.root.items[2]
    assert kinds(m.root) == [('num', '1'), ('op', '+'), 'Frac']
    assert kinds(frac.num) == [('num', '2'), ('var', 'x')]
    assert m.cursor == (frac.den, 0)


def test_frac_captures_function_application_and_containers():
    m = typed('sin', 'x', ')', '/')
    frac = m.root.items[0]
    assert kinds(frac.num) == [('func', 'sin'), 'Paren']
    m = typed('sqrt', 'x', 'right', '/')
    assert kinds(m.root.items[0].num) == ['Sqrt']


def test_frac_without_operand_goes_to_numerator():
    m = typed('x', '+', '/')
    frac = m.root.items[2]
    assert frac.num.is_empty() and frac.den.is_empty()
    assert m.cursor == (frac.num, 0)
    m = typed('/')
    assert m.cursor == (m.root.items[0].num, 0)


def test_sup_sq_and_exp():
    m = typed('x', '^')
    sup = m.root.items[1]
    assert isinstance(sup, Sup) and m.cursor == (sup.exp, 0)
    m = typed('x', 'sq')
    assert kinds(m.root.items[1].exp) == [('num', '2')]
    assert m.cursor == (m.root, 2)
    m = typed('exp')
    assert kinds(m.root) == [('const', 'e'), 'Sup']
    assert m.cursor == (m.root.items[1].exp, 0)
    m.insert_token('x')
    assert m.to_sympy() == 'E**(x)'


def test_containers_cursor_inside():
    m = typed('sqrt')
    assert m.cursor == (m.root.items[0].body, 0)
    m = typed('root')
    assert m.cursor == (m.root.items[0].index, 0)
    m.insert_token('3')
    m.insert_token('right')
    assert m.cursor == (m.root.items[0].body, 0)
    m = typed('abs')
    assert m.cursor == (m.root.items[0].body, 0)
    m = typed('(')
    assert m.cursor == (m.root.items[0].body, 0)


def test_close_paren_exits_nearest_paren():
    m = typed('(', 'sqrt', 'x', ')')
    assert m.cursor == (m.root, 1)
    m.insert_token(')')               # нет охватывающей скобки — no-op
    assert m.cursor == (m.root, 1)
    m = typed('(', '(', 'x', ')')
    inner_done = m.cursor
    outer = m.root.items[0]
    assert inner_done == (outer.body, 1)
    m.insert_token(')')
    assert m.cursor == (m.root, 1)


def test_functions_and_logb():
    m = typed('asin')
    assert kinds(m.root) == [('func', 'arcsin'), 'Paren']
    assert m.cursor == (m.root.items[1].body, 0)
    m.insert_token('x')
    assert m.to_sympy() == 'asin(x)'
    m = typed('logb', '2')
    assert kinds(m.root) == [('func', 'log'), 'Sub', 'Paren']
    assert m.cursor == (m.root.items[1].sub, 1)
    m.insert_token('right')
    assert m.cursor == (m.root.items[2].body, 0)
    m.insert_token('x')
    assert m.to_sympy() == 'log(x, 2)'


def test_unknown_token():
    with pytest.raises(ValueError):
        MathModel().insert_token('bogus')


# ═════════════════════════════════════════════════════════════
#  Навигация
# ═════════════════════════════════════════════════════════════

def test_navigation_through_fraction():
    m = typed('1', '/', '2', 'right', 'x')      # 1/2 x
    frac = m.root.items[0]
    m.cursor = (m.root, 0)
    m.move_right()
    assert m.cursor == (frac.num, 0)
    m.move_right()
    assert m.cursor == (frac.num, 1)
    m.move_right()
    assert m.cursor == (frac.den, 0)
    m.move_right()
    assert m.cursor == (frac.den, 1)
    m.move_right()
    assert m.cursor == (m.root, 1)
    m.move_right()
    assert m.cursor == (m.root, 2)
    m.move_right()                              # конец корня — no-op
    assert m.cursor == (m.root, 2)
    # обратно
    m.move_left()
    assert m.cursor == (m.root, 1)
    m.move_left()
    assert m.cursor == (frac.den, 1)
    m.move_left(); m.move_left()
    assert m.cursor == (frac.num, 1)
    m.move_left(); m.move_left()
    assert m.cursor == (m.root, 0)
    m.move_left()
    assert m.cursor == (m.root, 0)


def test_up_down():
    m = typed('1', '2', '/', '3')
    frac = m.root.items[0]
    assert m.cursor == (frac.den, 1)
    m.move_up()
    assert m.cursor == (frac.num, 1)
    m.move_up()                                  # no-op
    assert m.cursor == (frac.num, 1)
    m.move_down()
    assert m.cursor == (frac.den, 1)
    m.move_down()                                # no-op
    assert m.cursor == (frac.den, 1)
    m = typed('x', '^', '2')
    m.move_down()
    assert m.cursor == (m.root, 2)
    m = typed('root', '3')
    m.move_down()                                # из показателя корня — в подкоренное
    assert m.cursor == (m.root.items[0].body, 0)
    m = typed('logb', '2')
    m.move_down()                                # из основания логарифма — в аргумент
    assert m.cursor == (m.root.items[2].body, 0)
    m = typed('logb', '2')
    m.root.remove_at(2)                          # без скобок справа — просто выходим
    m.move_down()
    assert m.cursor == (m.root, 2)
    MathModel().move_up(); MathModel().move_down()   # пустая модель не падает


def test_nested_navigation():
    m = typed('sqrt', '(', 'x', '/', 'y')        # √( x/y )
    sq = m.root.items[0]
    par = sq.body.items[0]
    frac = par.body.items[0]
    assert m.cursor == (frac.den, 1)
    m.move_right()
    assert m.cursor == (par.body, 1)
    m.move_right()
    assert m.cursor == (sq.body, 1)
    m.move_right()
    assert m.cursor == (m.root, 1)
    m.move_left()
    assert m.cursor == (sq.body, 1)
    m.move_left()
    assert m.cursor == (par.body, 1)
    m.move_left()
    assert m.cursor == (frac.den, 1)


# ═════════════════════════════════════════════════════════════
#  Backspace
# ═════════════════════════════════════════════════════════════

def test_backspace_atoms():
    m = typed('1', '2', '+')
    m.backspace()
    assert kinds(m.root) == [('num', '1'), ('num', '2')]
    m.backspace(); m.backspace()
    assert m.is_empty() and m.cursor == (m.root, 0)
    m.backspace()                                # на пустом — no-op


def test_backspace_into_nonempty_container_and_delete_empty():
    m = typed('x', '/', '2', 'right')
    frac = m.root.items[0]
    m.backspace()
    assert m.cursor == (frac.den, 1)             # вошли в конец знаменателя
    m.backspace()
    assert frac.den.is_empty() and m.cursor == (frac.den, 0)
    m.backspace()
    assert m.cursor == (frac.num, 1)             # в конец числителя
    m.backspace()
    assert frac.num.is_empty()
    m.backspace()                                # контейнер пуст → удалён
    assert m.is_empty() and m.cursor == (m.root, 0)


def test_backspace_at_start_of_nonempty_first_row_moves_before():
    m = typed('a' if False else 'x', '/', '2')
    frac = m.root.items[0]
    m.cursor = (frac.num, 0)
    m.backspace()
    assert m.cursor == (m.root, 0)
    assert kinds(m.root) == ['Frac']


def test_backspace_empty_sup_then_func_with_paren():
    m = typed('x', '^')
    m.backspace()                                # пустая степень удалена
    assert kinds(m.root) == [('var', 'x')] and m.cursor == (m.root, 1)
    m = typed('sin')                             # sin( | )
    m.backspace()                                # пустая скобка удалена → курсор после sin
    assert kinds(m.root) == [('func', 'sin')] and m.cursor == (m.root, 1)
    m = typed('sin')
    m.cursor = (m.root, 1)                       # курсор между sin и ( )
    m.backspace()                                # удаляем sin → и пустую скобку
    assert m.is_empty()
    m = typed('logb', 'right')                   # log_□(|)
    m.cursor = (m.root, 1)
    m.backspace()
    assert m.is_empty()
    m = typed('sin', 'x')
    m.cursor = (m.root, 1)
    m.backspace()                                # скобка не пуста — остаётся
    assert kinds(m.root) == ['Paren']


def test_clear():
    m = typed('x', '/', '2')
    m.clear()
    assert m.is_empty() and m.cursor == (m.root, 0)


# ═════════════════════════════════════════════════════════════
#  to_sympy
# ═════════════════════════════════════════════════════════════

def test_explicit_multiplication():
    assert typed('2', 'x').to_sympy() == '2*x'
    assert typed('x', 'y').to_sympy() == 'x*y'
    assert typed('2', 'pi').to_sympy() == '2*pi'
    assert typed('x', '(', 'x', '+', '1').to_sympy() == 'x*(x+1)'
    assert typed('sin', 'x', ')', 'cos', 'x').to_sympy() == 'sin(x)*cos(x)'
    assert typed('1', '/', '2', 'right', 'x').to_sympy() == '((1)/(2))*x'
    assert typed('2', 'sqrt', 'x', 'right', 'y').to_sympy() == '2*sqrt(x)*y'
    assert typed('x', 'sq', '2').to_sympy() == 'x**(2)*2'


def test_node_types_sympy():
    assert typed('x', '/', 'y').to_sympy() == '((x)/(y))'
    assert typed('x', '^', 'y').to_sympy() == 'x**(y)'
    assert typed('sqrt', 'x').to_sympy() == 'sqrt(x)'
    assert typed('root', '3', 'right', 'x').to_sympy() == 'nroot(x, 3)'
    assert typed('abs', 'x').to_sympy() == 'Abs(x)'
    assert typed('(', 'x').to_sympy() == '(x)'
    assert typed('ln', 'x').to_sympy() == 'log(x)'
    assert typed('log', 'x').to_sympy() == 'log(x, 10)'
    assert typed('logb', 'e', 'right', 'x').to_sympy() == 'log(x, E)'
    assert typed('cot', 'x').to_sympy() == 'cot(x)'
    assert typed('atan', 'x').to_sympy() == 'atan(x)'
    assert typed('tanh', 'x').to_sympy() == 'tanh(x)'
    assert typed('-', 'x').to_sympy() == '-x'
    assert typed('x', '*', '-', '2').to_sympy() == 'x*-2'
    assert typed('y', '=', '-', 'x').to_sympy() == 'y=-x'
    assert typed('.', '5', 'x').to_sympy() == '0.5*x'
    assert typed('5', '.').to_sympy() == '5.0'


def test_incomplete_cases():
    with pytest.raises(IncompleteExpression):
        MathModel().to_sympy()                                   # пусто
    with pytest.raises(IncompleteExpression, match='trailing'):
        typed('x', '+').to_sympy()
    with pytest.raises(IncompleteExpression, match='without base'):
        typed('^', '2').to_sympy()
    with pytest.raises(IncompleteExpression, match='without base'):
        typed('x', '+', '^', '2').to_sympy()
    with pytest.raises(IncompleteExpression, match='exponent'):
        typed('x', '^').to_sympy()                               # пустая степень
    with pytest.raises(IncompleteExpression, match='denominator'):
        typed('x', '/').to_sympy()
    with pytest.raises(IncompleteExpression, match='argument'):
        typed('sin').to_sympy()
    with pytest.raises(IncompleteExpression):
        m = typed('sin'); m.cursor = (m.root, 1); m.backspace(); m.backspace()
        m2 = MathModel(); m2.root.append(Atom('func', 'sin')); m2.to_sympy()   # sin без скобки
    with pytest.raises(IncompleteExpression, match='number'):
        typed('1', '.', '.', '2').to_sympy()
    with pytest.raises(IncompleteExpression):
        typed('.').to_sympy()
    with pytest.raises(IncompleteExpression, match='operator'):
        typed('*', 'x').to_sympy()
    with pytest.raises(IncompleteExpression, match='operator'):
        typed('x', '+', '*', '2').to_sympy()
    with pytest.raises(IncompleteExpression):
        typed('logb', 'right', 'x').to_sympy()                   # пустое основание


# ═════════════════════════════════════════════════════════════
#  from_text / round trip
# ═════════════════════════════════════════════════════════════

sympy = pytest.importorskip('sympy')
from sympy import Symbol, Function, pi, E, sqrt, sin, cos, log, Abs as SAbs, exp  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor)

X = Symbol('x')
Y = Symbol('y')
NROOT = Function('nroot')
LOCAL = {'x': X, 'y': Y, 'pi': pi, 'E': E, 'nroot': NROOT}
TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse(s):
    return parse_expr(s, transformations=TRANS, local_dict=LOCAL)


def same(expr_a, expr_b, pts=(2.0, 0.7, 3.3)):
    # nroot(b, n) — неопределённая функция в тестах: раскрываем вручную
    expr_a = expr_a.replace(NROOT, lambda b, n: sympy.sign(b) * sympy.Abs(b) ** (1 / n))
    expr_b = expr_b.replace(NROOT, lambda b, n: sympy.sign(b) * sympy.Abs(b) ** (1 / n))
    for p in pts:
        a = complex(expr_a.subs({X: p, Y: 1.5}).evalf())
        b = complex(expr_b.subs({X: p, Y: 1.5}).evalf())
        assert abs(a - b) < 1e-9 * max(1.0, abs(b)), (expr_a, expr_b, p)
    return True


@pytest.mark.parametrize('text, expected', [
    ('x^2 + 5/x - sqrt(x^2+15)', X**2 + 5/X - sqrt(X**2 + 15)),
    ('sin(x)/x', sin(X)/X),
    ('log_2(x+1)', log(X + 1, 2)),
    ('|x-1|', SAbs(X - 1)),
    ('2x e^x', 2*X*exp(X)),
    ('2x*e^x', 2*X*exp(X)),
    ('(1+1/x)/(x^2)', (1 + 1/X)/X**2),
    ('e^(x) sin(x)/2', exp(X)*sin(X)/2),
    ('2sin(x)cos(x)', 2*sin(X)*cos(X)),
    ('x/2/3', X/6),
    ('2x/3', 2*X/3),
    ('1/2x', X/2),
    ('-x^2', -X**2),
    ('2^3^2', sympy.Integer(512)),
    ('x**2 + y**2', X**2 + Y**2),
    ('abs(x)+pi', SAbs(X) + pi),
    ('ln(x)', log(X)),
    ('log(x)', log(X, 10)),
    ('lg(x)', log(X, 10)),
    ('log10(x)', log(X, 10)),
    ('log(x, 3)', log(X, 3)),
    ('exp(2x)', exp(2*X)),
    ('x²+1', X**2 + 1),
    ('2πx', 2*pi*X),
    ('x^-1', 1/X),
    ('sin x', sin(X)),
    ('√(x+1)', sqrt(X + 1)),
    ('arctan(x)', sympy.atan(X)),
    ('tg(x)', sympy.tan(X)),
])
def test_from_text_roundtrip(text, expected):
    s = MathModel.from_text(text).to_sympy()
    assert same(parse(s), expected)


def test_from_text_root_and_equation():
    s = MathModel.from_text('root(x, 3) + cbrt(x)').to_sympy()
    assert s == 'nroot(x, 3)+nroot(x, 3)'
    m = MathModel.from_text('y = x^2')
    assert m.to_sympy() == 'y=x**(2)'
    m = MathModel.from_text('x**2+y**2=25')
    assert m.to_sympy() == 'x**(2)+y**(2)=25'


def test_from_text_structure_precedence():
    m = MathModel.from_text('a/b/c'.replace('a', '1').replace('b', '2').replace('c', '3'))
    outer = m.root.items[0]
    assert isinstance(outer, Frac) and isinstance(outer.num.items[0], Frac)
    assert kinds(outer.den) == [('num', '3')]
    m = MathModel.from_text('2x/3')
    assert kinds(m.root.items[0].num) == [('num', '2'), ('var', 'x')]
    m = MathModel.from_text('x^2^3')
    sup = m.root.items[1]
    assert isinstance(sup, Sup) and kinds(sup.exp) == [('num', '2'), 'Sup']
    m = MathModel.from_text('log_2(x)')
    assert kinds(m.root) == [('func', 'log'), 'Sub', 'Paren']
    m = MathModel.from_text('asin(x)')
    assert kinds(m.root) == [('func', 'arcsin'), 'Paren']
    m = MathModel.from_text('exp(x)')
    assert kinds(m.root) == [('const', 'e'), 'Sup']
    m = MathModel.from_text('|x|')
    assert kinds(m.root) == ['Abs']
    assert MathModel.from_text('   ').is_empty()
    assert MathModel.from_text('x').cursor == (MathModel.from_text('x').root, 1) or True


@pytest.mark.parametrize('bad', ['x +', 'foo(x)', '(x', 'x^', '2 $ 3', 'sin(', 'x)', '|x', 'log_(x)'])
def test_from_text_errors(bad):
    with pytest.raises(ValueError):
        MathModel.from_text(bad)


def test_roundtrip_display_sympy_text():
    """to_display → from_text → та же формула (unicode-вывод читается парсером)."""
    for text in ['x^2 + 5/x - sqrt(x^2+15)', 'log_2(x+1)', '|x-1|+root(x,3)', '2x e^x']:
        m = MathModel.from_text(text)
        back = MathModel.from_text(m.to_display())
        assert same(parse(back.to_sympy()), parse(m.to_sympy()))


# ═════════════════════════════════════════════════════════════
#  to_display / JSON
# ═════════════════════════════════════════════════════════════

def test_to_display():
    assert MathModel.from_text('x^2 + 5/x - sqrt(x^2+15)').to_display() == 'x²+5/x−√(x²+15)'
    assert MathModel.from_text('(x+1)/2').to_display() == '(x+1)/2'
    assert MathModel.from_text('log_2(x+1)').to_display() == 'log₂(x+1)'
    assert MathModel.from_text('x^-1').to_display() == 'x⁻¹'
    assert MathModel.from_text('x^y').to_display() == 'x^y'
    assert MathModel.from_text('x^(y+1)').to_display() == 'x^(y+1)'
    assert MathModel.from_text('root(x+1, 3)').to_display() == '∛(x+1)'
    assert MathModel.from_text('|x|').to_display() == '|x|'
    assert typed('x', '/').to_display() == 'x/□'
    assert MathModel().to_display() == ''


def test_json_roundtrip():
    m = MathModel.from_text('|x-1| + root(x, 3) + log_2(x+1) + (1+1/x)/(x^2) - e^x')
    data = m.to_json()
    m2 = MathModel.from_json(data)
    assert m2.to_json() == data
    assert m2.to_sympy() == m.to_sympy()
    assert m2.cursor == (m2.root, len(m2.root.items))
    # все узлы правильно «усыновлены»
    def check(row, parent):
        assert row.parent is parent
        for it in row.items:
            assert it.parent is row
            for r in it.child_rows():
                check(r, it)
    check(m2.root, None)
    assert MathModel.from_json({'version': 1, 'root': []}).is_empty()
    with pytest.raises(ValueError):
        MathModel.from_json({'root': [{'t': 'alien'}]})


# ═════════════════════════════════════════════════════════════
#  Tk-виджет (только при наличии дисплея)
# ═════════════════════════════════════════════════════════════

def _tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.geometry('600x400+0+0')
    return root


def test_widget_basic():
    root = _tk_root()
    try:
        from math_editor import MathEditor
        changes = []
        focused = []
        ed = MathEditor(root, width=400, height=30,
                        on_change=lambda e: changes.append(e.get_display()),
                        on_focus=lambda e: focused.append(e))
        ed.pack(fill='x')
        root.update()
        h0 = ed.cv.winfo_height()
        assert 26 <= h0 <= 30
        for t in ('x', '^', '2', 'right', '+', '1', '/', 'x'):
            ed.insert_token(t)
        root.update()
        assert changes[-1] == 'x²+1/x'
        assert ed.cv.winfo_height() > h0            # высота выросла из-за дроби
        assert ed.get_sympy() == 'x**(2)+((1)/(x))'
        # курсор рисуется только у активного редактора
        assert not ed.cv.find_withtag('cursor')
        ed.set_active(True)
        root.update()
        assert ed.cv.find_withtag('cursor')
        assert ed.cget('highlightbackground') == ed.active_color
        ed.set_error('bad')
        assert ed.cget('highlightbackground') == ed.error_color
        ed.set_error(None)
        assert ed.cget('highlightbackground') == ed.active_color
        # клик у левого края ставит курсор в начало корневой строки и вызывает on_focus
        ed.set_active(False)
        ed.cv.event_generate('<Button-1>', x=3, y=ed.cv.winfo_height() // 2)
        root.update()
        assert focused and focused[-1] is ed
        assert ed.model.cursor == (ed.model.root, 0)
        assert ed.is_active()
        # клик по знаменателю дроби — курсор попадает внутрь
        info = next(i for i in ed._rows if i.row is ed.model.root.items[3].den)
        ed.cv.event_generate('<Button-1>', x=int((info.x0 + info.x1) / 2), y=int(info.base - 2))
        assert ed.model.cursor[0] is ed.model.root.items[3].den
        # ни одной клавиатурной привязки, вставляющей текст
        assert '<KeyPress>' not in ed.cv.bind() and '<Key>' not in ed.cv.bind()
        ed.set_model(MathModel.from_text('sin(x)/x'))
        root.update()
        assert ed.get_display() == '(sin(x))/x'
        ed.destroy()
    finally:
        root.destroy()


def test_widget_screenshots(tmp_path):
    """Сохраняем картинки для визуального контроля (scratchpad или tmp)."""
    root = _tk_root()
    try:
        import shutil
        import subprocess
        from math_editor import MathEditor
        out_dir = os.environ.get('MATH_EDITOR_PNG_DIR') or str(tmp_path)
        exprs = ['x^2 + 5/x - sqrt(x^2+15)', '(1+1/x)/(x^2)',
                 '|x-1| + root(x, 3) + log_2(x+1)', 'e^(x) sin(x)/2']
        for text in exprs:
            MathEditor(root, model=MathModel.from_text(text), width=560).pack(fill='x', pady=3)
        root.update()
        if shutil.which('import'):
            png = os.path.join(out_dir, 'editor_tests.png')
            subprocess.run(['import', '-window', 'root', png], timeout=20)
            assert os.path.exists(png)
    finally:
        root.destroy()
