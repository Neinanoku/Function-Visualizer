# -*- coding: utf-8 -*-
"""
math_editor.py — 2-D редактор формул в стиле Desmos.

Два слоя:
  • МОДЕЛЬ (MathModel) — чистый Python без Tk: дерево Row/Node, курсор,
    вставка токенов экранной клавиатуры, навигация, backspace, экспорт в
    строку для sympy (to_sympy), линейное unicode-представление (to_display),
    сериализация (to_json/from_json) и разбор обычного текста (from_text).
  • ВИД (MathEditor) — tk.Frame с Canvas, который рисует формулу «как в
    учебнике»: настоящие дроби, приподнятые степени, √ с чертой, |…|,
    автоматически растущие скобки, мигающий курсор, клик мышью ставит курсор.
    Ввод с клавиатуры ВЫКЛЮЧЕН (KEYBOARD_INPUT = False) — весь ввод идёт
    через insert_token() с экранной клавиатуры приложения.

Зависимости: только tkinter + tkinter.font.
"""

import re
import tkinter as tk
import tkinter.font as tkfont

# Если True — для отладки виджет принимает стрелки/цифры/backspace с
# физической клавиатуры. В продукте должно быть False.
KEYBOARD_INPUT = False

# ═════════════════════════════════════════════════════════════
#  СЛОВАРИ ТОКЕНОВ
# ═════════════════════════════════════════════════════════════

# токен клавиатуры → текст op-атома
OP_TOKENS = {'+': '+', '-': '−', '*': '×', '=': '='}
# текст op-атома → строка для sympy
OP_SYMPY = {'+': '+', '−': '-', '×': '*', '=': '='}

# токен клавиатуры → отображаемое имя функции (хранится в Atom.text)
FUNC_TOKENS = {
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'sec': 'sec', 'csc': 'csc',
    'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
    'ln': 'ln', 'log': 'log',
}
# отображаемое имя → имя функции sympy
FUNC_SYMPY = {
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'sec': 'sec', 'csc': 'csc',
    'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
    'ln': 'log', 'log': 'log',
}
FUNC_NAMES = frozenset(FUNC_SYMPY)

NAV_TOKENS = ('left', 'right', 'up', 'down', 'backspace', 'clear')

# Полный список токенов, которые принимает insert_token (для сборки клавиатуры)
ALL_TOKENS = (tuple('0123456789.') + ('x', 'y') + tuple(OP_TOKENS) +
              ('/', '^', 'sq', 'sqrt', 'root', 'abs', '(', ')', 'pi', 'e', 'exp', 'logb') +
              tuple(FUNC_TOKENS) + NAV_TOKENS)

_SUP_MAP = str.maketrans('0123456789-−+.', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺·')
_SUB_MAP = str.maketrans('0123456789-+', '₀₁₂₃₄₅₆₇₈₉₋₊')


class IncompleteExpression(ValueError):
    """Выражение не закончено / содержит пустые места. Текст говорит, чего не хватает."""
    pass


# ═════════════════════════════════════════════════════════════
#  МОДЕЛЬ: ДЕРЕВО
# ═════════════════════════════════════════════════════════════

class Row:
    """Горизонтальная последовательность узлов."""
    __slots__ = ('items', 'parent')

    def __init__(self, items=None, parent=None):
        self.items = []
        self.parent = parent          # Node-владелец (None у корневой строки)
        for it in (items or []):
            self.append(it)

    def append(self, node):
        node.parent = self
        self.items.append(node)

    def insert(self, idx, node):
        node.parent = self
        self.items.insert(idx, node)

    def extend(self, nodes):
        for n in nodes:
            self.append(n)

    def remove_at(self, idx):
        node = self.items.pop(idx)
        node.parent = None
        return node

    def index(self, node):
        for i, it in enumerate(self.items):
            if it is node:
                return i
        raise ValueError("node is not in row")

    def is_empty(self):
        return not self.items

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return "Row(%r)" % (self.items,)


class Node:
    """Базовый узел; parent — Row, в которой он лежит."""
    __slots__ = ('parent',)

    def __init__(self):
        self.parent = None

    def child_rows(self):
        return []

    def is_empty(self):
        """Контейнер пуст, если пусты все его строки (атом — никогда)."""
        rows = self.child_rows()
        return bool(rows) and all(r.is_empty() for r in rows)


class Atom(Node):
    """kind ∈ {'num','var','const','op','func'}; text — один символ/имя."""
    __slots__ = ('kind', 'text')

    def __init__(self, kind, text):
        super().__init__()
        self.kind = kind
        self.text = text

    def __repr__(self):
        return "Atom(%s %r)" % (self.kind, self.text)


class Frac(Node):
    __slots__ = ('num', 'den')

    def __init__(self, num=None, den=None):
        super().__init__()
        self.num = num if num is not None else Row()
        self.den = den if den is not None else Row()
        self.num.parent = self
        self.den.parent = self

    def child_rows(self):
        return [self.num, self.den]

    def __repr__(self):
        return "Frac(%r / %r)" % (self.num.items, self.den.items)


class Sup(Node):
    """Степень; относится к операнду, стоящему перед ней."""
    __slots__ = ('exp',)

    def __init__(self, exp=None):
        super().__init__()
        self.exp = exp if exp is not None else Row()
        self.exp.parent = self

    def child_rows(self):
        return [self.exp]

    def __repr__(self):
        return "Sup(%r)" % (self.exp.items,)


class Sub(Node):
    """Нижний индекс — основание логарифма: log + Sub(base) + Paren(arg)."""
    __slots__ = ('sub',)

    def __init__(self, sub=None):
        super().__init__()
        self.sub = sub if sub is not None else Row()
        self.sub.parent = self

    def child_rows(self):
        return [self.sub]

    def __repr__(self):
        return "Sub(%r)" % (self.sub.items,)


class Sqrt(Node):
    __slots__ = ('body',)

    def __init__(self, body=None):
        super().__init__()
        self.body = body if body is not None else Row()
        self.body.parent = self

    def child_rows(self):
        return [self.body]

    def __repr__(self):
        return "Sqrt(%r)" % (self.body.items,)


class Root(Node):
    """Корень n-й степени: index — показатель, body — подкоренное."""
    __slots__ = ('index', 'body')

    def __init__(self, index=None, body=None):
        super().__init__()
        self.index = index if index is not None else Row()
        self.body = body if body is not None else Row()
        self.index.parent = self
        self.body.parent = self

    def child_rows(self):
        return [self.index, self.body]

    def __repr__(self):
        return "Root(%r, %r)" % (self.body.items, self.index.items)


class Abs(Node):
    __slots__ = ('body',)

    def __init__(self, body=None):
        super().__init__()
        self.body = body if body is not None else Row()
        self.body.parent = self

    def child_rows(self):
        return [self.body]

    def __repr__(self):
        return "Abs(%r)" % (self.body.items,)


class Paren(Node):
    """Автоматически растущие круглые скобки."""
    __slots__ = ('body',)

    def __init__(self, body=None):
        super().__init__()
        self.body = body if body is not None else Row()
        self.body.parent = self

    def child_rows(self):
        return [self.body]

    def __repr__(self):
        return "Paren(%r)" % (self.body.items,)


def _is_op(node, text=None):
    return (isinstance(node, Atom) and node.kind == 'op'
            and (text is None or node.text == text))


# ═════════════════════════════════════════════════════════════
#  МОДЕЛЬ: КУРСОР, ВСТАВКА, НАВИГАЦИЯ
# ═════════════════════════════════════════════════════════════

class MathModel:
    """
    Дерево формулы + курсор. cursor = (row, index): курсор стоит ПЕРЕД
    row.items[index]; index может равняться len(row.items) (конец строки).
    """

    def __init__(self, root=None):
        self.root = root if root is not None else Row()
        self.root.parent = None
        self.cursor = (self.root, len(self.root.items))

    # ── служебное ───────────────────────────────────────────
    def is_empty(self):
        return self.root.is_empty()

    def clear(self):
        self.root = Row()
        self.cursor = (self.root, 0)

    def cursor_to_end(self):
        self.cursor = (self.root, len(self.root.items))

    def _insert_node(self, node):
        """Вставляет узел в позицию курсора, курсор — после него."""
        row, i = self.cursor
        row.insert(i, node)
        self.cursor = (row, i + 1)
        return node

    # ── вставка токенов ─────────────────────────────────────
    def insert_token(self, token):
        t = str(token)
        if t in NAV_TOKENS:
            getattr(self, {'left': 'move_left', 'right': 'move_right',
                           'up': 'move_up', 'down': 'move_down',
                           'backspace': 'backspace', 'clear': 'clear'}[t])()
            return
        if len(t) == 1 and t in '0123456789.':
            self._insert_node(Atom('num', t))
        elif t in ('x', 'y'):
            self._insert_node(Atom('var', t))
        elif t in OP_TOKENS:
            self._insert_node(Atom('op', OP_TOKENS[t]))
        elif t == 'pi':
            self._insert_node(Atom('const', 'π'))
        elif t == 'e':
            self._insert_node(Atom('const', 'e'))
        elif t == 'exp':
            self._insert_node(Atom('const', 'e'))
            sup = self._insert_node(Sup())
            self.cursor = (sup.exp, 0)
        elif t == '/':
            self._insert_frac()
        elif t == '^':
            sup = self._insert_node(Sup())
            self.cursor = (sup.exp, 0)
        elif t == 'sq':
            sup = Sup()
            sup.exp.append(Atom('num', '2'))
            self._insert_node(sup)
        elif t == 'sqrt':
            node = self._insert_node(Sqrt())
            self.cursor = (node.body, 0)
        elif t == 'root':
            node = self._insert_node(Root())
            self.cursor = (node.index, 0)
        elif t == 'abs':
            node = self._insert_node(Abs())
            self.cursor = (node.body, 0)
        elif t == '(':
            node = self._insert_node(Paren())
            self.cursor = (node.body, 0)
        elif t == ')':
            self._close_paren()
        elif t in FUNC_TOKENS:
            self._insert_node(Atom('func', FUNC_TOKENS[t]))
            par = self._insert_node(Paren())
            self.cursor = (par.body, 0)
        elif t == 'logb':
            self._insert_node(Atom('func', 'log'))
            sub = self._insert_node(Sub())
            self._insert_node(Paren())
            self.cursor = (sub.sub, 0)
        else:
            raise ValueError("unknown token %r" % (token,))

    def _operand_start(self, row, i):
        """
        Начало «операнда» перед позицией i — как в Desmos/MathQuill: всё
        от ближайшего бинарного оператора (+ − × =) до курсора.
        Возвращает индекс начала (== i, если операнда нет).
        """
        j = i
        while j > 0 and not _is_op(row.items[j - 1]):
            j -= 1
        return j

    def _insert_frac(self):
        row, i = self.cursor
        j = self._operand_start(row, i)
        frac = Frac()
        if j < i:
            moved = row.items[j:i]
            del row.items[j:i]
            frac.num.extend(moved)
            row.insert(j, frac)
            self.cursor = (frac.den, 0)
        else:
            row.insert(i, frac)
            self.cursor = (frac.num, 0)

    def _close_paren(self):
        """Курсор — сразу после ближайшей охватывающей скобки Paren."""
        row = self.cursor[0]
        while row.parent is not None:
            node = row.parent
            prow = node.parent
            if isinstance(node, Paren):
                self.cursor = (prow, prow.index(node) + 1)
                return
            row = prow

    # ── навигация ───────────────────────────────────────────
    def _exit_after(self, node):
        prow = node.parent
        self.cursor = (prow, prow.index(node) + 1)

    def _exit_before(self, node):
        prow = node.parent
        self.cursor = (prow, prow.index(node))

    def move_right(self):
        row, i = self.cursor
        if i < len(row.items):
            item = row.items[i]
            rows = item.child_rows()
            if rows:
                self.cursor = (rows[0], 0)
            else:
                self.cursor = (row, i + 1)
            return
        node = row.parent
        if node is None:
            return
        rows = node.child_rows()
        k = next(n for n, r in enumerate(rows) if r is row)
        if k + 1 < len(rows):
            self.cursor = (rows[k + 1], 0)
        else:
            self._exit_after(node)
            # из основания логарифма (Sub) — сразу внутрь его скобки
            prow, j = self.cursor
            if isinstance(node, Sub) and j < len(prow.items) and isinstance(prow.items[j], Paren):
                self.cursor = (prow.items[j].body, 0)

    def move_left(self):
        row, i = self.cursor
        if i > 0:
            item = row.items[i - 1]
            rows = item.child_rows()
            if rows:
                last = rows[-1]
                self.cursor = (last, len(last.items))
            else:
                self.cursor = (row, i - 1)
            return
        node = row.parent
        if node is None:
            return
        rows = node.child_rows()
        k = next(n for n, r in enumerate(rows) if r is row)
        if k > 0:
            prev = rows[k - 1]
            self.cursor = (prev, len(prev.items))
        else:
            self._exit_before(node)

    def move_down(self):
        row, i = self.cursor
        node = row.parent
        if node is None:
            return
        if isinstance(node, Frac) and row is node.num:
            self.cursor = (node.den, min(i, len(node.den.items)))
        elif isinstance(node, Root) and row is node.index:
            # из показателя корня — в подкоренное выражение
            self.cursor = (node.body, 0)
        elif isinstance(node, Sub):
            # из основания логарифма — в его аргумент (скобки справа), если есть
            prow = node.parent
            k = prow.index(node) if prow is not None else -1
            if prow is not None and k + 1 < len(prow.items) and isinstance(prow.items[k + 1], Paren):
                self.cursor = (prow.items[k + 1].body, 0)
            else:
                self._exit_after(node)
        elif isinstance(node, Sup):
            self._exit_after(node)

    def move_up(self):
        row, i = self.cursor
        node = row.parent
        if node is None:
            return
        if isinstance(node, Frac) and row is node.den:
            self.cursor = (node.num, min(i, len(node.num.items)))
        elif isinstance(node, Root) and row is node.body:
            self.cursor = (node.index, len(node.index.items))

    # ── backspace ───────────────────────────────────────────
    def backspace(self):
        row, i = self.cursor
        if i > 0:
            prev = row.items[i - 1]
            if isinstance(prev, Atom):
                row.remove_at(i - 1)
                self.cursor = (row, i - 1)
                # Удалили имя функции — убираем и её пустые Sub/Paren справа
                if prev.kind == 'func':
                    k = i - 1
                    while k < len(row.items) and isinstance(row.items[k], (Sub, Paren)) \
                            and row.items[k].is_empty():
                        row.remove_at(k)
                return
            if prev.is_empty():
                row.remove_at(i - 1)
                self.cursor = (row, i - 1)
            else:
                last = prev.child_rows()[-1]
                self.cursor = (last, len(last.items))
            return
        node = row.parent
        if node is None:
            return
        prow = node.parent
        if node.is_empty():
            k = prow.index(node)
            prow.remove_at(k)
            self.cursor = (prow, k)
            return
        rows = node.child_rows()
        k = next(n for n, r in enumerate(rows) if r is row)
        if k > 0:
            prev = rows[k - 1]
            self.cursor = (prev, len(prev.items))
        else:
            self._exit_before(node)

    # ═════════════════════════════════════════════════════════
    #  ЭКСПОРТ ДЛЯ SYMPY
    # ═════════════════════════════════════════════════════════
    def to_sympy(self):
        """Строка для parse_expr (явные '*', '**', sqrt(), nroot(), Abs(), log(a, b))."""
        return self._row_sympy(self.root, 'expression')

    @staticmethod
    def _number_text(chars, what):
        s = ''.join(chars)
        if s.count('.') > 1:
            raise IncompleteExpression("bad number '%s'" % s)
        if s == '.':
            raise IncompleteExpression("lone '.' in %s" % what)
        if s.startswith('.'):
            s = '0' + s
        if s.endswith('.'):
            s = s + '0'
        return s

    def _row_sympy(self, row, what):
        items = row.items
        if not items:
            raise IncompleteExpression("empty %s" % what)
        pieces = []            # [(kind, text)], kind ∈ {'operand', 'op'}
        i = 0
        n = len(items)
        while i < n:
            it = items[i]
            if isinstance(it, Atom):
                if it.kind == 'num':
                    j = i
                    while j < n and isinstance(items[j], Atom) and items[j].kind == 'num':
                        j += 1
                    pieces.append(('operand', self._number_text(
                        [a.text for a in items[i:j]], what)))
                    i = j
                    continue
                if it.kind == 'var':
                    pieces.append(('operand', it.text))
                elif it.kind == 'const':
                    pieces.append(('operand', 'pi' if it.text == 'π' else 'E'))
                elif it.kind == 'op':
                    pieces.append(('op', OP_SYMPY[it.text]))
                elif it.kind == 'func':
                    text, i = self._func_sympy(items, i)
                    pieces.append(('operand', text))
                    continue
                else:
                    raise IncompleteExpression("unknown atom %r" % (it,))
            elif isinstance(it, Sup):
                if not pieces or pieces[-1][0] != 'operand':
                    raise IncompleteExpression("exponent without base")
                base = pieces[-1][1]
                pieces[-1] = ('operand', base + '**(' + self._row_sympy(it.exp, 'exponent') + ')')
            elif isinstance(it, Sub):
                raise IncompleteExpression("subscript is only allowed as a log base")
            elif isinstance(it, Frac):
                pieces.append(('operand', '((%s)/(%s))' % (
                    self._row_sympy(it.num, 'numerator'),
                    self._row_sympy(it.den, 'denominator'))))
            elif isinstance(it, Sqrt):
                pieces.append(('operand', 'sqrt(%s)' % self._row_sympy(it.body, 'root')))
            elif isinstance(it, Root):
                pieces.append(('operand', 'nroot(%s, %s)' % (
                    self._row_sympy(it.body, 'root'),
                    self._row_sympy(it.index, 'root index'))))
            elif isinstance(it, Abs):
                pieces.append(('operand', 'Abs(%s)' % self._row_sympy(it.body, 'absolute value')))
            elif isinstance(it, Paren):
                pieces.append(('operand', '(%s)' % self._row_sympy(it.body, 'parentheses')))
            else:
                raise IncompleteExpression("unknown node %r" % (it,))
            i += 1

        out = []
        prev = None
        for kind, text in pieces:
            if kind == 'operand':
                if prev == 'operand':
                    out.append('*')
                out.append(text)
            else:
                if prev in (None, 'op'):
                    if text not in ('+', '-'):
                        raise IncompleteExpression("operator '%s' without left operand" % text)
                out.append(text)
            prev = kind
        if prev == 'op':
            raise IncompleteExpression("trailing operator in %s" % what)
        return ''.join(out)

    def _func_sympy(self, items, i):
        """Функция + [Sub] + Paren → строка; возвращает (text, next_index)."""
        func = items[i]
        name = func.text
        j = i + 1
        base = None
        if j < len(items) and isinstance(items[j], Sub):
            if name != 'log':
                raise IncompleteExpression("subscript after %s" % name)
            base = self._row_sympy(items[j].sub, 'log base')
            j += 1
        if j >= len(items) or not isinstance(items[j], Paren):
            raise IncompleteExpression("%s without argument" % name)
        arg = self._row_sympy(items[j].body, 'argument of %s' % name)
        if name == 'log':
            text = 'log(%s, %s)' % (arg, base if base is not None else '10')
        else:
            text = '%s(%s)' % (FUNC_SYMPY[name], arg)
        return text, j + 1

    # ═════════════════════════════════════════════════════════
    #  ЛИНЕЙНОЕ UNICODE-ПРЕДСТАВЛЕНИЕ
    # ═════════════════════════════════════════════════════════
    def to_display(self):
        return self._row_display(self.root)

    @classmethod
    def _row_display(cls, row):
        out = []
        items = row.items
        for i, it in enumerate(items):
            if isinstance(it, Atom):
                if it.kind == 'func':
                    out.append(it.text)
                else:
                    out.append(it.text)
            elif isinstance(it, Sup):
                s = cls._row_display(it.exp)
                if s and re.fullmatch(r'[−-]?[0-9]+', s):
                    out.append(s.translate(_SUP_MAP))
                elif cls._single_unit(it.exp):
                    out.append('^' + s)
                else:
                    out.append('^(' + s + ')')
            elif isinstance(it, Sub):
                s = cls._row_display(it.sub)
                if re.fullmatch(r'[0-9]+', s):
                    out.append(s.translate(_SUB_MAP))
                else:
                    out.append('_(' + s + ')')
            elif isinstance(it, Frac):
                out.append(cls._wrap(it.num) + '/' + cls._wrap(it.den))
            elif isinstance(it, Sqrt):
                out.append('√' + cls._wrap(it.body, atom_only=True))
            elif isinstance(it, Root):
                idx = cls._row_display(it.index)
                body = cls._wrap(it.body, atom_only=True)
                if idx == '3':
                    out.append('∛' + body)
                elif idx == '4':
                    out.append('∜' + body)
                elif re.fullmatch(r'[0-9]+', idx):
                    out.append(idx.translate(_SUP_MAP) + '√' + body)
                else:
                    out.append('root(%s, %s)' % (cls._row_display(it.body), idx))
            elif isinstance(it, Abs):
                out.append('|' + cls._row_display(it.body) + '|')
            elif isinstance(it, Paren):
                out.append('(' + cls._row_display(it.body) + ')')
        return ''.join(out)

    @staticmethod
    def _single_unit(row):
        """Строка — один операнд (число, переменная, константа или один контейнер)."""
        items = row.items
        if not items:
            return False
        if all(isinstance(a, Atom) and a.kind == 'num' for a in items):
            return True
        if len(items) == 1:
            return not (isinstance(items[0], Atom) and items[0].kind in ('op', 'func'))
        return False

    @classmethod
    def _wrap(cls, row, atom_only=False):
        s = cls._row_display(row)
        if not s:
            return '□'
        items = row.items
        simple = cls._single_unit(row)
        if atom_only:
            simple = simple and all(isinstance(a, Atom) for a in items)
        return s if simple else '(' + s + ')'

    # ═════════════════════════════════════════════════════════
    #  JSON
    # ═════════════════════════════════════════════════════════
    def to_json(self):
        return {'version': 1, 'root': self._row_json(self.root)}

    @classmethod
    def _row_json(cls, row):
        out = []
        for it in row.items:
            if isinstance(it, Atom):
                out.append({'t': 'atom', 'kind': it.kind, 'text': it.text})
            elif isinstance(it, Frac):
                out.append({'t': 'frac', 'num': cls._row_json(it.num), 'den': cls._row_json(it.den)})
            elif isinstance(it, Sup):
                out.append({'t': 'sup', 'exp': cls._row_json(it.exp)})
            elif isinstance(it, Sub):
                out.append({'t': 'sub', 'sub': cls._row_json(it.sub)})
            elif isinstance(it, Sqrt):
                out.append({'t': 'sqrt', 'body': cls._row_json(it.body)})
            elif isinstance(it, Root):
                out.append({'t': 'root', 'index': cls._row_json(it.index), 'body': cls._row_json(it.body)})
            elif isinstance(it, Abs):
                out.append({'t': 'abs', 'body': cls._row_json(it.body)})
            elif isinstance(it, Paren):
                out.append({'t': 'paren', 'body': cls._row_json(it.body)})
        return out

    @classmethod
    def from_json(cls, data):
        if isinstance(data, dict):
            items = data.get('root', [])
        else:
            items = data or []
        return cls(cls._row_from_json(items))

    @classmethod
    def _row_from_json(cls, items):
        row = Row()
        for d in items:
            t = d.get('t')
            if t == 'atom':
                row.append(Atom(d['kind'], d['text']))
            elif t == 'frac':
                row.append(Frac(cls._row_from_json(d.get('num', [])), cls._row_from_json(d.get('den', []))))
            elif t == 'sup':
                row.append(Sup(cls._row_from_json(d.get('exp', []))))
            elif t == 'sub':
                row.append(Sub(cls._row_from_json(d.get('sub', []))))
            elif t == 'sqrt':
                row.append(Sqrt(cls._row_from_json(d.get('body', []))))
            elif t == 'root':
                row.append(Root(cls._row_from_json(d.get('index', [])), cls._row_from_json(d.get('body', []))))
            elif t == 'abs':
                row.append(Abs(cls._row_from_json(d.get('body', []))))
            elif t == 'paren':
                row.append(Paren(cls._row_from_json(d.get('body', []))))
            else:
                raise ValueError("unknown node type %r" % (t,))
        return row

    # ═════════════════════════════════════════════════════════
    #  РАЗБОР ОБЫЧНОГО ТЕКСТА
    # ═════════════════════════════════════════════════════════
    @classmethod
    def from_text(cls, text):
        """
        Разбирает линейный текст: числа, x, y, pi/π, e, + - * / ^ ** =, ( ),
        sqrt( ), abs( ), |…|, функции (sin … tanh, ln, log, log_2(…), log(a, b),
        lg), root(x, n), cbrt, exp(…), неявное умножение (2x, 2sin(x)).
        Пустая строка → пустая модель. ValueError при ошибке.
        """
        items = _Parser(text).parse()
        model = cls(Row(items))
        model.cursor_to_end()
        return model


# ─────────────────────────────────────────────────────────────
#  Парсер текста (рекурсивный спуск, строит дерево Row/Node)
# ─────────────────────────────────────────────────────────────

_DIGITS = '0123456789'      # str.isdigit() истинен и для '²' — поэтому явный список
_SUPER_CHARS = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
                '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+'}
_SUBSC_CHARS = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
                '₆': '6', '₇': '7', '₈': '8', '₉': '9'}
_OP_CHARS = {'+': '+', '-': '-', '−': '-', '–': '-', '*': '*', '×': '*', '·': '*',
             '/': '/', '÷': '/', '^': '^', '=': '=', '(': '(', ')': ')',
             '|': '|', ',': ',', '_': '_', '√': '√', '∛': '∛', '∜': '∜',
             '[': '(', ']': ')', '{': '(', '}': ')'}
_NAME_ALIASES = {
    'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan', 'arctg': 'atan',
    'tg': 'tan', 'ctg': 'cot', 'arcctg': 'acot',
}
_KNOWN_NAMES = (set(FUNC_TOKENS) | set(_NAME_ALIASES) |
                {'x', 'y', 'pi', 'e', 'exp', 'sqrt', 'abs', 'root', 'nroot',
                 'cbrt', 'lg'})
_NAMES_LONGEST = sorted(_KNOWN_NAMES, key=len, reverse=True)


def _tokenize(text):
    toks = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in _DIGITS or (c == '.' and i + 1 < n and text[i + 1] in _DIGITS):
            m = re.match(r'[0-9]*\.?[0-9]*', text[i:])
            toks.append(('num', m.group(0)))
            i += len(m.group(0))
            continue
        if ('a' <= c <= 'z') or ('A' <= c <= 'Z'):
            m = re.match(r'[A-Za-z]+', text[i:])
            word = m.group(0)
            i += len(word)
            # log2( / log10( — основание, прилипшее к имени
            if word.lower() == 'log' and i < n and text[i] in _DIGITS:
                m2 = re.match(r'[0-9]+', text[i:])
                j = i + len(m2.group(0))
                if j < n and text[j] in '(':
                    toks.append(('id', 'log'))
                    toks.append(('op', '_'))
                    toks.append(('num', m2.group(0)))
                    i = j
                    continue
            for name in _split_ident(word):
                toks.append(('id', name))
            continue
        if c == 'π':
            toks.append(('id', 'pi'))
            i += 1
            continue
        if c in _SUPER_CHARS:
            m = re.match(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+', text[i:])
            s = ''.join(_SUPER_CHARS[ch] for ch in m.group(0))
            toks.append(('sup', s))
            i += len(m.group(0))
            continue
        if c in _SUBSC_CHARS:
            m = re.match(r'[₀₁₂₃₄₅₆₇₈₉]+', text[i:])
            s = ''.join(_SUBSC_CHARS[ch] for ch in m.group(0))
            toks.append(('subd', s))
            i += len(m.group(0))
            continue
        if text.startswith('**', i):
            toks.append(('op', '^'))
            i += 2
            continue
        if c in _OP_CHARS:
            toks.append(('op', _OP_CHARS[c]))
            i += 1
            continue
        raise ValueError("unexpected character %r" % c)
    return toks


def _split_ident(word):
    """'2xsinx' → ['x','sin','x']: жадно, самые длинные имена первыми."""
    w = word.lower() if word != 'E' else 'e'
    if w == 'e' or word == 'E':
        return ['e']
    out = []
    i = 0
    while i < len(w):
        for name in _NAMES_LONGEST:
            if w.startswith(name, i):
                out.append(name)
                i += len(name)
                break
        else:
            raise ValueError("unknown identifier %r" % word)
    return out


class _Parser:
    def __init__(self, text):
        self.toks = _tokenize(text or '')
        self.pos = 0
        self.abs_depth = 0

    # ── утилиты ─────────────────────────────────────────────
    def peek(self):
        if self.pos < len(self.toks):
            return self.toks[self.pos]
        return ('end', '')

    def next(self):
        t = self.peek()
        self.pos += 1
        return t

    def is_op(self, v):
        t = self.peek()
        return t[0] == 'op' and t[1] == v

    def expect(self, v):
        if not self.is_op(v):
            raise ValueError("expected '%s' but found %r" % (v, self.peek()[1] or 'end'))
        self.next()

    def starts_primary(self, implicit):
        tt, tv = self.peek()
        if tt in ('num', 'id'):
            return True
        if tt == 'op':
            if tv in ('(', '√', '∛', '∜'):
                return True
            if tv == '|':
                return not (implicit and self.abs_depth > 0)
        return False

    # ── грамматика ──────────────────────────────────────────
    def parse(self):
        if not self.toks:
            return []
        items = self.expr()
        if self.is_op('='):
            self.next()
            items.append(Atom('op', '='))
            items.extend(self.expr())
        if self.peek()[0] != 'end':
            raise ValueError("unexpected %r" % (self.peek()[1],))
        return items

    def expr(self):
        items = self.term()
        while self.is_op('+') or self.is_op('-'):
            op = self.next()[1]
            items.append(Atom('op', '+' if op == '+' else '−'))
            items.extend(self.term())
        return items

    def term(self):
        items = self.unary()
        while True:
            if self.is_op('*'):
                self.next()
                items.append(Atom('op', '×'))
                items.extend(self.unary())
            elif self.is_op('/'):
                self.next()
                den = self.unary()
                items = [Frac(Row(items), Row(den))]
            elif self.starts_primary(implicit=True):
                items.extend(self.power())        # неявное умножение
            else:
                return items

    def unary(self):
        signs = []
        while self.is_op('+') or self.is_op('-'):
            op = self.next()[1]
            signs.append(Atom('op', '+' if op == '+' else '−'))
        return signs + self.power()

    def power(self):
        base = self.primary()
        tt, tv = self.peek()
        if tt == 'op' and tv == '^':
            self.next()
            base.append(Sup(Row(self.exponent())))
        elif tt == 'sup':
            self.next()
            base.append(Sup(Row([Atom('op', '−') if ch == '-' else
                                 Atom('op', '+') if ch == '+' else Atom('num', ch)
                                 for ch in tv])))
        return base

    def exponent(self):
        signs = []
        while self.is_op('+') or self.is_op('-'):
            op = self.next()[1]
            signs.append(Atom('op', '+' if op == '+' else '−'))
        return signs + self.power()          # правоассоциативно: x^2^3 = x^(2^3)

    def call_args(self, nmin, nmax):
        self.expect('(')
        saved = self.abs_depth
        self.abs_depth = 0
        args = [self.expr()]
        while self.is_op(','):
            self.next()
            args.append(self.expr())
        self.expect(')')
        self.abs_depth = saved
        if not (nmin <= len(args) <= nmax):
            raise ValueError("wrong number of arguments")
        return args

    def func_arg(self):
        """Аргумент функции: (expr) или, без скобок, следующий операнд (sin x)."""
        if self.is_op('('):
            return self.call_args(1, 1)[0]
        if self.starts_primary(implicit=False):
            return self.power()
        raise ValueError("function without argument")

    def radicand(self):
        if self.is_op('('):
            return self.call_args(1, 1)[0]
        if self.starts_primary(implicit=False):
            return self.power()
        raise ValueError("√ without argument")

    def log_base(self):
        tt, tv = self.peek()
        if tt == 'num':
            self.next()
            return [Atom('num', ch) for ch in tv]
        if tt == 'id' and tv in ('x', 'y', 'pi', 'e'):
            return self.primary()
        if self.is_op('('):
            return self.call_args(1, 1)[0]
        raise ValueError("bad log base")

    def primary(self):
        tt, tv = self.peek()
        if tt == 'num':
            self.next()
            return [Atom('num', ch) for ch in tv]
        if tt == 'op':
            if tv == '(':
                body = self.call_args(1, 1)[0]
                return [Paren(Row(body))]
            if tv == '|':
                self.next()
                self.abs_depth += 1
                body = self.expr()
                self.abs_depth -= 1
                self.expect('|')
                return [Abs(Row(body))]
            if tv == '√':
                self.next()
                return [Sqrt(Row(self.radicand()))]
            if tv in ('∛', '∜'):
                self.next()
                idx = '3' if tv == '∛' else '4'
                return [Root(Row([Atom('num', idx)]), Row(self.radicand()))]
            raise ValueError("unexpected '%s'" % tv)
        if tt == 'id':
            self.next()
            name = _NAME_ALIASES.get(tv, tv)
            if name in ('x', 'y'):
                return [Atom('var', name)]
            if name == 'pi':
                return [Atom('const', 'π')]
            if name == 'e':
                return [Atom('const', 'e')]
            if name == 'exp':
                arg = self.func_arg()
                return [Atom('const', 'e'), Sup(Row(arg))]
            if name == 'sqrt':
                return [Sqrt(Row(self.func_arg()))]
            if name == 'abs':
                return [Abs(Row(self.call_args(1, 1)[0]))]
            if name in ('root', 'nroot'):
                a = self.call_args(2, 2)
                return [Root(Row(a[1]), Row(a[0]))]
            if name == 'cbrt':
                return [Root(Row([Atom('num', '3')]), Row(self.func_arg()))]
            if name == 'lg':
                return [Atom('func', 'log'), Paren(Row(self.func_arg()))]
            if name == 'log':
                base = None
                tt2, tv2 = self.peek()
                if self.is_op('_'):
                    self.next()
                    base = self.log_base()
                elif tt2 == 'subd':
                    self.next()
                    base = [Atom('num', ch) for ch in tv2]
                if self.is_op('('):
                    args = self.call_args(1, 2)
                    arg = args[0]
                    if len(args) == 2:
                        base = args[1]
                else:
                    arg = self.func_arg()
                if base is None:
                    return [Atom('func', 'log'), Paren(Row(arg))]
                return [Atom('func', 'log'), Sub(Row(base)), Paren(Row(arg))]
            if name == 'acot':
                raise ValueError("arccot is not supported")
            if name in FUNC_TOKENS:
                return [Atom('func', FUNC_TOKENS[name]), Paren(Row(self.func_arg()))]
            raise ValueError("unknown identifier %r" % tv)
        raise ValueError("unexpected end of expression")


# ═════════════════════════════════════════════════════════════
#  ВИД: РАСКЛАДКА (BOXES)
# ═════════════════════════════════════════════════════════════

PREFERRED_FONTS = ["TeX Gyre Schola", "TeXGyreSchola", "Century Schoolbook", "Times New Roman",
                   "Liberation Serif", "Cambria", "DejaVu Serif"]

# Пропорции (в долях размера шрифта px), откалиброваны по Liberation Serif /
# Times New Roman: ось дроби ≈ центр знака «−», x-height ≈ 0.47 em,
# высота цифр ≈ 0.63 em, скобки +0.67/−0.17 em.
_AXIS = 0.33       # высота математической оси над базовой линией
_ASC = 0.74        # «рабочая» высота строки над базовой линией
_DESC = 0.22       # «рабочая» глубина строки под базовой линией
_SUP_SCALE = 0.72
_SUB_SCALE = 0.72
_IDX_SCALE = 0.6
_FRAC_SCALE = 1.0
_MIN_PX = 8


class _Fonts:
    """Кэш шрифтов и метрик для одного Tk-приложения."""
    _family = None

    def __init__(self, widget):
        self.widget = widget
        self.cache = {}
        if _Fonts._family is None:
            try:
                fams = set(tkfont.families(widget))
            except Exception:
                fams = set()
            fam = next((f for f in PREFERRED_FONTS if f in fams), None)
            _Fonts._family = fam or PREFERRED_FONTS[1]
        self.family = _Fonts._family

    def font(self, px, italic=False):
        key = (px, italic)
        f = self.cache.get(key)
        if f is None:
            f = tkfont.Font(root=self.widget, family=self.family, size=-int(px),
                            slant='italic' if italic else 'roman')
            self.cache[key] = f
        return f

    def measure(self, text, px, italic=False):
        return self.font(px, italic).measure(text)

    @staticmethod
    def asc(px):
        return max(1, int(round(px * _ASC)))

    @staticmethod
    def desc(px):
        return max(1, int(round(px * _DESC)))

    @staticmethod
    def axis(px):
        return max(1, int(round(px * _AXIS)))

    def real_desc(self, px, italic=False):
        key = ('desc', px, italic)
        v = self.cache.get(key)
        if v is None:
            v = self.font(px, italic).metrics('descent')
            self.cache[key] = v
        return v


def _scaled(px, k):
    return max(_MIN_PX, int(px * k + 0.5))


class _RowInfo:
    """Абсолютное положение строки на холсте — для курсора и кликов."""
    __slots__ = ('row', 'x0', 'x1', 'top', 'bottom', 'base', 'slots', 'px', 'depth', 'empty')

    def __init__(self, row, x0, x1, top, bottom, base, slots, px, depth, empty):
        self.row, self.x0, self.x1, self.top, self.bottom = row, x0, x1, top, bottom
        self.base, self.slots, self.px, self.depth, self.empty = base, slots, px, depth, empty


class _Box:
    """Прямоугольник раскладки: ширина, подъём над базовой линией, глубина под ней."""
    __slots__ = ('w', 'asc', 'desc')

    def __init__(self, w=0, asc=0, desc=0):
        self.w, self.asc, self.desc = w, asc, desc

    @property
    def h(self):
        return self.asc + self.desc

    def draw(self, ed, x, base):
        pass


class _TextBox(_Box):
    __slots__ = ('text', 'font', 'px', 'italic', 'font_desc')

    def __init__(self, ed, text, px, italic=False):
        f = ed.fonts.font(px, italic)
        w = f.measure(text)
        if italic:
            w += 1                    # поправка на наклон курсива
        super().__init__(w, _Fonts.asc(px), _Fonts.desc(px))
        self.text, self.font, self.px, self.italic = text, f, px, italic
        self.font_desc = ed.fonts.real_desc(px, italic)

    def draw(self, ed, x, base):
        # anchor='sw': y — низ строки шрифта = базовая линия + реальный descent
        ed.cv.create_text(x, base + self.font_desc, text=self.text,
                          anchor='sw', font=self.font, fill=ed.fg, tags='content')


class _PlaceholderBox(_Box):
    """Пустая строка: маленький пунктирный квадрат (как в Desmos)."""
    __slots__ = ('px',)

    def __init__(self, px):
        w = max(7, int(round(px * 0.6)))
        super().__init__(w, max(5, int(round(px * 0.62))), max(1, int(round(px * 0.08))))
        self.px = px

    def draw(self, ed, x, base, active=False):
        # с курсором внутри — светлая заливка цвета акцента, иначе серый пунктир
        if active:
            ed.cv.create_rectangle(x + 0.5, base - self.asc + 0.5, x + self.w - 0.5, base + self.desc - 0.5,
                                   outline=ed.active_color, fill=ed.placeholder_fill, tags='content')
        else:
            ed.cv.create_rectangle(x + 0.5, base - self.asc + 0.5, x + self.w - 0.5, base + self.desc - 0.5,
                                   outline=ed.placeholder_color, dash=(2, 2), tags='content')


class _RowBox(_Box):
    __slots__ = ('row', 'px', 'parts', 'slots', 'depth', 'placeholder')

    def __init__(self, row, px, depth):
        super().__init__()
        self.row, self.px, self.depth = row, px, depth
        self.parts = []          # [(box, dx)]
        self.slots = []          # dx курсорных позиций 0..n
        self.placeholder = None

    def draw(self, ed, x, base):
        if self.placeholder is not None:
            active = ed.model.cursor[0] is self.row
            self.placeholder.draw(ed, x, base, active)
        for box, dx in self.parts:
            box.draw(ed, x + dx, base)
        ed._rows.append(_RowInfo(
            self.row, x, x + self.w, base - self.asc, base + self.desc, base,
            [x + s for s in self.slots], self.px, self.depth,
            self.placeholder is not None))


class _FracBox(_Box):
    __slots__ = ('num', 'den', 'num_up', 'den_up', 'axis', 'pad')

    def __init__(self, ed, num, den, px):
        axis = _Fonts.axis(px)
        gap = max(1, int(round(px * 0.12)))
        pad = max(2, int(round(px * 0.13)))
        self.num, self.den, self.axis, self.pad = num, den, axis, pad
        # Спуск числителя берём не меньше реального спуска шрифта (у курсивного
        # «y» и скобок хвосты достают до него) + 1 px, чтобы не касаться черты.
        try:
            real_desc = ed.fonts.real_desc(px, italic=True)
        except Exception:
            real_desc = num.desc
        self.num_up = axis + 1 + gap + max(num.desc, real_desc) + 1   # базовая линия числителя (вверх)
        self.den_up = axis - 1 - gap - den.asc           # базовая линия знаменателя (вверх, < 0)
        super().__init__(max(num.w, den.w) + 2 * pad,
                         self.num_up + num.asc,
                         -(self.den_up - den.desc))

    def draw(self, ed, x, base):
        self.num.draw(ed, x + (self.w - self.num.w) // 2, base - self.num_up)
        self.den.draw(ed, x + (self.w - self.den.w) // 2, base - self.den_up)
        y = base - self.axis
        ed.cv.create_line(x, y, x + self.w, y, fill=ed.fg, width=1, tags='content')


class _ShiftBox(_Box):
    """Строка, сдвинутая по вертикали (степень / индекс)."""
    __slots__ = ('inner', 'up', 'lpad')

    def __init__(self, inner, up, lpad=0, rpad=1):
        self.inner, self.up, self.lpad = inner, up, lpad
        super().__init__(inner.w + lpad + rpad, max(0, up + inner.asc), max(0, inner.desc - up))

    def draw(self, ed, x, base):
        self.inner.draw(ed, x + self.lpad, base - self.up)


class _SqrtBox(_Box):
    __slots__ = ('body', 'idx', 'rad_w', 'rad_x', 'idx_up', 'px', 'body_x')

    def __init__(self, ed, body, px, idx=None):
        gap = max(1, int(round(px * 0.1)))
        asc = body.asc + gap + 2
        desc = body.desc + 1
        h = asc + desc
        rad_w = int(round(px * 0.5)) + int(round(max(0, h - _Fonts.asc(px) - _Fonts.desc(px)) * 0.12))
        self.body, self.idx, self.rad_w, self.px = body, idx, rad_w, px
        rad_x = 0
        self.idx_up = 0
        if idx is not None:
            # показатель корня: правый край над «галочкой», низ — чуть выше середины
            rad_x = max(0, idx.w - int(round(rad_w * 0.45)) + 1)
            self.idx_up = int(round(h * 0.55)) - desc + idx.desc
            asc = max(asc, self.idx_up + idx.asc)
        self.rad_x = rad_x
        self.body_x = rad_x + rad_w + 1
        super().__init__(self.body_x + body.w + max(2, int(round(px * 0.12))), asc, desc)

    def draw(self, ed, x, base):
        body = self.body
        gap_top = base - body.asc - max(1, int(round(self.px * 0.1))) - 1   # y черты
        y_top = gap_top + 0.5
        y_bot = base + body.desc + 1
        h = y_bot - y_top
        rx = x + self.rad_x
        rw = self.rad_w
        lw = 1.0 + max(0.0, (h / (self.px * 1.1) - 1.0)) * 0.25
        # «галочка»: короткий штрих вниз, длинный вверх, затем черта
        ed.cv.create_line(rx, y_top + h * 0.6, rx + rw * 0.35, y_bot,
                          fill=ed.fg, width=lw, tags='content')
        ed.cv.create_line(rx + rw * 0.35, y_bot, rx + rw, y_top,
                          fill=ed.fg, width=lw + 0.4, tags='content')
        ed.cv.create_line(rx + rw, y_top, x + self.w - 1, y_top,
                          fill=ed.fg, width=1, tags='content')
        body.draw(ed, x + self.body_x, base)
        if self.idx is not None:
            self.idx.draw(ed, x, base - self.idx_up)


class _AbsBox(_Box):
    __slots__ = ('body', 'pad')

    def __init__(self, ed, body, px):
        self.body = body
        self.pad = max(2, int(round(px * 0.15)))
        super().__init__(body.w + 2 * (self.pad + 1) + 2, body.asc + 1, body.desc + 1)

    def draw(self, ed, x, base):
        top, bot = base - self.asc, base + self.desc
        x1 = x + 1.5
        x2 = x + self.w - 1.5
        ed.cv.create_line(x1, top, x1, bot, fill=ed.fg, width=1, tags='content')
        ed.cv.create_line(x2, top, x2, bot, fill=ed.fg, width=1, tags='content')
        self.body.draw(ed, x + self.pad + 2, base)


class _ParenBox(_Box):
    __slots__ = ('body', 'pw', 'px', 'lw')

    def __init__(self, ed, body, px):
        self.body, self.px = body, px
        h = body.asc + body.desc + 2
        line_h = _Fonts.asc(px) + _Fonts.desc(px)
        self.pw = int(round(px * 0.27)) + int(round(max(0, h - line_h) * 0.07))
        self.lw = 1.0 + max(0.0, (h / line_h - 1.0)) * 0.3
        super().__init__(body.w + 2 * (self.pw + 2), body.asc + 1, body.desc + 1)

    def draw(self, ed, x, base):
        top, bot = base - self.asc, base + self.desc
        pw = self.pw
        cv = ed.cv
        cv.create_arc(x + 1, top, x + 1 + 2 * pw, bot, start=90, extent=180,
                      style='arc', outline=ed.fg, width=self.lw, tags='content')
        cv.create_arc(x + self.w - 1 - 2 * pw, top, x + self.w - 1, bot, start=270, extent=180,
                      style='arc', outline=ed.fg, width=self.lw, tags='content')
        self.body.draw(ed, x + pw + 2, base)


# ═════════════════════════════════════════════════════════════
#  ВИД: ВИДЖЕТ
# ═════════════════════════════════════════════════════════════

class MathEditor(tk.Frame):
    """
    Редактор формулы на Canvas. Ввод только через insert_token().
    on_change(editor) вызывается после каждого изменения модели,
    on_focus(editor) — при клике мышью по виджету.
    """

    def __init__(self, master, model=None, font_size=15, on_change=None, on_focus=None,
                 width=300, height=44, **kw):
        self.fg = kw.pop('fg', '#1a1a2e')
        bg = kw.pop('bg', '#ffffff')
        self.border_color = kw.pop('border_color', '#d1d5db')
        self.active_color = kw.pop('active_color', '#2a7ae0')
        self.error_color = kw.pop('error_color', '#d93025')
        self.placeholder_color = kw.pop('placeholder_color', '#b0b6bf')
        self.placeholder_fill = kw.pop('placeholder_fill', '#e4eefb')
        self.padx = kw.pop('padx', 6)
        self.pady = kw.pop('pady', 4)
        kw.setdefault('highlightthickness', 1)
        kw.setdefault('highlightbackground', self.border_color)
        kw.setdefault('highlightcolor', self.border_color)
        super().__init__(master, bg=bg, width=width, height=height, **kw)

        self.model = model if model is not None else MathModel()
        self.font_size = int(font_size)
        self.on_change = on_change
        self.on_focus = on_focus
        self.min_height = int(height)
        self._active = False
        self._error = None
        self._cursor_visible = True
        self._blink_job = None
        self._tip = None
        self._tip_job = None
        self._rows = []
        self._content_w = 0
        self._xoff = 0
        self._render_job = None

        self.fonts = _Fonts(self)
        self.cv = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0,
                            width=max(10, width - 2), height=max(10, height - 2),
                            takefocus=0, cursor='xterm')
        self.cv.pack(fill='both', expand=True)

        self.cv.bind('<Button-1>', self._on_click)
        self.bind('<Button-1>', self._on_frame_click)      # клик по рамке тоже активирует
        self.cv.bind('<Configure>', lambda e: self._schedule_render())
        self.cv.bind('<Enter>', self._on_enter)
        self.cv.bind('<Leave>', self._on_leave)
        self.bind('<Destroy>', self._on_destroy)
        if KEYBOARD_INPUT:
            self._bind_debug_keys()

        self._render()

    # ── публичный API ───────────────────────────────────────
    def insert_token(self, token):
        # on_change вызываем только если СОДЕРЖИМОЕ изменилось: навигация
        # (left/right/up/down) и холостые токены не должны запускать
        # перерисовку графика в приложении.
        before = self.model.to_json()
        self.model.insert_token(token)
        self._cursor_visible = True
        self._render()
        self._restart_blink()
        if self.on_change is not None and self.model.to_json() != before:
            self.on_change(self)

    def set_active(self, active):
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self._cursor_visible = active
        self._update_border()
        self._render()
        self._restart_blink()

    def is_active(self):
        return self._active

    def set_error(self, msg):
        self._error = msg if msg else None
        self._update_border()

    def get_error(self):
        return self._error

    def get_sympy(self):
        return self.model.to_sympy()

    def get_display(self):
        return self.model.to_display()

    def is_empty(self):
        return self.model.is_empty()

    def set_model(self, model, notify=False):
        self.model = model if model is not None else MathModel()
        self._render()
        if notify and self.on_change is not None:
            self.on_change(self)

    def set_text(self, text, notify=False):
        """Удобно для загрузки старых проектов: текст → модель (ValueError при ошибке)."""
        self.set_model(MathModel.from_text(text), notify=notify)

    # ── внутреннее: рамка, мигание, подсказка ───────────────
    def _update_border(self):
        if self._error:
            color = self.error_color
        elif self._active:
            color = self.active_color
        else:
            color = self.border_color
        try:
            self.configure(highlightbackground=color, highlightcolor=color)
        except tk.TclError:
            pass

    def _restart_blink(self):
        if self._blink_job is not None:
            try:
                self.after_cancel(self._blink_job)
            except Exception:
                pass
            self._blink_job = None
        if self._active:
            self._blink_job = self.after(530, self._blink)

    def _blink(self):
        self._blink_job = None
        if not self._active or not self.winfo_exists():
            return
        self._cursor_visible = not self._cursor_visible
        self._draw_cursor()
        self._blink_job = self.after(530, self._blink)

    def _on_destroy(self, event=None):
        if event is not None and event.widget is not self:
            return
        for job in (self._blink_job, self._tip_job, self._render_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._blink_job = self._tip_job = self._render_job = None
        self._hide_tip()

    def _on_enter(self, event=None):
        if self._error and self._tip_job is None:
            self._tip_job = self.after(400, self._show_tip)

    def _on_leave(self, event=None):
        if self._tip_job is not None:
            try:
                self.after_cancel(self._tip_job)
            except Exception:
                pass
            self._tip_job = None
        self._hide_tip()

    def _show_tip(self):
        self._tip_job = None
        if not self._error or not self.winfo_exists():
            return
        self._hide_tip()
        try:
            tip = tk.Toplevel(self)
            tip.wm_overrideredirect(True)
            tk.Label(tip, text=self._error, bg='#fff4f4', fg=self.error_color,
                     relief='solid', bd=1, padx=6, pady=3,
                     font=('Segoe UI', 9)).pack()
            x = self.winfo_rootx() + 8
            y = self.winfo_rooty() + self.winfo_height() + 2
            tip.wm_geometry('+%d+%d' % (x, y))
            self._tip = tip
        except tk.TclError:
            self._tip = None

    def _hide_tip(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    # ── отладочный ввод с клавиатуры (только при KEYBOARD_INPUT) ──
    def _bind_debug_keys(self):
        self.cv.configure(takefocus=1)
        keymap = {'Left': 'left', 'Right': 'right', 'Up': 'up', 'Down': 'down',
                  'BackSpace': 'backspace', 'slash': '/', 'asciicircum': '^',
                  'plus': '+', 'minus': '-', 'asterisk': '*', 'equal': '=',
                  'parenleft': '(', 'parenright': ')', 'period': '.'}

        def on_key(e):
            tok = keymap.get(e.keysym)
            if tok is None and len(e.char) == 1 and (e.char.isdigit() or e.char in 'xy'):
                tok = e.char
            if tok is not None:
                self.insert_token(tok)
                return 'break'
        self.cv.bind('<KeyPress>', on_key)
        self.cv.bind('<Button-1>', lambda e: self.cv.focus_set(), add='+')

    # ── раскладка ───────────────────────────────────────────
    def _layout_row(self, row, px, depth):
        rb = _RowBox(row, px, depth)
        items = row.items
        if not items:
            ph = _PlaceholderBox(px)
            rb.placeholder = ph
            rb.w, rb.asc, rb.desc = ph.w, max(ph.asc, _Fonts.asc(px)), max(ph.desc, _Fonts.desc(px))
            rb.slots = [ph.w // 2]
            return rb
        x = 0
        prev_box = None
        prev_item = None
        boxes = []
        for i, it in enumerate(items):
            box = self._layout_node(it, px, depth, prev_box, prev_item)
            gap_l, gap_r = self._gaps(it, items, i, px)
            x += gap_l
            rb.slots.append(x - (gap_l + 1) // 2 if i > 0 else x)
            rb.parts.append((box, x))
            x += box.w + gap_r
            boxes.append(box)
            prev_box, prev_item = box, it
        rb.slots.append(x)
        rb.slots[0] = 0
        rb.w = x
        rb.asc = max([_Fonts.asc(px)] + [b.asc for b in boxes])
        rb.desc = max([_Fonts.desc(px)] + [b.desc for b in boxes])
        return rb

    @staticmethod
    def _gaps(it, items, i, px):
        """Отступы слева/справа от элемента (тонкие пробелы вокруг операторов)."""
        if isinstance(it, Atom):
            if it.kind == 'op':
                prev = items[i - 1] if i > 0 else None
                unary = prev is None or _is_op(prev)
                if it.text == '=':
                    g = max(2, int(round(px * 0.3)))
                    return g, g
                if unary:
                    return 0, max(1, int(round(px * 0.04)))
                g = max(2, int(round(px * 0.17)))
                return g, g
            if it.kind == 'func':
                return (max(1, int(round(px * 0.06))) if i > 0 else 0), 0
            if it.kind == 'num':
                prev = items[i - 1] if i > 0 else None
                if prev is not None and not (isinstance(prev, Atom) and prev.kind in ('num', 'op')):
                    return max(1, int(round(px * 0.08))), 0
                return 0, 0
            return 0, 0
        if isinstance(it, (Sup, Sub)):
            return 0, 0
        if isinstance(it, Frac):
            g = max(2, int(round(px * 0.12)))
            prev = items[i - 1] if i > 0 else None
            return (g if prev is not None else 0), g
        if isinstance(it, Paren):
            prev = items[i - 1] if i > 0 else None
            if isinstance(prev, (Sub,)) or (isinstance(prev, Atom) and prev.kind == 'func'):
                return 0, 1
            return (1 if prev is not None else 0), 1
        return (1 if i > 0 else 0), 1

    def _layout_node(self, node, px, depth, prev_box, prev_item):
        if isinstance(node, Atom):
            # Переменные и e — курсивом, π — прямая (в Times New Roman она читается однозначно)
            italic = node.kind == 'var' or (node.kind == 'const' and node.text == 'e')
            return _TextBox(self, node.text, px, italic)
        if isinstance(node, Frac):
            fpx = _scaled(px, _FRAC_SCALE)
            num = self._layout_row(node.num, fpx, depth + 1)
            den = self._layout_row(node.den, fpx, depth + 1)
            return _FracBox(self, num, den, px)
        if isinstance(node, Sup):
            spx = _scaled(px, _SUP_SCALE)
            exp = self._layout_row(node.exp, spx, depth + 1)
            base_asc = prev_box.asc if prev_box is not None else _Fonts.asc(px)
            up = max(int(round(px * 0.42)), base_asc - int(round(exp.asc * 0.55)))
            return _ShiftBox(exp, up, lpad=0, rpad=1)
        if isinstance(node, Sub):
            spx = _scaled(px, _SUB_SCALE)
            sub = self._layout_row(node.sub, spx, depth + 1)
            down = int(round(px * 0.22))
            return _ShiftBox(sub, -down, lpad=0, rpad=1)
        if isinstance(node, Sqrt):
            body = self._layout_row(node.body, px, depth + 1)
            return _SqrtBox(self, body, px)
        if isinstance(node, Root):
            body = self._layout_row(node.body, px, depth + 1)
            idx = self._layout_row(node.index, _scaled(px, _IDX_SCALE), depth + 1)
            return _SqrtBox(self, body, px, idx)
        if isinstance(node, Abs):
            body = self._layout_row(node.body, px, depth + 1)
            return _AbsBox(self, body, px)
        if isinstance(node, Paren):
            body = self._layout_row(node.body, px, depth + 1)
            return _ParenBox(self, body, px)
        return _Box()

    # ── отрисовка ───────────────────────────────────────────
    def _schedule_render(self):
        if self._render_job is None:
            try:
                self._render_job = self.after_idle(self._render_scheduled)
            except tk.TclError:
                pass

    def _render_scheduled(self):
        self._render_job = None
        if self.winfo_exists():
            self._render()

    def _render(self):
        cv = self.cv
        try:
            root = self._layout_row(self.model.root, self.font_size, 0)
        except tk.TclError:
            return
        need_h = root.asc + root.desc + 2 * self.pady
        h = max(self.min_height - 2, need_h)
        cur_h = int(cv.cget('height'))
        if cur_h != h:
            cv.configure(height=h)
            try:
                self.configure(height=h + 2)
            except tk.TclError:
                pass
        base = self.pady + root.asc + (h - need_h) // 2
        cv.delete('all')
        self._rows = []
        root.draw(self, self.padx, base)
        self._content_w = root.w + 2 * self.padx
        vis_w = max(cv.winfo_width(), 1)
        cv.configure(scrollregion=(0, 0, max(self._content_w, vis_w), h))
        self._draw_cursor()

    def _cursor_geometry(self):
        row, idx = self.model.cursor
        for info in self._rows:
            if info.row is row:
                if not info.slots:
                    return None
                x = info.slots[min(idx, len(info.slots) - 1)]
                top = info.base - _Fonts.asc(info.px) - 1
                bot = info.base + _Fonts.desc(info.px) + 1
                return x, top, bot
        # курсор указывает на строку, которой уже нет — переставим в конец
        self.model.cursor_to_end()
        return None

    def _draw_cursor(self):
        cv = self.cv
        cv.delete('cursor')
        if not self._active:
            return
        geom = self._cursor_geometry()
        if geom is None:
            return
        x, top, bot = geom
        if self._cursor_visible:
            cv.create_line(x + 0.5, top, x + 0.5, bot, fill=self.fg, width=1, tags='cursor')
        self._scroll_to(x)

    def _scroll_to(self, x):
        cv = self.cv
        vis_w = cv.winfo_width()
        if vis_w <= 1:
            return
        total = max(self._content_w, vis_w)
        margin = 12
        xoff = self._xoff
        if self._content_w <= vis_w:
            xoff = 0
        elif x > xoff + vis_w - margin:
            xoff = x - vis_w + margin
        elif x < xoff + margin:
            xoff = max(0, x - margin)
        xoff = max(0, min(xoff, total - vis_w))
        self._xoff = xoff
        cv.xview_moveto(xoff / float(total))

    # ── мышь ────────────────────────────────────────────────
    def _on_click(self, event):
        x = self.cv.canvasx(event.x)
        y = self.cv.canvasy(event.y)
        self._place_cursor_at(x, y)
        self._cursor_visible = True
        if not self._active:
            self._active = True
            self._update_border()
        self._draw_cursor()
        self._restart_blink()
        if self.on_focus is not None:
            self.on_focus(self)

    def _on_frame_click(self, event):
        if event.widget is not self:
            return
        self._cursor_visible = True
        if not self._active:
            self._active = True
            self._update_border()
        self._draw_cursor()
        self._restart_blink()
        if self.on_focus is not None:
            self.on_focus(self)

    def _place_cursor_at(self, x, y):
        best = None
        for info in self._rows:
            if info.x0 - 2 <= x <= info.x1 + 2 and info.top - 2 <= y <= info.bottom + 2:
                if best is None or info.depth > best.depth:
                    best = info
        if best is None:
            # вне всех строк: ближайшая по вертикали среди строк глубины 0
            roots = [i for i in self._rows if i.depth == 0]
            if not roots:
                return
            best = roots[0]
        if not best.slots:
            return
        idx = min(range(len(best.slots)), key=lambda k: abs(best.slots[k] - x))
        if best.empty:
            idx = 0
        self.model.cursor = (best.row, idx)
