# -*- coding: utf-8 -*-
"""
Генератор spec-файлов PyInstaller (generate_spec.py / generate_spec_he.py):
обе сборки должны давать синтаксически корректный spec (регрессия: шаблон,
сдвинутый внутрь функции, давал IndentationError в PyInstaller).

Запуск:
    /usr/bin/python3.12 -m pytest tests/test_build_spec.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_spec  # noqa: E402


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts" / "texgyreschola-regular.otf").write_bytes(b"\0")
    (tmp_path / "fonts" / "LICENSE-TeX-Gyre-Schola.txt").write_text("x")
    for entry in ("app.py", "app_he.py", "app_ru.py"):
        (tmp_path / entry).write_text("pass\n")
    monkeypatch.setattr(generate_spec, "base", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("entry,name", [("app.py", "FuncVisualizer"),
                                        ("app_he.py", "FuncVisualizer_he"),
                                        ("app_ru.py", "FuncVisualizer_ru")])
def test_spec_compiles(fake_project, entry, name):
    path = generate_spec.write_spec(entry, name)
    assert os.path.basename(path) == name + ".spec"
    text = open(path, encoding="utf-8").read()
    compile(text, path, "exec")                    # то, что делает PyInstaller
    assert text.startswith("# -*- mode: python")
    assert "\nimport sys\n" in text                # первая строка без отступа
    assert repr(os.path.join(str(fake_project), entry)) in text
    assert f"name={name!r}" in text
    assert "'fonts')" in text and "texgyreschola-regular.otf" in text
    assert "'PIL._tkinter_finder'" in text
    assert "icon=None" in text                     # icon.ico нет — не падаем


def test_he_generator_uses_hebrew_entry(fake_project):
    import generate_spec_he  # noqa: F401  — импорт без побочных эффектов
    path = generate_spec.write_spec("app_he.py", "FuncVisualizer_he")
    text = open(path, encoding="utf-8").read()
    assert "app_he.py" in text and "FuncVisualizer_he" in text
    assert "'app.py'" not in text
