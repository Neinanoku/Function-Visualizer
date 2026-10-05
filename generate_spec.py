"""
Генерирует FuncVisualizer.spec для PyInstaller.
Запускается автоматически из build.bat
"""
import os
import sys

base = os.path.dirname(os.path.abspath(__file__))

# Файлы данных: (источник, папка назначения внутри exe)
# Модули function_visualizer / math_editor попадают в exe как обычные
# скомпилированные модули (hiddenimports + pathex), копировать исходники
# в datas не нужно.
datas = []
# Необязательные ресурсы: добавляем только если файл есть рядом
for optional in ("ariadna-logo1-trnsp.png", "icon.ico"):
    if os.path.exists(os.path.join(base, optional)):
        datas.append((os.path.join(base, optional), "."))
    else:
        print(f"warning: {optional} not found — building without it")

# Форматируем для spec файла
datas_str = ",\n        ".join(
    f"({repr(src)}, {repr(dst)})" for src, dst in datas
)

spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import sys
block_cipher = None

a = Analysis(
    [{repr(os.path.join(base, 'app.py'))}],
    pathex=[{repr(base)}],
    binaries=[],
    datas=[
        {datas_str}
    ],
    hiddenimports=[
        'function_visualizer',
        'math_editor',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends._backend_tk',
        'sympy',
        'scipy',
        'scipy.signal',
        'scipy.optimize',
        'scipy.special',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL._tkinter_finder',   # импортируется из C-кода PIL/_imagingtk — анализ его не видит
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tests', 'pytest', 'py', '_pytest', 'pluggy', 'iniconfig', 'mpmath.tests',
              'IPython', 'jupyter', 'tkinter.test'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FuncVisualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    icon={repr(os.path.join(base, 'icon.ico')) if os.path.exists(os.path.join(base, 'icon.ico')) else None},
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""

spec_path = os.path.join(base, "FuncVisualizer.spec")
with open(spec_path, "w", encoding="utf-8") as f:
    f.write(spec_content)

print(f"Spec file written: {spec_path}")
