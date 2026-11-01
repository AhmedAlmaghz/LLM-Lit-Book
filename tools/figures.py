# -*- coding: utf-8 -*-
"""figures.py — الإنتاج الجماعي للمخططات:
يجمع دوال f01..f34 من الملفات الثلاثة، يستدعيها، ويكتب مخرجات SVG إلى figures/.
"""
import sys, importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    mods = [importlib.import_module(m) for m in ("figures1", "figures2", "figures3")]
    figs = {}
    for m in mods:
        for name in sorted(dir(m)):
            if name.startswith("f") and len(name) > 2 and name[1].isdigit():
                fn = getattr(m, name)
                if callable(fn):
                    figs[name] = fn()
    from figurekit import write_all
    out = HERE.parent / "figures"
    write_all(figs, out)
    print(f"total={len(figs)}")


if __name__ == "__main__":
    main()
