# -*- coding: utf-8 -*-
"""
make_notebooks.py — دفتر Jupyter (nbformat 4.5) لكل جزء من الأجزاء العشرة:
- خلية Markdown تعريفية + خلية إعداد (تثبيت الناقص + اكتشاف CPU/GPU)
- خلية Markdown لكل فصل، وخلية كود لكل كتلة بايثون، وأسوار Markdown لباقي اللغات
- أسماء ASCII داخل code/notebooks/ والتحقق من صحة JSON
"""
import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from make_code import PART_SLUGS, SLUGS

ROOT = Path(__file__).resolve().parent.parent
NBDIR = ROOT / "code" / "notebooks"
NBDIR.mkdir(parents=True, exist_ok=True)

SETUP_CELL = '''# ⚙️ خلية الإعداد — شغّلها أولًا (Colab / Kaggle / محليًا)
# تثبّت ما ينقص فقط، وتتجاهل ما لا يدعم جهازك.
import sys, subprocess, importlib

PKGS = [
    ("torch", "torch"), ("transformers", "transformers"),
    ("datasets", "datasets"), ("accelerate", "accelerate"),
    ("peft", "peft"), ("trl", "trl"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"), ("pandas", "pandas"),
    ("numpy", "numpy"), ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"), ("evaluate", "evaluate"),
]
todo = []
for mod, pipname in PKGS:
    try:
        importlib.import_module(mod)
    except Exception:
        todo.append(pipname)
if todo:
    print("تثبيت الحزم الناقصة:", todo)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
    except Exception as e:
        print("تعذّر تثبيت بعض الحزم (ثبّتها يدويًا إن احتجتها):", e)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print("device:", DEVICE)
print("جاهز ✅")
'''


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook(book, pt):
    pn = pt["num"]
    chs = [c for c in book["chapters"] if c["part"] == pn]
    py_total = sum(1 for c in chs for b in c["blocks"]
                   if b["t"] == "code" and b.get("lang") in ("python", "py"))
    intro = [
        f"# الجزء {pt['ordinal']}: {pt['title']}",
        "",
        f"دفتر كود هذا الجزء — **{len(chs)} فصلًا** و**{py_total} كتلة بايثون**. "
        "الكتل مرتّبة كما وردت في الكتاب، ويمكن تشغيلها خليةً خلية.",
        "",
        "**Colab:** File ▸ Upload notebook، ثم شغّل خلية الإعداد. "
        "**Kaggle:** أضف الدفتر ثم شغّل الخلايا.",
        "",
        "**CPU/GPU:** كل شيء يعمل على CPU، لكن فصول التدريب أسرع على GPU "
        "(Colab: Runtime ▸ Change runtime type ▸ T4/A100 — Kaggle: Settings ▸ Accelerator ▸ GPU).",
        "",
        "## فصول الجزء",
        "",
    ]
    intro += [f"- **الفصل {c['num']}:** {c['title']}" for c in chs]
    cells = [md_cell("\n".join(intro)), code_cell(SETUP_CELL)]

    for c in chs:
        blocks = B.render_blocks(c)
        codes = [b for b in blocks if b["t"] == "code"]
        if not codes:
            continue
        cells.append(md_cell(
            f"## الفصل {c['num']}: {c['title']}\n\nالجزء {c['part']} · عدد الكتل: {len(codes)}"))
        for b in codes:
            lang = (b.get("lang") or "text").lower()
            code = b["code"].rstrip("\n")
            if lang in ("python", "py"):
                cells.append(code_cell(code + "\n"))
            else:
                fence = "```" + ("" if lang in ("text", "txt") else lang)
                cells.append(md_cell(fence + "\n" + code + "\n```"))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fname = f"part{pn:02d}_{PART_SLUGS.get(pn, 'part')}.ipynb"
    path = NBDIR / fname
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, len(cells), py_total


def main():
    book = B.load()
    total_cells = 0
    report = []
    for pt in book["part_titles"]:
        p, n, py = build_notebook(book, pt)
        total_cells += n
        report.append((p.name, n, py, p.stat().st_size))
        # تحقق JSON
        json.loads(p.read_text(encoding="utf-8"))
    print(f"notebooks: {len(report)} · cells: {total_cells}")
    for name, n, py, size in report:
        print(f"  {name}  cells={n} py={py} {size//1024}KB")


if __name__ == "__main__":
    main()
