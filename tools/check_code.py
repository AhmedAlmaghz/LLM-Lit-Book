# -*- coding: utf-8 -*-
"""فحص صياغة كل كتل بايثون في المخطوطة + تقرير الأخطاء لكل فصل."""
import re, json, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lines = (ROOT / "LLM-ds.md").read_text(encoding="utf-8").splitlines()

out = io.StringIO()
cur_ch = None
blocks = []
in_code = False
lang = ""
buf = []

for i, ln in enumerate(lines, 1):
    m = re.match(r"^# (الفصل|ملحق)", ln)
    if m:
        cur_ch = ln.strip("# ").strip()
    if ln.strip().startswith("```"):
        if not in_code:
            in_code = True
            lang = ln.strip()[3:].strip()
            buf = []
        else:
            in_code = False
            blocks.append((i, cur_ch, lang, "\n".join(buf)))
    elif in_code:
        buf.append(ln)

ok = bad = 0
for ln_no, ch, lang, code in blocks:
    if lang not in ("python", "py", ""):
        continue
    if lang == "":
        # استخدم heuristic: هل يشبه بايثون؟
        head = "\n".join(code.splitlines()[:6])
        py_like = re.search(r"\b(import|def |class |print\(|[a-z_]+ ?=)", head)
        if not py_like:
            continue
    try:
        compile(code.replace("\t", "    "), "<block>", "exec")
        ok += 1
    except SyntaxError as e:
        bad += 1
        msg = f"[{ln_no}] {'/'.join((ch or '')[:28].split())} :: {e}"
        out.write(msg + "\n")

print(f"python-like blocks: ok={ok} syntax-errors={bad}")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
print(out.getvalue())
