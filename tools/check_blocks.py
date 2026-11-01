# -*- coding: utf-8 -*-
"""فحص صياغة كل بلاكات بايثون في build/book.json."""
import json, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "build" / "book.json").read_text(encoding="utf-8"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ok = bad = 0
for ch in d["chapters"]:
    for bi, b in enumerate(ch["blocks"]):
        if b["t"] != "code" or b["lang"] not in ("python", "py"):
            continue
        code = "import torch, numpy as np, torch.nn as nn, torch.nn.functional as F\n" if False else b["code"]
        annotated = "\n".join(
            ("    pass" if i == 0 else "pass")
            for i, _ in enumerate([])
        )
        raw = b["code"]
        # نقطة حقن حزمة استثناء: بعض الأمثلة تستخدم client/llm/models جرّب: أضف defensives؟ لا شيء.
        try:
            compile(raw, "<block>", "exec")
            ok += 1
        except SyntaxError as e:
            bad += 1
            where = f"CH{ch['num']:02d}/block{bi}"
            print(f"{where}: line {e.lineno}: {e.msg} :: {raw.splitlines()[e.lineno-1][:70] if e.lineno and e.lineno-1 < len(raw.splitlines()) else ''}")
print(f"python blocks ok={ok} bad={bad}")
