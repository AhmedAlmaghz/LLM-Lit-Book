# -*- coding: utf-8 -*-
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mathlite, build_docx as D

book = json.loads(Path("build/book.json").read_text(encoding="utf-8"))
texts = []
def collect(container):
    for b in container.get("blocks", []):
        t = b["t"]
        if t in ("p", "quote", "h1", "h2", "h3"): texts.append(b.get("text", ""))
        elif t in ("ul", "ol"): texts.extend(b["items"])
        elif t == "table":
            texts.extend(b["header"]); texts.extend(c for r in b["rows"] for c in r)
for c in book["chapters"]: collect(c)
for a in book["appendices"]: collect(a)
for k in ("dedication", "preface", "summary"):
    for b in book["front"][k]:
        if b["t"] in ("p", "quote", "h1", "h2", "h3"): texts.append(b.get("text", ""))
        elif b["t"] in ("ul", "ol"): texts.extend(b["items"])

out = []
# 1) هل توجد ** داخل صيغة رياضية؟
pat = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
n = 0
for tx in texts:
    for m in pat.finditer(tx):
        if "**" in m.group(0):
            n += 1
            out.append("math-with-stars: " + m.group(0)[:90])
out.append(f"math bodies containing '**': {n}")

# 2) الصيغة التي أنتجت ⟦⟧ — أعدها الآن
dpo = [tx for tx in texts if "y_w" in tx and "DPO" in tx]
for tx in dpo:
    m = pat.search(tx)
    if m:
        conv = mathlite.convert(m.group(0)[2:-2] if m.group(0).startswith("\\[") else m.group(0)[2:-2])
        out.append("DPO formula: " + m.group(0)[:200])
        out.append("DPO convert : " + conv[:200])
        out.append("sentinel? " + str("⟦" in conv))

# 3) كل الصيغ: عدّ منتجات بها ⟦
snt = 0
for tx in texts:
    for m in pat.finditer(tx):
        body = m.group(0)
        body = body[2:-2]
        try:
            cv = mathlite.convert(body)
        except Exception:
            cv = "?"
        if "⟦" in cv:
            snt += 1
            if snt <= 5:
                out.append("sentinel body: " + body[:100])
                out.append("sentinel conv : " + cv[:100])
out.append(f"total formulas with sentinel: {snt}")
Path("build/_probe_edge.txt").write_text("\n".join(out), encoding="utf-8")
print("ok")
