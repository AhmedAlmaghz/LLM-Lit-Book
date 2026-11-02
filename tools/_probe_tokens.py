# -*- coding: utf-8 -*-
"""فحص نهائي: tokenize على كل النصوص السطرية + إعادة فتح الـdocx."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import build_docx as D

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

bad = {"stars": [], "ph": [], "tex": [], "snt": []}
for tx in texts:
    for s, k in D.tokenize(tx):
        if "**" in s: bad["stars"].append((tx[:60], s[:60]))
        if "\x00" in s: bad["ph"].append((tx[:60], s[:60]))
        if "\\(" in s or "\\[" in s or "\\hat" in s or "\\frac" in s: bad["tex"].append((tx[:60], s[:60]))
        if "⟦" in s: bad["snt"].append((tx[:60], s[:60]))

out = [f"texts: {len(texts)}"]
for k, v in bad.items():
    out.append(f"{k}: {len(v)}")
    for a, b_ in v[:4]:
        out.append(f"   {a!r} -> {b_!r}")
Path("build/_probe_tok.txt").write_text("\n".join(out), encoding="utf-8")
print("token-check ok")
