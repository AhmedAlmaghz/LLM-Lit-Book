# -*- coding: utf-8 -*-
"""تحقق سريع من ملف docx: الفتح، العدّادات، البنية، العينات."""
from pathlib import Path
from docx import Document

d = Document("dist/llm-ds.docx")
out = []
out.append(f"paragraphs: {len(d.paragraphs)}")
out.append(f"tables: {len(d.tables)}")
imgs = d.part.package.image_parts if hasattr(d.part.package, "image_parts") else []
out.append(f"image parts: {len(list(d.part.package.image_parts))}")
# أنماط العناوين
h1 = [p.text for p in d.paragraphs if p.style.name == "Heading 1"]
h2 = [p.text for p in d.paragraphs if p.style.name == "Heading 2"]
out.append(f"Heading1: {len(h1)} — first 3: {[t[:40] for t in h1[:3]]}")
out.append(f"Heading2: {len(h2)} — first 3: {[t[:40] for t in h2[:3]]}")
# فقرات bidi
bidi = sum(1 for p in d.paragraphs if p._p.pPr is not None and p._p.pPr.find(
    '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}bidi') is not None)
out.append(f"bidi paragraphs: {bidi}")
# عينة من أول فقرة بعد الغلاف
texts = [p.text for p in d.paragraphs if p.text.strip()][:8]
for t in texts:
    out.append("P: " + t[:80])
# لا بقايا نجوم/لا ماركداون خام
stars = [p.text for p in d.paragraphs if "**" in p.text]
out.append(f"paragraphs containing raw '**': {len(stars)}")
if stars:
    out.append("  e.g. " + stars[0][:90])
latex = [p.text for p in d.paragraphs if "\\frac" in p.text or "\\hat" in p.text]
out.append(f"paragraphs containing raw latex: {len(latex)}")
if latex:
    out.append("  e.g. " + latex[0][:90])
Path("build/_probe_docx.txt").write_text("\n".join(out), encoding="utf-8")
print("ok")
