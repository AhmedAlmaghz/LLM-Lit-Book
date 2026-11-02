# -*- coding: utf-8 -*-
"""
qa_final.py — فحص نهائي شامل لكل المخرجات:
PDF (صفحات/تسريبات/صور) · DOCX (بنية/RTL) · EPUB (روابط/mimetype) ·
MD (فصول/مخططات/رياضيات خام) · HTML (أقسام/صور) · code (ملفات/دفاتر JSON)
يكتب التقرير في build/qa_final.txt
"""
import re, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
out = []


def head(t):
    out.append("")
    out.append("=" * 18 + " " + t + " " + "=" * 18)


# ---------------------------------------------------------------- PDF
head("PDF")
import pymupdf
d = pymupdf.open(DIST / "llm-ds.pdf")
leaks = [i + 1 for i in range(d.page_count)
         if re.search(r"\\frac\{|\\begin\{|\\sum_|\\mathbf", d[i].get_text())]
sent = [i + 1 for i in range(d.page_count) if "⟦" in d[i].get_text()]
imgs = sum(len(d[i].get_images(full=True)) for i in range(d.page_count))
out.append(f"pages={d.page_count} size={d[0].rect.width:.0f}x{d[0].rect.height:.0f}pt "
           f"latex_leaks={len(leaks)} sentinels={len(sent)} image_refs={imgs}")

# ---------------------------------------------------------------- DOCX
head("DOCX")
from docx import Document
from docx.oxml.ns import qn
doc = Document(DIST / "llm-ds.docx")
h1 = sum(1 for p in doc.paragraphs if p.style.name == "Heading 1")
h2 = sum(1 for p in doc.paragraphs if p.style.name == "Heading 2")
bidi = sum(1 for p in doc.paragraphs
           if p._p.pPr is not None and p._p.pPr.find(qn("w:bidi")) is not None)
imgs_docx = len(list(doc.part.package.image_parts))
real_stars = 0
for p in doc.paragraphs:
    if "**" not in p.text:
        continue
    fonts = set()
    for r in p.runs:
        rf = r._r.rPr.find(qn("w:rFonts")) if r._r.rPr is not None else None
        fonts.add(rf.get(qn("w:ascii")) if rf is not None else None)
    if not fonts <= {"Consolas", None}:
        real_stars += 1
out.append(f"paras={len(doc.paragraphs)} tables={len(doc.tables)} images={imgs_docx} "
           f"H1={h1} H2={h2} bidi={bidi} markdown_leaks={real_stars}")

# ---------------------------------------------------------------- EPUB
head("EPUB")
z = zipfile.ZipFile(DIST / "llm-ds.epub")
names = z.namelist()
first = z.infolist()[0]
opf = z.read("OEBPS/content.opf").decode("utf-8")
hrefs = re.findall(r'href="([^"]+)"', opf)
missing = [h for h in hrefs if f"OEBPS/{h}" not in names]
broken = 0
for name in names:
    if not name.endswith(".xhtml"):
        continue
    body = z.read(name).decode("utf-8")
    base = name.rsplit("/", 1)[0]
    from xml.etree import ElementTree as ET
    docx = ET.fromstring(body.encode())
    for el in docx.iter():
        for attr in ("href", "src"):
            v = el.get(attr)
            if not v or v.startswith(("http", "#")):
                continue
            path = v.split("#")[0]
            if not path:
                continue
            parts = (base + "/" + path).split("/")
            stack = []
            for seg in parts:
                if seg == ".." and stack:
                    stack.pop()
                elif seg not in (".", ""):
                    stack.append(seg)
            if "/".join(stack) not in names:
                broken += 1
out.append(f"entries={len(names)} mimetype_first={first.filename == 'mimetype'} "
           f"stored={first.compress_type == 0} missing_manifest={len(missing)} broken_links={broken}")

# ---------------------------------------------------------------- MD
head("MD")
md = (DIST / "llm-ds.md").read_text(encoding="utf-8")
chs = len(re.findall(r"^## الفصل \d+", md, re.M))
figs = len(re.findall(r"!\[[^\]]*\]\(images/", md))
math_inline = len(re.findall(r"\\\(", md))
math_disp = len(re.findall(r"\\\[", md))
raw = len(re.findall(r"⟦", md))
out.append(f"lines={len(md.splitlines())} chapters={chs} figures={figs} "
           f"inline_math={math_inline} display_math={math_disp} sentinels={raw}")

# ---------------------------------------------------------------- HTML
head("HTML")
html = (DIST / "llm-ds.html").read_text(encoding="utf-8")
out.append(f"bytes={len(html):,} sections={html.count('<section')} figures={html.count('<figure>')} "
           f"codeboxes={html.count('class=\"codebox\"')} imgs={html.count('<img ')} "
           f"title={'<title>' in html} closing={html.rstrip().endswith('</html>')}")

# ---------------------------------------------------------------- code
head("CODE")
py = sorted((ROOT / "code").glob("ch*.py"))
nb = sorted((ROOT / "code" / "notebooks").glob("*.ipynb"))
bad_nb = []
for p in nb:
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        bad_nb.append((p.name, str(e)))
out.append(f"chapter_files={len(py)} notebooks={len(nb)} invalid_notebooks={len(bad_nb)}")
out.append(f"requirements={(ROOT / 'code' / 'requirements.txt').exists()} "
           f"readme={(ROOT / 'code' / 'README.md').exists()}")

# ---------------------------------------------------------------- الأحجام
head("FILES")
for f in sorted(DIST.glob("*")):
    if f.is_file():
        out.append(f"{f.name:22s} {f.stat().st_size/1048576:8.2f} MB")
out.append(f"images={len(list((DIST / 'images').glob('*.png')))} "
           f"images_print={len(list((DIST / 'images_print').glob('*.png')))}")

(ROOT / "build" / "qa_final.txt").write_text("\n".join(out), encoding="utf-8")
print("qa_final -> build/qa_final.txt")
