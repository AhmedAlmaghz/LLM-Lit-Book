# -*- coding: utf-8 -*-
"""
qa_final.py — فحص نهائي شامل لكل المخرجات الخمس + الكود:
PDF (رؤوس/تذييلات/أرقام فهرس/إشارات مرجعية/رياضيات خام) ·
DOCX (مقاطع/ترقيم/حقول رأس وتذييل/افتتاحيات/خصائص) ·
EPUB (بيانات وصفية/روابط/XML/رياضيات) ·
MD (فصول/مخططات/LaTeX خام) · HTML (رياضيات/أقسام) · code (ملفات/دفاتر)
يكتب التقرير في build/qa_final.txt
"""
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
AUTHOR = "أحمد صالح المغز"
AUTHOR_DISPLAY = "م. أحمد صالح المغز"
out = []
checks = []


def head(t):
    out.append("")
    out.append("=" * 18 + " " + t + " " + "=" * 18)


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    out.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def strip_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]+`", "", text)


# ---------------------------------------------------------------- PDF
head("PDF")
import pymupdf

pdf_path = DIST / "llm-ds.pdf"
d = pymupdf.open(pdf_path)
pages = d.page_count
leaks, sentinels = [], []
imgs = 0
toc_numbers = 0
foot_pages = head_pages = 0
# مرور واحد على الصفحات: نص + إحداثيات + صور (أسرع بكثير من ثلاث مرورات)
for i in range(pages):
    page = d[i]
    imgs += len(page.get_images(full=True))
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    top = bottom = False
    chunks = []
    for b in data.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if not t.strip():
                    continue
                chunks.append(t)
                r = pymupdf.Rect(sp["bbox"])
                if r.y1 < 45:
                    top = True
                elif r.y0 > 800:
                    bottom = True
                if (i in (2, 3) and t.strip().isdigit()
                        and 7.8 <= sp.get("size", 0) <= 9.4
                        and r.y1 < 780):
                    toc_numbers += 1
    text = "\n".join(chunks)
    if re.search(r"\\frac\{|\\begin\{|\\sum_|\\mathbf", text):
        leaks.append(i + 1)
    if "⟦" in text:
        sentinels.append(i + 1)
    head_pages += top
    foot_pages += bottom

toc = d.get_toc()
meta = d.metadata
out.append(f"pages={pages} size={d[0].rect.width:.0f}x{d[0].rect.height:.0f}pt "
           f"bytes={pdf_path.stat().st_size:,} image_refs={imgs}")
check("PDF: عدد الصفحات 600+", pages >= 600, str(pages))
check("PDF: لا رياضيات LaTeX خام", not leaks, f"leaks={len(leaks)}")
check("PDF: لا علامات تسريب", not sentinels, f"sentinels={len(sentinels)}")
check("PDF: 70 رقم فهرس مطبوع", toc_numbers == 70, str(toc_numbers))
check("PDF: إشارات مرجعية 70+", len(toc) >= 70, str(len(toc)))
check("PDF: تذييل في كل الصفحات عدا الغلاف", foot_pages >= pages - 2,
      f"{foot_pages}/{pages}")
check("PDF: رؤوس في معظم الصفحات", head_pages >= pages // 2, f"{head_pages}/{pages}")
check("PDF: المؤلف في البيانات الوصفية", AUTHOR in (meta.get("author") or ""),
      meta.get("author") or "")
d.close()

# ---------------------------------------------------------------- DOCX
head("DOCX")
from docx import Document
from docx.oxml.ns import qn

docx_path = DIST / "llm-ds.docx"
doc = Document(docx_path)
h1 = sum(1 for p in doc.paragraphs if p.style.name == "Heading 1")
h2 = sum(1 for p in doc.paragraphs if p.style.name == "Heading 2")
bidi = sum(1 for p in doc.paragraphs
           if p._p.pPr is not None and p._p.pPr.find(qn("w:bidi")) is not None)

sections = len(doc.sections)
starts = []
for s in doc.sections:
    pg = s._sectPr.find(qn("w:pgNumType"))
    starts.append(pg.get(qn("w:start")) if pg is not None else None)
title_pg = sum(1 for s in doc.sections if s._sectPr.find(qn("w:titlePg")) is not None)

with zipfile.ZipFile(docx_path) as z:
    names = z.namelist()
    hf = [n for n in names if re.match(r"word/(header|footer)\d*\.xml$", n)]
    instr = ""
    for n in hf:
        xml = z.read(n).decode("utf-8")
        instr += " ".join(re.findall(r"<w:instrText[^>]*>([^<]*)</w:instrText>", xml))
        instr += " ".join(re.findall(r'<w:fldSimple[^>]*w:instr="([^"]*)"', xml))
    body = z.read("word/document.xml").decode("utf-8")
    latex_cmds = (r"\\(?:frac|mathbf|mathrm|mathsf|mathcal|mathbb|text|begin|end|sum|prod|"
                  r"int|sigma|alpha|beta|gamma|theta|lambda|mu|partial|nabla|left|right|"
                  r"cdot|times|sqrt|hat|bar|vec|underbrace|approx|le|ge|neq|infty|log|exp|"
                  r"tanh|softmax|operatorname)")
    docx_macros = len(re.findall(latex_cmds, body))

dark = 0
for t in doc.tables:
    try:
        tcPr = t.rows[0].cells[0]._tc.find(qn("w:tcPr"))
        shd = tcPr.find(qn("w:shd")) if tcPr is not None else None
        fill = (shd.get(qn("w:fill"), "") if shd is not None else "").lower()
        if fill in ("1e1b4b", "31267b", "2e2a6e", "312e81"):
            dark += 1
    except Exception:
        pass

out.append(f"bytes={docx_path.stat().st_size:,} sections={sections} "
           f"paras={len(doc.paragraphs)} tables={len(doc.tables)} "
           f"H1={h1} H2={h2} bidi={bidi} header_footer_parts={len(hf)}")
check("DOCX: 13 مقطعًا (غلاف + مقدمة + 10 أجزاء + ملاحق)",
      sections == 13, str(sections))
check("DOCX: ترقيم الصفحات يبدأ من 0 في المقطع الأول فقط",
      starts[0] == "0" and all(s is None for s in starts[1:]), str(starts[:3]))
check("DOCX: رأس/تذييل مختلف للصفحة الأولى في افتتاحيات الأجزاء",
      title_pg >= 11, str(title_pg))
check("DOCX: حقول STYLEREF و PAGE", "STYLEREF" in instr and "PAGE" in instr)
check("DOCX: 11 لوحة افتتاح أجزاء داكنة", dark >= 10, str(dark))
check("DOCX: لا رياضيات LaTeX خام", docx_macros == 0, str(docx_macros))
check("DOCX: المؤلف في الخصائص", AUTHOR_DISPLAY in (doc.core_properties.author or ""),
      doc.core_properties.author or "")
check("DOCX: اللغة ar", doc.core_properties.language == "ar",
      str(doc.core_properties.language))

# ---------------------------------------------------------------- EPUB
head("EPUB")
epub_path = DIST / "llm-ds.epub"
z = zipfile.ZipFile(epub_path)
names = z.namelist()
first = z.infolist()[0]
opf = z.read("OEBPS/content.opf").decode("utf-8")
hrefs = re.findall(r'href="([^"]+)"', opf)
missing = [h for h in hrefs if f"OEBPS/{h}" not in names]
text = ""
for name in names:
    if name.endswith((".xhtml", ".opf", ".ncx")):
        text += "\n" + z.read(name).decode("utf-8", "ignore")
nocode = re.sub(r"<pre.*?</pre>", "", text, flags=re.S)
nocode = re.sub(r"<code.*?</code>", "", nocode, flags=re.S)
epub_macros = len(re.findall(r"\\[A-Za-z]{2,}", nocode))
epub_delims = len(re.findall(r"\\[\(\[]", nocode))
spans = re.findall(r'<span class="(?:mth|mthd)">', text)
from xml.etree import ElementTree as ET

bad_xml = 0
for name in names:
    if name.endswith((".xhtml", ".opf", ".ncx")):
        try:
            ET.fromstring(z.read(name))
        except Exception:
            bad_xml += 1
creator = re.search(r"<dc:creator[^>]*>([^<]+)</dc:creator>", opf)
creator = creator.group(1) if creator else ""
out.append(f"bytes={epub_path.stat().st_size:,} entries={len(names)} "
           f"math_spans={len(spans)} creator={creator}")
check("EPUB: mimetype أولًا وغير مضغوط",
      first.filename == "mimetype" and first.compress_type == 0)
check("EPUB: لا رياضيات LaTeX خام", epub_macros == 0 and epub_delims == 0,
      f"macros={epub_macros} delims={epub_delims}")
check("EPUB: 400+ عنصر رياضيات", len(spans) >= 400, str(len(spans)))
check("EPUB: XML صالح لكل المستندات", bad_xml == 0, str(bad_xml))
check("EPUB: لا روابط مفقودة", not missing, str(len(missing)))
check("EPUB: المؤلف", creator == AUTHOR_DISPLAY, creator)

# ---------------------------------------------------------------- MD
head("MD")
md = (DIST / "llm-ds.md").read_text(encoding="utf-8")
nocode = strip_code(md)
md_macros = len(re.findall(r"\\[A-Za-z]{2,}", nocode))
md_delims = len(re.findall(r"\\[\(\[]", nocode))
md_at = len(re.findall(r"@\(.{0,80}?\)@", nocode, flags=re.S))
chs = len(re.findall(r"^## الفصل \d+", md, re.M))
figs = len(re.findall(r"!\[[^\]]*\]\(images/", md))
out.append(f"bytes={len(md.encode('utf-8')):,} lines={len(md.splitlines())} "
           f"chapters={chs} figures={figs}")
check("MD: 48 فصلًا", chs == 48, str(chs))
check("MD: 33 مخططًا", figs == 33, str(figs))
check("MD: لا LaTeX خام خارج الكود", md_macros == 0 and md_delims == 0,
      f"macros={md_macros} delims={md_delims}")
check("MD: لا رياضيات @(...)@ متبقية", md_at == 0, str(md_at))
check("MD: اسم المؤلف", AUTHOR in md)

# ---------------------------------------------------------------- HTML
head("HTML")
html = (DIST / "llm-ds.html").read_text(encoding="utf-8")
import html as _html

math_spans = re.findall(r'<span class="(?:mth|mthd)">(.*?)</span>', html, flags=re.S)
bad_math = []
for body in math_spans:
    txt = _html.unescape(re.sub(r"<[^>]+>", "", body))
    if any(ch in txt for ch in "\\{}*"):
        bad_math.append(txt)
out.append(f"bytes={len(html):,} sections={html.count('<section')} "
           f"figures={html.count('<figure>')} codeboxes={html.count('class=\"codebox\"')} "
           f"math_spans={len(math_spans)}")
check("HTML: 400+ عنصر رياضيات", len(math_spans) >= 400, str(len(math_spans)))
check("HTML: لا رموز LaTeX داخل الرياضيات", not bad_math, str(len(bad_math)))
check("HTML: 33 مخططًا", html.count("<figure>") == 33, str(html.count("<figure>")))
check("HTML: مستند مكتمل", "<title>" in html and html.rstrip().endswith("</html>"))

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
check("CODE: 40+ ملف فصل", len(py) >= 40, str(len(py)))
check("CODE: دفاتر Colab/Kaggle صالحة", nb and not bad_nb, f"{len(nb)} notebooks")

# ---------------------------------------------------------------- الأحجام
head("FILES")
for f in sorted(DIST.glob("*")):
    if f.is_file():
        out.append(f"{f.name:22s} {f.stat().st_size/1048576:8.2f} MB")
out.append(f"images={len(list((DIST / 'images').glob('*.png')))} "
           f"images_print={len(list((DIST / 'images_print').glob('*.png')))}")

passed = sum(1 for _, ok in checks if ok)
out.insert(0, f"النتيجة: {passed}/{len(checks)} فحصًا ناجحًا")
(ROOT / "build" / "qa_final.txt").write_text("\n".join(out), encoding="utf-8")
print(f"qa_final -> build/qa_final.txt ({passed}/{len(checks)} pass)")
