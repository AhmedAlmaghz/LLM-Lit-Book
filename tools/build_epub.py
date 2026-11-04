# -*- coding: utf-8 -*-
"""
build_epub.py — إصدار EPUB3 عربي RTL مبني على zip خام (بلا مكتبات خارجية):
- mimetype غير مضغوط أولًًا، ثم META-INF/container.xml
- OEBPS: content.opf (spine RTL) + nav.xhtml + toc.ncx + style.css
- ملف xhtml لكل جزء (10) + الواجهة + الفهرس + الملاحق + الخاتمة
- صور بحجم نسخة القراءة (images_print ~1560px) + الغلاف
- يعيد استخدام شظايا build_html (inline/block/chapter/front)
- يتحقق من صحة كل مستند XML قبل الكتابة
"""
import re, sys, uuid, zipfile, datetime
from pathlib import Path
from xml.etree import ElementTree as ET
import html as _html

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from booklib import load, FIGS
import build_html as H
import mathlite

RUNMARK_RE = re.compile(r'<span class="runmark">.*?</span>')


def strip_marks(s):
    return RUNMARK_RE.sub("", s)


def plain(s):
    """نص عادي لعناصر التنقل: رياضيات يونيكود + تهريب XML."""
    return _html.escape(mathlite.render(s or ""))

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
OUT = DIST / "llm-ds.epub"

BOOK_ID = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, "https://bookai.example/llm-ds-ar"))
MODIFIED = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

XHTML_HEAD = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" dir="rtl" lang="ar" xml:lang="ar">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="{css}"/>
</head>
<body>
{body}
</body>
</html>"""


def xhtml(title, body, css="../style.css"):
    return XHTML_HEAD.format(title=_html.escape(title), css=css, body=body)


def _resolver(k):
    return f"../images/{k}.png"


# ------------------------------------------------------------------ الأجزاء
def part_section(book, pt):
    pn = pt["num"]
    chs = [c for c in book["chapters"] if c["part"] == pn]
    fig = "".join(
        f'<div class="partfig">{H.figure_html(key, f"شكل ج{pn}", _resolver)}</div>'
        for key in B.figs_for_part(pn))
    return "\n".join([
        f'<section class="part" id="part{pn}">',
        '<div class="part-inner">',
        f'<div class="pghost">{pn:02d}</div>',
        f'<span class="pbadge">الجزء {pt["ordinal"]}</span>',
        f'<h1>{H.inline(pt["title"])}</h1>',
        f'<div class="plight">الفصول {chs[0]["num"]}–{chs[-1]["num"]} · {len(chs)} فصولًا</div>',
        '<div class="pmeta">المحوّلات والنماذج اللغوية الكبيرة · من النظرية إلى بناء أنظمة ذكية</div>',
        fig,
        "</div>",
        "</section>",
    ])


def build_part(book, pt):
    chunks = [part_section(book, pt)]
    for c in [c for c in book["chapters"] if c["part"] == pt["num"]]:
        chunks.append(strip_marks(H.chapter_html(c, _resolver)))
    title = f"الجزء {pt['ordinal']}: {mathlite.render(pt['title'])}"
    return xhtml(title, "\n".join(chunks))


def build_front(book):
    meta = book["meta"]
    st = meta["stats"]
    cover = ('<div class="coverpage">'
             '<img src="../images/cover.png" alt="غلاف الكتاب"/></div>')
    author = meta.get("author_display", meta.get("author", ""))
    head = f"""<div class="dochead">
  <div class="eyebrow">كتاب عملي · إصدار 2026</div>
  <h1>{H.inline(meta['title'])}</h1>
  <div class="sub">{H.inline(meta['subtitle'])}</div>
  <div class="author">{H.inline(author)}</div>
  <div class="meta">{H.inline(meta['edition'])} · {meta['publisher']} ·
   ⏱ ~{st['minutes']} دقيقة قراءة ·
   48 فصلًا · 10 أجزاء · 9 ملاحق · 33 مخططًا</div>
</div>"""
    body = cover + head + "\n" + strip_marks(H.front_html(book["front"], _resolver))
    return xhtml("المقدمة والهداء", body)


def build_toc(book):
    parts = book["part_titles"]
    out = ['<section class="tocpage"><h1>📑 فهرس الكتاب</h1>', '<ol class="toc">']
    out.append('<li><a href="front.xhtml">المقدمة والهداء وخطة الكتاب</a></li>')
    for pt in parts:
        chs = [c for c in book["chapters"] if c["part"] == pt["num"]]
        out.append(f'<li><a href="part{pt["num"]:02d}.xhtml">الجزء {pt["ordinal"]}: '
                   f'{plain(pt["title"])}</a>')
        out.append("<ol>")
        for c in chs:
            out.append(f'<li><a href="part{pt["num"]:02d}.xhtml#ch{c["num"]}">'
                       f'الفصل {c["num"]}: {plain(c["title"])}</a></li>')
        out.append("</ol></li>")
    out.append('<li><a href="appendices.xhtml">الملاحق (ملحق أ — ط)</a><ol>')
    for i, a in enumerate(book["appendices"]):
        out.append(f'<li><a href="appendices.xhtml#apx-{i}">{plain(a["title"])}</a></li>')
    out.append("</ol></li>")
    out.append('<li><a href="colophon.xhtml">خاتمة الكتاب</a></li>')
    out.append("</ol></section>")
    return xhtml("فهرس الكتاب", "\n".join(out))


def build_appendices(book):
    out = ['<section class="part" id="apx"><div class="part-inner">'
           '<span class="pbadge">الملاحق</span>'
           '<h1>الملاحق التسعة</h1>'
           '<div class="plight">ملحق أ — ط · قوالب وChecklists ومراجع عملية</div>'
           '</div></section>']
    for i, a in enumerate(book["appendices"]):
        ap = dict(a)
        ap["ord"] = i
        out.append(strip_marks(H.appendices_html(ap, _resolver)))
    return xhtml("الملاحق", "\n".join(out))


def build_colophon(book):
    meta = book["meta"]
    st = meta["stats"]
    body = f"""<section class="colophon"><div class="rule"></div>
<p>انتهى كتاب «{H.inline(meta['title'])}» — {H.inline(meta['edition'])} · {meta['publisher']}</p>
<p>~{st['words']:,} كلمة · {st['code_blocks']} كتلة كود · 33 مخططًا توضيحيًا · صيغ إخراج MD/HTML/PDF/DOCX/EPUB</p>
</section>"""
    return xhtml("خاتمة", body)


def build_cover(book):
    meta = book["meta"]
    body = ('<div class="coverpage cover-root">'
            '<img src="../images/cover.png" alt="غلاف الكتاب"/></div>')
    return xhtml(meta["title"], body, css="../style.css")


# ------------------------------------------------------------------ nav + ncx
def build_nav(book):
    parts = book["part_titles"]
    nav = ['<nav epub:type="toc" id="toc"><h1>فهرس الكتاب</h1><ol>']
    nav.append('<li><a href="text/front.xhtml">المقدمة والهداء</a>'
               '<ol>'
               '<li><a href="text/front.xhtml#fr-ded">الهداء</a></li>'
               '<li><a href="text/front.xhtml#fr-preface">المقدمة</a></li>'
               '<li><a href="text/front.xhtml#fr-summary">خطة الكتاب</a></li>'
               '</ol></li>')
    nav.append('<li><a href="text/toc.xhtml">الفهرس</a></li>')
    for pt in parts:
        chs = [c for c in book["chapters"] if c["part"] == pt["num"]]
        nav.append(f'<li><a href="text/part{pt["num"]:02d}.xhtml">الجزء {pt["ordinal"]}: '
                   f'{plain(pt["title"])}</a><ol>')
        for c in chs:
            nav.append(f'<li><a href="text/part{pt["num"]:02d}.xhtml#ch{c["num"]}">'
                       f'الفصل {c["num"]}: {plain(c["title"])}</a></li>')
        nav.append("</ol></li>")
    nav.append('<li><a href="text/appendices.xhtml">الملاحق</a><ol>')
    for i, a in enumerate(book["appendices"]):
        nav.append(f'<li><a href="text/appendices.xhtml#apx-{i}">{plain(a["title"])}</a></li>')
    nav.append("</ol></li>")
    nav.append('<li><a href="text/colophon.xhtml">خاتمة الكتاب</a></li>')
    nav.append("</ol></nav>")
    landmarks = ('<nav epub:type="landmarks" hidden="hidden"><h2>معالم</h2><ol>'
                 '<li><a epub:type="cover" href="text/cover.xhtml">الغلاف</a></li>'
                 '<li><a epub:type="bodymatter" href="text/front.xhtml">بداية الكتاب</a></li>'
                 '<li><a epub:type="toc" href="text/toc.xhtml">الفهرس</a></li>'
                 '</ol></nav>')
    body = "\n".join(nav) + "\n" + landmarks
    return xhtml("فهرس الكتاب", body, css="style.css")


def ncx_navpoints(book):
    """(label, src, level) مسطّح بمستويين."""
    items = [("المقدمة والهداء", "text/front.xhtml", 1),
             ("الفهرس", "text/toc.xhtml", 1)]
    for pt in book["part_titles"]:
        items.append((f"الجزء {pt['ordinal']}: {mathlite.render(pt['title'])}",
                      f"text/part{pt['num']:02d}.xhtml", 1))
        for c in [c for c in book["chapters"] if c["part"] == pt["num"]]:
            items.append((f"الفصل {c['num']}: {mathlite.render(c['title'])}",
                          f"text/part{pt['num']:02d}.xhtml#ch{c['num']}", 2))
    items.append(("الملاحق", "text/appendices.xhtml", 1))
    for i, a in enumerate(book["appendices"]):
        items.append((mathlite.render(a["title"]), f"text/appendices.xhtml#apx-{i}", 2))
    items.append(("خاتمة الكتاب", "text/colophon.xhtml", 1))
    return items


def build_ncx(book):
    meta = book["meta"]
    out = ['<?xml version="1.0" encoding="utf-8"?>',
           '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="ar">',
           '<head>',
           f'<meta name="dtb:uid" content="{BOOK_ID}"/>',
           '<meta name="dtb:depth" content="2"/>',
           '<meta name="dtb:totalPageCount" content="0"/>',
           '<meta name="dtb:maxPageNumber" content="0"/>',
           '</head>',
           f'<docTitle><text>{_html.escape(meta["title"])}</text></docTitle>',
           '<navMap>']
    order = 0
    for label, src, lvl in ncx_navpoints(book):
        order += 1
        out.append(f'<navPoint id="np{order}" playOrder="{order}">'
                   f'<navLabel><text>{_html.escape(label)}</text></navLabel>'
                   f'<content src="{src}"/></navPoint>')
    out.append("</navMap></ncx>")
    return "\n".join(out)


# ------------------------------------------------------------------ OPF
def build_opf(book, files):
    meta = book["meta"]
    st = meta["stats"]
    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="css" href="style.css" media-type="text/css"/>',
        '<item id="cover-xhtml" href="text/cover.xhtml" media-type="application/xhtml+xml"/>',
    ]
    spine = ['<itemref idref="cover-xhtml"/>']
    for name in files:  # text/*.xhtml أسماء ملفات
        fid = "x-" + Path(name).stem
        manifest.append(f'<item id="{fid}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{fid}"/>')
    manifest.append('<item id="cover-image" href="images/cover.png" media-type="image/png" properties="cover-image"/>')
    for key in FIGS:
        manifest.append(f'<item id="img-{key}" href="images/{key}.png" media-type="image/png"/>')
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="ar" prefix="dcterms: http://purl.org/dc/terms/">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{BOOK_ID}</dc:identifier>
    <dc:title>{_html.escape(meta['title'])}</dc:title>
    <dc:creator id="author">{_html.escape(meta.get('author_display', meta.get('author', '')))}</dc:creator>
    <dc:publisher>{_html.escape(meta['publisher'])}</dc:publisher>
    <dc:language>ar</dc:language>
    <dc:date>2026</dc:date>
    <dc:description>{_html.escape(meta['subtitle'])}</dc:description>
    <dc:subject>LLM</dc:subject>
    <dc:subject>Transformers</dc:subject>
    <dc:subject>RAG</dc:subject>
    <dc:subject>Agents</dc:subject>
    <dc:subject>الذكاء الاصطناعي</dc:subject>
    <meta property="dcterms:modified">{MODIFIED}</meta>
    <meta name="cover" content="cover-image"/>
  </metadata>
  <manifest>
    {chr(10).join('    ' + m for m in manifest)}
  </manifest>
  <spine toc="ncx" page-progression-direction="rtl">
    {chr(10).join('    ' + s for s in spine)}
  </spine>
</package>"""
    return opf


CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


# ------------------------------------------------------------------ CSS
EPUB_CSS = """
@charset "utf-8";
html, body { margin: 0; padding: 0; }
body {
  direction: rtl; text-align: right;
  font-family: "Amiri", "Noto Naskh Arabic", "Traditional Arabic", "Segoe UI", serif;
  font-size: 1em; line-height: 1.95; color: #1E293B;
  padding: 0 0.6em;
}
h1, h2, h3, h4 { line-height: 1.4; page-break-after: avoid; }
h1 { color: #31267B; font-size: 1.6em; margin: 0.8em 0 0.4em; }
h2 { color: #372C92; font-size: 1.35em; margin: 1em 0 0.45em; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.2em; }
h3 { color: #3D338F; font-size: 1.15em; margin: 0.9em 0 0.35em; }
h4 { color: #4B448E; font-size: 1.02em; margin: 0.8em 0 0.3em; }
p { margin: 0 0 0.65em; text-align: justify; }
em { font-style: italic; }
strong { font-weight: bold; color: #0F172A; }
a { color: #4338CA; text-decoration: none; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 0.86em; background: #F1F5F9; border-radius: 3px; padding: 0 0.2em; direction: ltr; unicode-bidi: embed; }
pre { direction: ltr; text-align: left; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; }
section.part { page-break-before: always; text-align: center; padding: 10% 1em 0; }
.part-inner { padding: 1.2em 0.4em; }
section.part h1 { font-size: 1.9em; border: 0; color: #31267B; }
.pghost { display: none; }
.pbadge { display: inline-block; background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE;
  border-radius: 999px; padding: 0.15em 0.9em; font-size: 0.85em; margin-bottom: 0.6em; }
.plight { color: #64748B; font-size: 0.9em; margin: 0.3em 0 1.2em; }
.pmeta { color: #94A3B8; font-size: 0.82em; margin: 0.2em 0 0.6em; }
.partfig { margin: 1.4em 0 0; }
.chead { page-break-inside: avoid; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.45em;
  margin-bottom: 0.7em; }
.chead h2 { border: 0; margin: 0.1em 0 0; }
.cnum-ghost { display: none; }
.cbody { padding: 0; }
.ch { page-break-before: always; margin-top: 0.4em; }
.chnum { display: block; color: #0891B2; font-size: 0.8em; letter-spacing: 0.02em; margin: 0.4em 0 0.1em; }
.dochead { text-align: center; padding: 2em 0 1em; }
.dochead .eyebrow { display: inline-block; color: #4338CA; background: #EEF2FF; border: 1px solid #C7D2FE;
  border-radius: 999px; padding: 0.1em 0.9em; font-size: 0.82em; margin-bottom: 0.8em; }
.dochead h1 { font-size: 2em; margin-bottom: 0.2em; }
.dochead .sub { color: #475569; font-size: 1.15em; margin-bottom: 0.6em; }
.dochead .author { color: #3730A3; font-weight: bold; font-size: 1.02em; margin-bottom: 0.4em; }
.dochead .meta { color: #64748B; font-size: 0.88em; }
.mth { direction: ltr; unicode-bidi: isolate; font-family: "Cambria Math", Cambria, serif;
  font-style: italic; color: #4338CA; }
.mthd { display: block; text-align: center; direction: ltr; unicode-bidi: isolate;
  font-family: "Cambria Math", Cambria, serif; font-style: italic; color: #4338CA;
  margin: 0.6em 0; }
.runmark { display: none; }
.tocpage { display: none; }
.coverpage { text-align: center; page-break-after: always; }
.coverpage img { max-width: 100%; height: auto; }
blockquote.qa { margin: 0.9em 0; padding: 0.6em 0.8em; border-right: 4px solid #818CF8;
  background: #F8FAFC; border-radius: 6px; font-size: 0.95em; }
blockquote.qa .k { margin-left: 0.35em; }
blockquote.qa.tip { border-color: #10B981; background: #ECFDF5; }
blockquote.qa.warn { border-color: #F59E0B; background: #FFFBEB; }
blockquote.qa.danger, blockquote.qa.mistake { border-color: #EF4444; background: #FEF2F2; }
blockquote.qa.note { border-color: #3B82F6; background: #EFF6FF; }
blockquote.qa.gold { border-color: #D97706; background: #FFFBEB; }
blockquote.qa.think { border-color: #7C3AED; background: #F5F3FF; }
blockquote.qa.compare { border-color: #6366F1; background: #EEF2FF; }
blockquote.qa.hook { border-color: #06B6D4; background: #ECFEFF; }
blockquote.qa.lesson { border-color: #14B8A6; background: #F0FDFA; }
blockquote.qa.rule { border-color: #64748B; background: #F8FAFC; }
blockquote.qa.best { border-color: #A855F7; background: #FAF5FF; }
.codebox { margin: 0.9em 0; border: 1px solid #E2E8F0; border-radius: 6px; background: #F8FAFC; }
.codebox .lang { display: inline-block; direction: ltr; font-family: "Consolas", monospace;
  font-size: 0.68em; color: #64748B; background: #EEF2FF; border-bottom-left-radius: 6px;
  padding: 0.1em 0.6em; }
.codebox pre { margin: 0; padding: 0.55em 0.7em; font-family: "Consolas", "Courier New", monospace;
  font-size: 0.78em; line-height: 1.55; color: #0F172A; }
.tblwrap { overflow-x: auto; margin: 0.9em 0; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 0.86em; }
th, td { border: 1px solid #CBD5E1; padding: 0.32em 0.45em; vertical-align: top; }
th { background: #EEF2FF; color: #312E81; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { color: #64748B; font-size: 0.82em; margin-top: 0.35em; }
.chfig { margin-top: 1.2em; padding-top: 0.8em; border-top: 1px dashed #CBD5E1; }
.chfig .kicker { display: block; color: #0891B2; font-size: 0.82em; margin-bottom: 0.5em; }
ul, ol { margin: 0.4em 0 0.8em; padding-right: 1.4em; padding-left: 0; }
li { margin-bottom: 0.25em; }
hr.hr { border: 0; border-top: 1px solid #E2E8F0; margin: 1em 0; }
.tocpage h1 { border: 0; }
ol.toc { list-style: none; padding-right: 0.2em; font-size: 0.95em; }
ol.toc ol { list-style: none; padding-right: 1em; font-size: 0.92em; }
ol.toc li { margin: 0.35em 0; }
.colophon { text-align: center; color: #64748B; margin-top: 3em; font-size: 0.9em; }
.colophon .rule { border-top: 2px solid #E2E8F0; width: 45%; margin: 0 auto 1.2em; }
.cover-root { padding: 0; page-break-after: auto; }
"""


def build_css():
    return EPUB_CSS + "\n" + H.PYG_CSS + "\n.codebox .highlight, .codebox .hll { background: transparent; }\n"


# ------------------------------------------------------------------ التجميع
def build():
    mathlite.last_failures.clear()
    book = load()
    meta = book["meta"]
    parts = book["part_titles"]

    docs = {}  # مسار داخل OEBPS -> نص xhtml
    docs["text/cover.xhtml"] = build_cover(book)
    docs["text/front.xhtml"] = build_front(book)
    docs["text/toc.xhtml"] = build_toc(book)
    for pt in parts:
        docs[f"text/part{pt['num']:02d}.xhtml"] = build_part(book, pt)
    docs["text/appendices.xhtml"] = build_appendices(book)
    docs["text/colophon.xhtml"] = build_colophon(book)
    docs["nav.xhtml"] = build_nav(book)
    docs["toc.ncx"] = build_ncx(book)
    docs["style.css"] = build_css()

    # ترتيب المتن (spine) — بدون الغلاف (يُضاف في opf)
    spine_files = ["text/front.xhtml", "text/toc.xhtml"] + \
                  [f"text/part{pt['num']:02d}.xhtml" for pt in parts] + \
                  ["text/appendices.xhtml", "text/colophon.xhtml"]
    docs["content.opf"] = build_opf(book, spine_files)

    # تحقق XML لكل مستندات xhtml/ncx/opf (عدا css)
    for name, content in docs.items():
        if name.endswith((".xhtml", ".ncx", ".opf")):
            try:
                ET.fromstring(content.encode("utf-8"))
            except ET.ParseError as e:
                raise SystemExit(f"XML invalid in {name}: {e}\n---\n{content[:400]}")
    # container.xml
    ET.fromstring(CONTAINER.encode("utf-8"))

    # الصور: نسخة القراءة إن وُجدت
    src_dir = DIST / "images_print" if (DIST / "images_print").exists() else ROOT / "figures" / "png"
    images = {}
    for key in FIGS:
        p = src_dir / f"{key}.png"
        if not p.exists():
            p = ROOT / "figures" / "png" / f"{key}.png"
        images[f"OEBPS/images/{key}.png"] = p.read_bytes()
    cover_p = src_dir / "cover.png"
    if not cover_p.exists():
        cover_p = DIST / "images" / "cover.png"
    images["OEBPS/images/cover.png"] = cover_p.read_bytes()

    # كتابة الأرشيف
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w") as z:
        zi = zipfile.ZipInfo("mimetype", date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, "application/epub+zip")
        zi = zipfile.ZipInfo("META-INF/container.xml", date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, CONTAINER.encode("utf-8"))
        for name in sorted(docs):
            arc = "OEBPS/" + name
            zi = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, docs[name].encode("utf-8"))
        for arc in sorted(images):
            zi = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, images[arc])

    print(f"epub: {OUT.stat().st_size:,} bytes · docs={len(docs)} · images={len(images)}")
    return OUT


if __name__ == "__main__":
    build()
