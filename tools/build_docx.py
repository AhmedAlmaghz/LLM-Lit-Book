# -*- coding: utf-8 -*-
"""
build_docx.py — إصدار Word (docx) بـ python-docx مع دعم RTL كامل:
- فقرات w:bidi + خطوط complex-script
- أنماط عناوين (Heading 1..4) + حقل TOC آلي مع تحديث تلقائي
- صناديق تنبيه مظللة لكل نوع، جداول bidiVisual، كود مصشّق بألوان Pygments
- صور المخططات مضمّنة بتعليقاتها، غلاف PDF/HTML نفسه
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from booklib import load, quote_label, FIGS
import mathlite
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.styles import get_style_by_name
from pygments.token import Token

DIST = Path(__file__).resolve().parent.parent / "dist"
AR_FONT = "Segoe UI"
MONO_FONT = "Consolas"
MATH_FONT = "Cambria Math"

style_theme = get_style_by_name("default")

# -------------------------------------------------- أدوات XML
def p_bidi(p, on=True):
    pPr = p._p.get_or_add_pPr()
    if pPr.find(qn("w:bidi")) is None:
        el = OxmlElement("w:bidi")
        el.set(qn("w:val"), "1" if on else "0")
        pPr.append(el)


def run_rtl(run, on=True):
    rPr = run._r.get_or_add_rPr()
    if rPr.find(qn("w:rtl")) is None:
        el = OxmlElement("w:rtl")
        el.set(qn("w:val"), "1" if on else "0")
        rPr.append(el)


def run_cs(run, font=AR_FONT, size=None):
    """خط complex script للعربية + حجم cس"""
    rPr = run._r.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.insert(0, rf)
    rf.set(qn("w:cs"), font)
    if size is not None:
        sz = rPr.find(qn("w:szCs"))
        if sz is None:
            sz = OxmlElement("w:szCs")
            rPr.append(sz)
        sz.set(qn("w:val"), str(int(size * 2)))


def shade_p(p, fill):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def shade_r(run, fill):
    rPr = run._r.get_or_add_rPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    rPr.append(shd)


def spacing(p, before=0, after=6, line=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing = line


# -------------------------------------------------- تجزئة النص المنسّق
_MATH_PAT = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)", re.DOTALL)
_CODE_PAT = re.compile(r"`([^`\n]+)`")
_BOLD_PAT = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITAL_PAT = re.compile(r"(?<![\w*])\*([^*\n]{1,120})\*(?![\w*])")
_PH_PAT = re.compile(r"\x00(\d+)\x00")


def tokenize(text):
    """يعيد قائمة (نص، نوع): plain/b/i/c/m/md — بنفس خوارزمية build_html.inline:
    حماية الكود السطري بعلامات مؤقتة، ثم رياضيات/غامق/مائل على السلسلة الواحدة،
    ثم استرجاع الكود — حتى تُقرأ أزواج **…** التي تتقاطع مع `كود` قراءةً صحيحة."""
    spans = []

    def _coderep(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    t = _CODE_PAT.sub(_coderep, text)
    out = []
    _split_struct(t, "plain", out, spans)
    return [(x, k) for x, k in out if x]


def _restore(s, spans):
    """يعيد نص الكود مكان علاماته المؤقتة (يُستعمل قبل تحويل الصيغ الرياضية)."""
    return _PH_PAT.sub(lambda m: spans[int(m.group(1))], s)


def _split_struct(text, kind, out, spans):
    """الغامق ثم المائل في النص الساده (كما في build_html)، ثم الرياضيات في أي مستوى."""
    if not text:
        return
    if kind == "plain":
        m = _BOLD_PAT.search(text)
        if m:
            _split_struct(text[:m.start()], kind, out, spans)
            _split_struct(m.group(1), "b", out, spans)
            _split_struct(text[m.end():], kind, out, spans)
            return
        m = _ITAL_PAT.search(text)
        if m:
            _split_struct(text[:m.start()], kind, out, spans)
            _split_struct(m.group(1), "i", out, spans)
            _split_struct(text[m.end():], kind, out, spans)
            return
    m = _MATH_PAT.search(text)
    if m:
        _split_struct(text[:m.start()], kind, out, spans)
        body = m.group(1) if m.group(1) is not None else m.group(2)
        conv = mathlite.convert(_restore(body, spans))
        out.append((conv, "md" if m.group(1) is not None else "m"))
        _split_struct(text[m.end():], kind, out, spans)
        return
    _emit(text, kind, out, spans)


def _emit(text, kind, out, spans):
    """يُخرج المقاطع مع تحويل علامات الكود المؤقتة إلى رموز c."""
    pos = 0
    for m in _PH_PAT.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], kind))
        out.append((spans[int(m.group(1))], "c"))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], kind))


def add_inline_runs(p, text, size=10.5, base_bold=False, base_color=None):
    for t, k in tokenize(text):
        r = p.add_run(t)
        r.font.size = Pt(size)
        r.bold = base_bold
        if base_color:
            r.font.color.rgb = base_color
        if k == "b":
            r.bold = True
            run_rtl(r, True)
        elif k == "i":
            r.italic = True
            run_rtl(r, True)
        elif k == "c":
            r.font.name = MONO_FONT
            run_cs(r, MONO_FONT, size - 1.0)
            run_rtl(r, False)
            shade_r(r, "F1F5F9")
        elif k in ("m", "md"):
            r.font.name = MATH_FONT
            run_cs(r, MATH_FONT, size - 0.5)
            r.italic = True
            run_rtl(r, False)
        else:
            run_rtl(r, True)
        if k in ("b", "i", "plain"):
            run_cs(r, AR_FONT, size)


def para(doc, text, size=10.5, align="justify", bold=False, color=None,
         before=0, after=7, line=1.35, indent=None, rtl=True):
    p = doc.add_paragraph()
    p_bidi(p, rtl)
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "right": WD_ALIGN_PARAGRAPH.RIGHT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER, "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    spacing(p, before, after, line)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
        p.paragraph_format.right_indent = Cm(indent)
    add_inline_runs(p, text, size, bold, color)
    return p


# -------------------------------------------------- عناوين
def heading(doc, text, level, size, color, before=16, after=8):
    p = doc.add_paragraph(style=f"Heading {level}")
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(p, before, after, 1.25)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, size)
    run_rtl(r)
    return p


def figure(doc, key, label):
    img = B.fig_path(key)
    if not img.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacing(p, 10, 2)
    r = p.add_run()
    r.add_picture(str(img), width=Cm(15.2))
    c = para(doc, f"{label} — {FIGS[key]['title']}", size=9, align="center",
             color=RGBColor(0x64, 0x74, 0x8B), after=10, line=1.2)
    return p


# -------------------------------------------------- الكود الملوّن
_TOK_STYLE = {}


def token_style(ttype):
    """نمط Pygments مع تراجع إلى رمز الأب عند غيابه (مثل Punctuation.Indicator في YAML)."""
    if ttype in _TOK_STYLE:
        return _TOK_STYLE[ttype]
    st = {}
    t = ttype
    while t is not None:
        try:
            st = style_theme.style_for_token(t)
            break
        except KeyError:
            t = t.parent
    _TOK_STYLE[ttype] = st
    return st


def add_codeblock(doc, lang, code):
    code = code.rstrip("\n")
    try:
        lexer = get_lexer_by_name({"txt": "text", "prompt": "text"}.get(lang, lang))
    except Exception:
        lexer = TextLexer()
    lines = code.split("\n")
    for line in lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        spacing(p, 0, 0, 1.12)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.25)
        pf.right_indent = Cm(0.25)
        shade_p(p, "F8FAFC")
        if not line.strip():
            r = p.add_run("")
            r.font.size = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            continue
        for ttype, val in lexer.get_tokens(line):
            if val == "":
                continue
            st = token_style(ttype)
            r = p.add_run(val)
            r.font.size = Pt(8)
            r.font.name = MONO_FONT
            run_cs(r, MONO_FONT, 8)
            if st.get("color"):
                r.font.color.rgb = RGBColor.from_string(st["color"])
            if st.get("bold"):
                r.bold = True
            if st.get("italic"):
                r.italic = True
        # مسافة سفلية خفيفة بين الأسطر
        p.paragraph_format.space_after = Pt(0)
    # فاصل فراغ بعد الكتلة
    q = doc.add_paragraph()
    q.paragraph_format.space_before = Pt(0)
    q.paragraph_format.space_after = Pt(6)
    q.add_run("").font.size = Pt(3)


KIND_BG = {"tip": "EAF7F0", "warn": "FFF6E6", "danger": "FDECEC", "mistake": "FCEFF1",
           "note": "EAF4FD", "gold": "FFF9E3", "think": "F1EDFE", "compare": "EAEDFE",
           "hook": "E8F7FA", "lesson": "E6F7F4", "rule": "F1F4F8", "best": "F8EFFD"}


def add_quote(doc, kind, text):
    p = doc.add_paragraph()
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    spacing(p, 6, 8, 1.3)
    p.paragraph_format.left_indent = Cm(0.45)
    p.paragraph_format.right_indent = Cm(0.45)
    p.paragraph_format.keep_together = True
    shade_p(p, KIND_BG.get(kind, "F1F4F8"))
    txt = quote_label(text, kind or "note")
    for t, k in tokenize(txt):
        r = p.add_run(t)
        r.font.size = Pt(9.8)
        if k == "b":
            r.bold = True
        elif k == "i":
            r.italic = True
        elif k == "c":
            r.font.name = MONO_FONT
            run_rtl(r, False)
            shade_r(r, "E8EEF7")
        elif k in ("m", "md"):
            r.font.name = MATH_FONT
            r.italic = True
            run_rtl(r, False)
        if k in ("b", "i", "plain"):
            run_cs(r, AR_FONT, 9.8)


def add_table(doc, header, rows):
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.style = "Table Grid"
    # حصل على الجدول RTL (تدفق الأعمدة من اليمين)
    tblPr = t._tbl.tblPr
    bidi = OxmlElement("w:bidiVisual")
    tblPr.append(bidi)
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        p = cell.paragraphs[0]
        p_bidi(p)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        shade_p(p, "E8ECFA")
        r = p.add_run(re.sub(r"\*\*(.+?)\*\*", r"\1", h))
        r.bold = True
        r.font.size = Pt(9.6)
        r.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)
        run_cs(r, AR_FONT, 9.6)
        run_rtl(r)
    for i, row in enumerate(rows, 1):
        for j, cc in enumerate(row):
            cell = t.rows[i].cells[j]
            p = cell.paragraphs[0]
            p_bidi(p)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.space_after = Pt(2)
            for tt, kk in tokenize(cc):
                r = p.add_run(tt)
                r.font.size = Pt(9.4)
                if kk == "b":
                    r.bold = True
                elif kk == "i":
                    r.italic = True
                elif kk == "c":
                    r.font.name = MONO_FONT
                    run_rtl(r, False)
                elif kk in ("m", "md"):
                    r.font.name = MATH_FONT
                    r.italic = True
                    run_rtl(r, False)
                if kk in ("b", "i", "plain"):
                    run_cs(r, AR_FONT, 9.4)


def add_list(doc, items, ordered=False):
    for k, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p_bidi(p)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        spacing(p, 0, 4, 1.3)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.55)
        pf.right_indent = Cm(0.55)
        prefix = f"{k}. " if ordered else "•  "
        r = p.add_run(prefix)
        run_cs(r, AR_FONT, 10.5)
        r.font.size = Pt(10.5)
        add_inline_runs(p, it.replace("\n", " "))


# -------------------------------------------------- Build
def main():
    mathlite.last_failures.clear()
    book = load()
    meta = book["meta"]
    parts = book["part_titles"]
    st = meta["stats"]

    doc = Document()
    # إعدادات المستند: تحديث الحقول تلقائيًا (لإنشاء TOC عند فتحه في Word)
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)

    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)

    base = doc.styles["Normal"]
    base.font.name = AR_FONT
    base.font.size = Pt(10.5)
    rpr = base.element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:ascii"), AR_FONT)
    rf.set(qn("w:hAnsi"), AR_FONT)
    rf.set(qn("w:cs"), AR_FONT)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), "21")
    rpr.append(szcs)

    # ---------- الغلاف ----------
    cover = DIST / "images" / "cover.png"
    if cover.exists():
        pcv = doc.add_paragraph()
        pcv.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pcv.paragraph_format.space_before = Pt(0)
        pcv.paragraph_format.space_after = Pt(10)
        r = pcv.add_run()
        r.add_picture(str(cover), width=Cm(13.5))
    para(doc, f"**{meta['title']}**", size=24, align="center", after=4)
    para(doc, meta["subtitle"], size=13.5, align="center", color=RGBColor(0x64, 0x74, 0x8B), after=8)
    para(doc, f"{meta['edition']} · {meta['publisher']} · ⏱ ~{st['minutes']} دقيقة قراءة · 📄 ~{st['pages_est']} صفحة",
         size=10.5, align="center", after=14)

    # جدول بطاقة (2 صفوف × 3 أعمدة، بصريًا من اليمين)
    doc.add_paragraph()
    t = doc.add_table(rows=2, cols=3)
    t.style = "Table Grid"
    t._tbl.tblPr.append(OxmlElement("w:bidiVisual"))
    data = [("الأجزاء", "10"), ("الفصول", "48"), ("الملاحق", "9"),
            ("المخططات", "33"), ("الكلمات", f"~{st['words']:,}"), ("كتل الكود", f"{st['code_blocks']}+")]
    for r_i in range(2):
        cells = t.rows[r_i].cells
        for j in range(3):
            label, val = data[r_i * 3 + j]
            pp = cells[j].paragraphs[0]
            p_bidi(pp)
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            shade_p(pp, "F5F6FB")
            rr = pp.add_run(f"{label}: {val}")
            rr.font.size = Pt(10)
            run_cs(rr, AR_FONT, 10)
            run_rtl(rr)
    doc.add_paragraph()

    add_quote(doc, "note", "**الحزمة المرفقة**: مجلد `code/` فيه نسخة `.py` لكل فصل (جاهزة لـ Colab/Kaggle)؛ "
                           "ومجلد `code/notebooks/` فيه دفتر Jupyter لكل جزء؛ و`images/` كل المخططات بدقة عالية.")

    # ---------- TOC حقل ----------
    pcf = doc.add_paragraph()
    p_bidi(pcf)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
    rr = OxmlElement("w:r")
    rt = OxmlElement("w:t")
    rt.text = "فهرس آلي: انقر بزر الفأرة الأيمن ثم «تحديث الحقل» لتوليد الفهرس الكامل."
    rr.append(rt)
    fld.append(rr)
    pcf._p.append(fld)
    doc.add_paragraph().add_run("").font.size = Pt(6)

    # ---------- الهداء/المقدمة/الخطة ----------
    fr = book["front"]
    heading(doc, "الهداء", 2, 16, RGBColor(0x43, 0x36, 0x9B))
    for b in fr["dedication"][0:9]:
        if b["t"] == "p":
            para(doc, b["text"], align="center", size=11)
    heading(doc, "المقدمة", 1, 20, RGBColor(0x31, 0x26, 0x7B), before=20)
    for b in B.render_blocks(fr["preface"]):
        render_lb(doc, b)
    heading(doc, "خطة الكتاب", 1, 20, RGBColor(0x31, 0x26, 0x7B), before=20)
    for b in B.render_blocks(fr["summary"]):
        render_lb(doc, b)

    # ---------- الأجزاء ----------
    for pt in parts:
        doc.add_page_break()
        pn = pt["num"]
        heading(doc, f"الجزء {pt['ordinal']}: {pt['title']}", 1, 20, RGBColor(0x44, 0x36, 0xAF), before=26)
        para(doc, f"الفصول {[c['num'] for c in book['chapters'] if c['part'] == pn][0]}–"
                  f"{[c['num'] for c in book['chapters'] if c['part'] == pn][-1]}",
             size=10.5, align="right", color=RGBColor(0x64, 0x74, 0x8B), after=6)
        for key in B.figs_for_part(pn):
            figure(doc, key, f"شكل ج{pn}")
        for c in [c for c in book["chapters"] if c["part"] == pn]:
            heading(doc, f"الفصل {c['num']}: {c['title']}", 2, 15, RGBColor(0x37, 0x2C, 0x92), before=22)
            for b in B.render_blocks(c):
                render_lb(doc, b)
            ks = B.figs_for_chapter(c["num"])
            if ks:
                heading(doc, "المخططات التوضيحية" if len(ks) > 1 else "المخطط التوضيحي", 3, 12.5,
                        RGBColor(0x08, 0x91, 0xB2), before=14)
                for j, key in enumerate(ks, 1):
                    figure(doc, key, f"شكل {c['num']}.{j}")
                for key in ks:
                    m = FIGS[key]
                    if m["page_extra"]:
                        extra = "، ".join(str(x) for x in m["page_extra"])
                        para(doc, f"⤴⤴ يتفرع هذا المخطط عمليًا في الفصل {extra} من هذا الكتاب.",
                             size=9.5, align="center", color=RGBColor(0x64, 0x74, 0x8B), after=8)

    # ---------- الملاحق ----------
    doc.add_page_break()
    heading(doc, "الملاحق التسعة", 1, 20, RGBColor(0x44, 0x36, 0xAF), before=26)
    for a in book["appendices"]:
        heading(doc, a["title"], 2, 15, RGBColor(0x37, 0x2C, 0x92), before=20)
        for b in B.render_blocks(a):
            render_lb(doc, b)

    # ---------- الخاتمة ----------
    para(doc, "", size=9)
    para(doc, f"*انتهى كتاب «{meta['title']}» — {meta['edition']}*. أُنجز عبر خط أنابيب المنشور في `tools/`: "
              f"~{st['words']:,} كلمة، {st['code_blocks']} كتلة كود، 33 مخططًا توضيحيًا، و5 صيغ إخراج.",
         size=9.5, align="center", color=RGBColor(0x94, 0x9E, 0xAC))
    para(doc, "تم البناء باستخدام python-docx مع دعم كامل للاتجاه من اليمين لليسار.",
         size=8.5, align="center", color=RGBColor(0xB6, 0xBE, 0xC8), after=2)

    out = DIST / "llm-ds.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print("docx:", out.stat().st_size)


def render_lb(doc, b):
    """بلوك → مخرج docx مشترك بين المقدمة والفصول."""
    t = b["t"]
    if t == "h2":
        heading(doc, b["text"], 3, 12.5, RGBColor(0x3D, 0x33, 0x8F), before=14, after=6)
    elif t == "h3":
        heading(doc, b["text"], 4, 11, RGBColor(0x4B, 0x44, 0x8E), before=10, after=4)
    elif t == "p":
        para(doc, b["text"])
    elif t == "quote":
        add_quote(doc, b.get("kind", "note"), b["text"])
    elif t == "code":
        add_codeblock(doc, b.get("lang"), b["code"])
    elif t in ("ul", "ol"):
        add_list(doc, b["items"], ordered=(t == "ol"))
    elif t == "table":
        add_table(doc, b["header"], b["rows"])
        para(doc, "", size=6, after=4)
    elif t == "hr":
        pass
    elif t == "h1":
        heading(doc, b["text"], 2, 15, RGBColor(0x37, 0x2C, 0x92))
    else:
        raise ValueError(t)


if __name__ == "__main__":
    main()
