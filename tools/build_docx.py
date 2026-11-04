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
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
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
    old = pPr.find(qn("w:shd"))
    if old is not None:
        pPr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    pPr.insert_element_before(shd, "w:tabs", "w:suppressAutoHyphens", "w:kinsoku",
                              "w:wordWrap", "w:overflowPunct", "w:topLinePunct",
                              "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
                              "w:adjustRightInd", "w:snapToGrid", "w:spacing",
                              "w:ind", "w:contextualSpacing", "w:mirrorIndents",
                              "w:suppressOverlap", "w:jc", "w:textDirection",
                              "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
                              "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
                              "w:pPrChange")


def shade_r(run, fill):
    rPr = run._r.get_or_add_rPr()
    old = rPr.find(qn("w:shd"))
    if old is not None:
        rPr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    rPr.insert_element_before(shd, "w:fitText", "w:vertAlign", "w:rtl", "w:cs",
                              "w:em", "w:lang", "w:eastAsianLayout",
                              "w:specVanish", "w:oMath")


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn("w:shd"))
    if old is not None:
        tcPr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.insert_element_before(shd, "w:noWrap", "w:tcMar", "w:textDirection",
                               "w:tcFitText", "w:vAlign", "w:hideMark",
                               "w:headers", "w:cellIns", "w:cellDel",
                               "w:cellMerge", "w:tcPrChange")


def spacing(p, before=0, after=6, line=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing = line


def p_border(p, edge="bottom", color="4F46E5", sz=12, space=2):
    """حدّ فقرة (bottom/top) بسماكة sz (1/8 pt)."""
    pPr = p._p.get_or_add_pPr()
    old = pPr.find(qn("w:pBdr"))
    if old is not None:
        pPr.remove(old)
    bdr = OxmlElement("w:pBdr")
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), str(space))
    el.set(qn("w:color"), color)
    bdr.append(el)
    pPr.insert_element_before(bdr, "w:shd", "w:tabs", "w:suppressAutoHyphens",
                              "w:kinsoku", "w:wordWrap", "w:overflowPunct",
                              "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
                              "w:bidi", "w:adjustRightInd", "w:snapToGrid",
                              "w:spacing", "w:ind", "w:contextualSpacing",
                              "w:mirrorIndents", "w:suppressOverlap", "w:jc",
                              "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId",
                              "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")


def cell_margins(cell, top=40, bottom=40, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn("w:tcMar"))
    if old is not None:
        tcPr.remove(old)
    mar = OxmlElement("w:tcMar")
    for name, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.insert_element_before(mar, "w:textDirection", "w:tcFitText", "w:vAlign",
                               "w:hideMark", "w:headers", "w:cellIns", "w:cellDel",
                               "w:cellMerge", "w:tcPrChange")


def table_borders(table, edge="bottom", color="D8DEE9", sz=6):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{name}")
        if name == edge:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), color)
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar",
                                "w:tblLook", "w:tblCaption", "w:tblDescription",
                                "w:tblPrChange")


def table_bidi(table):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:bidiVisual"))
    if old is not None:
        tblPr.remove(old)
    el = OxmlElement("w:bidiVisual")
    tblPr.insert_element_before(el, "w:tblStyleRowBandSize", "w:tblStyleColBandSize",
                                "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
                                "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar",
                                "w:tblLook", "w:tblCaption", "w:tblDescription",
                                "w:tblPrChange")


def field(paragraph, instr, placeholder="", size=8.5, color=None, bold=False):
    """حقل Word بسيط (PAGE / STYLEREF / NUMPAGES) مع نتيجة مبدئية منسّقة."""
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), AR_FONT)
    rPr.append(rf)
    if bold:
        rPr.append(OxmlElement("w:b"))
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2)))
    szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), str(int(size * 2)))
    if color:
        c = OxmlElement("w:color"); c.set(qn("w:val"), str(color)); rPr.append(c)
    rPr.append(sz)
    rPr.append(szcs)
    rtl = OxmlElement("w:rtl"); rtl.set(qn("w:val"), "1"); rPr.append(rtl)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = placeholder
    r.append(t)
    fld.append(r)
    paragraph._p.append(fld)


def clear_hf(container):
    """تفريغ رأس/تذييل مع الإبقاء على فقرة فارغة واحدة."""
    body = container._element
    for child in list(body):
        body.remove(child)
    p = container.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    return p


def _hf_para(cell, align, bidi=True):
    p = cell.paragraphs[0]
    if bidi:
        p_bidi(p)
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    return p


def build_header(header, book_title, mode="chapter"):
    """رأس حديث: اسم الكتاب يمينًا + عنوان الفصل/القسم يسارًا، مع خط فاصل."""
    clear_hf(header)
    if mode == "none":
        return header
    t = header.add_table(rows=1, cols=2, width=Cm(17.0))
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_bidi(t)
    table_borders(t, edge="bottom", color="D8DEE9", sz=6)
    widths = (Cm(9.0), Cm(8.0))
    for i, w in enumerate(widths):
        t.columns[i].width = w
        t.rows[0].cells[i].width = w
        cell_margins(t.rows[0].cells[i], top=0, bottom=50)
        t.rows[0].cells[i].vertical_alignment = WD_ALIGN_VERTICAL.BOTTOM
    # يمين: اسم الكتاب
    p = _hf_para(t.rows[0].cells[0], WD_ALIGN_PARAGRAPH.RIGHT)
    r = p.add_run(book_title)
    r.font.size = Pt(8.2)
    r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 8.2)
    run_rtl(r)
    # يسار: القسم الحالي
    p = _hf_para(t.rows[0].cells[1], WD_ALIGN_PARAGRAPH.LEFT)
    if mode == "chapter":
        field(p, r' STYLEREF "Heading 2" \* MERGEFORMAT ', "عنوان الفصل", 8.4,
              RGBColor(0x37, 0x30, 0xA3), True)
    elif mode == "front":
        field(p, r' STYLEREF "Heading 1" \* MERGEFORMAT ', "القسم", 8.4,
              RGBColor(0x37, 0x30, 0xA3), True)
    elif mode.startswith("static:"):
        r = p.add_run(mode.split(":", 1)[1])
        r.bold = True
        r.font.size = Pt(8.4)
        r.font.color.rgb = RGBColor(0x37, 0x30, 0xA3)
        r.font.name = AR_FONT
        run_cs(r, AR_FONT, 8.4)
        run_rtl(r)
    tail = header.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = Pt(1)
    tail.add_run("").font.size = Pt(1)
    return header


def build_footer(footer, author):
    """تذييل: المؤلف يمينًا + رقم الصفحة يسارًا، مع خط علوي رفيع."""
    clear_hf(footer)
    t = footer.add_table(rows=1, cols=2, width=Cm(17.0))
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_bidi(t)
    table_borders(t, edge="top", color="D8DEE9", sz=6)
    widths = (Cm(10.5), Cm(6.5))
    for i, w in enumerate(widths):
        t.columns[i].width = w
        t.rows[0].cells[i].width = w
        cell_margins(t.rows[0].cells[i], top=50, bottom=0)
    p = _hf_para(t.rows[0].cells[0], WD_ALIGN_PARAGRAPH.RIGHT)
    r = p.add_run(author)
    r.font.size = Pt(8.0)
    r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 8.0)
    run_rtl(r)
    p = _hf_para(t.rows[0].cells[1], WD_ALIGN_PARAGRAPH.LEFT)
    r = p.add_run("صفحة ")
    r.font.size = Pt(8.0)
    r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 8.0)
    run_rtl(r)
    field(p, r' PAGE \* MERGEFORMAT ', "1", 8.8, RGBColor(0x43, 0x38, 0xCA), True)
    tail = footer.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = Pt(1)
    tail.add_run("").font.size = Pt(1)
    return footer


def page_start(section, start=0):
    """بدء ترقيم الصفحات من قيمة محددة (0 للغلاف)."""
    sectPr = section._sectPr
    old = sectPr.find(qn("w:pgNumType"))
    if old is not None:
        sectPr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:start"), str(start))
    sectPr.insert_element_before(el, "w:cols", "w:formProt", "w:vAlign",
                                 "w:noEndnote", "w:titlePg", "w:textDirection",
                                 "w:bidi", "w:rtlGutter", "w:docGrid",
                                 "w:printerSettings")


def setup_section(sec, mode="chapter", first_blank=False, number_start=None,
                  first_footer=True):
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.header_distance = Cm(1.0)
    sec.footer_distance = Cm(0.9)
    if number_start is not None:
        page_start(sec, number_start)
    else:
        old = sec._sectPr.find(qn("w:pgNumType"))
        if old is not None:
            sec._sectPr.remove(old)
    sec.different_first_page_header_footer = first_blank
    sec.header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False
    build_header(sec.header, "المحوّلات والنماذج اللغوية الكبيرة", mode)
    build_footer(sec.footer, "م. أحمد صالح المغز")
    if first_blank:
        sec.first_page_header.is_linked_to_previous = False
        sec.first_page_footer.is_linked_to_previous = False
        clear_hf(sec.first_page_header)
        if first_footer:
            build_footer(sec.first_page_footer, "م. أحمد صالح المغز")
        else:
            clear_hf(sec.first_page_footer)
    return sec


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
def heading(doc, text, level, size, color, before=16, after=8,
            page_break=False, border=False):
    """عنوان مع تحويل رياضيات داخلي (حتى لا يظهر LaTeX في Word)."""
    p = doc.add_paragraph(style=f"Heading {level}")
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(p, before, after, 1.25)
    p.paragraph_format.keep_with_next = True
    if page_break:
        p.paragraph_format.page_break_before = True
    for t, k in tokenize(text):
        r = p.add_run(t)
        r.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = color
        if k in ("c",):
            r.font.name = MONO_FONT
            run_cs(r, MONO_FONT, size - 1)
            run_rtl(r, False)
        elif k in ("m", "md"):
            r.font.name = MATH_FONT
            r.italic = True
            run_cs(r, MATH_FONT, size - 0.5)
            run_rtl(r, False)
        else:
            r.font.name = AR_FONT
            run_cs(r, AR_FONT, size)
            run_rtl(r)
    if border:
        p_border(p, "bottom", "4F46E5", 14, 4)
    return p


def opener(doc, kicker, title, num=None, page_break=True, level=2,
           size=19, color=RGBColor(0x1E, 0x1B, 0x4B)):
    """افتتاحية فصل/ملحق حديثة: شريط رمزي + عنوان كبير + خط سفلي."""
    kp = doc.add_paragraph()
    p_bidi(kp)
    kp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(kp, 0, 3, 1.05)
    kp.paragraph_format.keep_with_next = True
    if page_break:
        kp.paragraph_format.page_break_before = True
    shade_p(kp, "EEF2FF")
    r = kp.add_run(f"   {kicker}   ")
    r.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x43, 0x38, 0xCA)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 9.5)
    run_rtl(r)
    h = heading(doc, title, level, size, color, before=4, after=14,
                border=True)
    return h


def figure(doc, key, label, width=15.2):
    img = B.fig_path(key)
    if not img.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacing(p, 10, 2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run()
    r.add_picture(str(img), width=Cm(width))
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
    table_bidi(t)
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        p = cell.paragraphs[0]
        p_bidi(p)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        shade_p(p, "E8ECFA")
        add_inline_runs(p, h, size=9.6, base_bold=True,
                        base_color=RGBColor(0x31, 0x2E, 0x81))
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


def part_opener(doc, book, pt, chs, label="الجزء"):
    """صفحة افتتاح الجزء: لوحة داكنة عصرية + عنوان + مدى الفصول + مخطط الجزء."""
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.columns[0].width = Cm(17.0)
    cell = t.rows[0].cells[0]
    cell.width = Cm(17.0)
    table_borders(t, edge="none")
    shade_cell(cell, "1E1B4B")
    cell_margins(cell, top=300, bottom=300, left=260, right=260)

    p = cell.paragraphs[0]
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(p, 0, 8, 1.1)
    r = p.add_run(f"{label} {pt['ordinal']}")
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0xC7, 0xD2, 0xFE)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 11)
    run_rtl(r)

    p = cell.add_paragraph()
    p.style = doc.styles["Heading 1"]
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(p, 0, 10, 1.15)
    for tt, kk in tokenize(pt["title"]):
        r = p.add_run(tt)
        r.bold = True
        r.font.size = Pt(23)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if kk in ("m", "md"):
            r.font.name = MATH_FONT
            run_cs(r, MATH_FONT, 22)
            run_rtl(r, False)
        else:
            r.font.name = AR_FONT
            run_cs(r, AR_FONT, 23)
            run_rtl(r)

    p = cell.add_paragraph()
    p_bidi(p)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    spacing(p, 0, 12, 1.2)
    sub = (f"الفصول {chs[0]['num']}–{chs[-1]['num']}  ·  {len(chs)} فصولًا"
           if chs else "تسعة ملاحق عملية · قوالب وChecklists جاهزة")
    r = p.add_run(sub)
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0xC7, 0xD2, 0xFE)
    r.font.name = AR_FONT
    run_cs(r, AR_FONT, 10.5)
    run_rtl(r)
    p_border(p, "bottom", "818CF8", 12, 6)

    # مساحة تنفّس داخل اللوحة الداكنة
    sp = cell.add_paragraph()
    spacing(sp, 0, 0, 1)
    sp.add_run("").font.size = Pt(96)

    for key in B.figs_for_part(pt["num"]):
        figure(doc, key, f"شكل ج{pt['num']}", width=13.6)


# -------------------------------------------------- Build
def main():
    mathlite.last_failures.clear()
    book = load()
    meta = book["meta"]
    parts = book["part_titles"]
    st = meta["stats"]
    author = meta.get("author_display", meta.get("author", ""))

    doc = Document()
    cp = doc.core_properties
    cp.author = author
    cp.last_modified_by = author
    cp.title = meta["title"]
    cp.subject = meta["subtitle"]
    cp.language = "ar"
    cp.category = "LLM · Arabic"
    cp.keywords = "LLM, Transformers, MoE, Mamba, DPO, GRPO, RAG, Agents, Arabic"
    cp.comments = f"{meta['edition']} · {meta['publisher']}"

    # إعدادات المستند: تحديث الحقول تلقائيًا (لإنشاء TOC عند فتحه في Word)
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)

    # ---------- الأنماط العامة ----------
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

    # ---------- القسم 1: الغلاف + صفحة العنوان + الفهرس ----------
    sec1 = doc.sections[0]
    setup_section(sec1, mode="none", first_blank=True, number_start=0,
                  first_footer=False)

    # الغلاف (صفحة 0 بلا رأس/تذييل)
    cover = DIST / "images" / "cover.png"
    if cover.exists():
        pcv = doc.add_paragraph()
        pcv.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pcv.paragraph_format.space_before = Pt(0)
        pcv.paragraph_format.space_after = Pt(10)
        pcv.add_run().add_picture(str(cover), width=Cm(14.2))
    doc.add_page_break()

    # صفحة عنوان أنيقة
    para(doc, "كتاب عملي · إصدار 2026", size=10.5, align="center",
         color=RGBColor(0x64, 0x74, 0x8B), after=14)
    para(doc, f"**{meta['title']}**", size=27, align="center", after=6)
    para(doc, meta["subtitle"], size=14, align="center",
         color=RGBColor(0x47, 0x55, 0x69), after=10)
    p_rule = para(doc, "", size=2, align="center", after=14)
    p_border(p_rule, "bottom", "4F46E5", 18, 1)
    para(doc, f"**{author}**", size=15, align="center",
         color=RGBColor(0x37, 0x30, 0xA3), after=4)
    para(doc, f"{meta['edition']} · {meta['publisher']}", size=10.5, align="center",
         color=RGBColor(0x64, 0x74, 0x8B), after=4)
    para(doc, f"⏱ ~{st['minutes']} دقيقة قراءة · "
              f"{st['words']:,} كلمة", size=9.5, align="center",
         color=RGBColor(0x94, 0xA3, 0xB8), after=18)

    # بطاقة إحصاءات
    t = doc.add_table(rows=2, cols=3)
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_bidi(t)
    table_borders(t, edge="none")
    data = [("الأجزاء", "10"), ("الفصول", "48"), ("الملاحق", "9"),
            ("المخططات", "33"), ("الكلمات", f"~{st['words']:,}"), ("كتل الكود", f"{st['code_blocks']}+")]
    for r_i in range(2):
        for j in range(3):
            label, val = data[r_i * 3 + j]
            cell = t.rows[r_i].cells[j]
            cell.width = Cm(5.4)
            shade_cell(cell, "F5F6FB")
            cell_margins(cell, 120, 120)
            pp = cell.paragraphs[0]
            p_bidi(pp)
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            spacing(pp, 0, 0, 1.2)
            rr = pp.add_run(f"{label}: {val}")
            rr.bold = True
            rr.font.size = Pt(10)
            rr.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)
            run_cs(rr, AR_FONT, 10)
            run_rtl(rr)
    doc.add_page_break()

    # فهرس آلي
    p = para(doc, "**فهرس الكتاب**", size=21, align="right", after=10)
    p_border(p, "bottom", "4F46E5", 12, 4)
    pcf = doc.add_paragraph()
    p_bidi(pcf)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-2" \h \z \u')
    rr = OxmlElement("w:r")
    rp = OxmlElement("w:rPr")
    rf2 = OxmlElement("w:rFonts")
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf2.set(qn(a), AR_FONT)
    rp.append(rf2)
    col = OxmlElement("w:color"); col.set(qn("w:val"), "64748B"); rp.append(col)
    rt = OxmlElement("w:t")
    rt.text = "فهرس آلي: حدّث الحقل (F9) عند فتح الملف لتوليد الفهرس الكامل بأرقام الصفحات."
    rr.append(rp)
    rr.append(rt)
    fld.append(rr)
    pcf._p.append(fld)

    # ---------- القسم 2: الهداء والمقدمة وخطة الكتاب ----------
    front_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(front_sec, mode="front")
    fr = book["front"]
    heading(doc, "الهداء", 1, 21, RGBColor(0x31, 0x26, 0x7B), border=True,
            before=0, after=16)
    para(doc, "", size=4, after=10)
    for b in fr["dedication"][0:9]:
        if b["t"] == "p":
            para(doc, b["text"], align="center", size=12, line=1.6, after=10)
    heading(doc, "المقدمة", 1, 21, RGBColor(0x31, 0x26, 0x7B), page_break=True,
            border=True, before=0)
    for b in B.render_blocks(fr["preface"]):
        render_lb(doc, b)
    heading(doc, "خطة الكتاب", 1, 21, RGBColor(0x31, 0x26, 0x7B), page_break=True,
            border=True, before=0)
    for b in B.render_blocks(fr["summary"]):
        render_lb(doc, b)

    # ---------- الأجزاء ----------
    for pt in parts:
        pn = pt["num"]
        chs = [c for c in book["chapters"] if c["part"] == pn]
        part_sec = doc.add_section(WD_SECTION.NEW_PAGE)
        setup_section(part_sec, mode="chapter", first_blank=True)
        part_opener(doc, book, pt, chs, label="الجزء")

        for c in chs:
            opener(doc, f"الفصل {c['num']}  ·  الجزء {pt['ordinal']}",
                   c["title"], page_break=True)
            for b in B.render_blocks(c):
                render_lb(doc, b)
            ks = B.figs_for_chapter(c["num"])
            if ks:
                heading(doc, "المخططات التوضيحية" if len(ks) > 1 else "المخطط التوضيحي",
                        3, 13, RGBColor(0x08, 0x91, 0xB2), before=16)
                for j, key in enumerate(ks, 1):
                    figure(doc, key, f"شكل {c['num']}.{j}")
                for key in ks:
                    m = FIGS[key]
                    if m["page_extra"]:
                        extra = "، ".join(str(x) for x in m["page_extra"])
                        para(doc, f"⤴⤴ يتفرع هذا المخطط عمليًا في الفصل {extra} من هذا الكتاب.",
                             size=9.5, align="center", color=RGBColor(0x64, 0x74, 0x8B), after=8)

    # ---------- الملاحق ----------
    apx_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(apx_sec, mode="chapter", first_blank=True)
    apx_pt = {"num": 0, "ordinal": "", "title": "الملاحق التسعة"}
    part_opener(doc, book, apx_pt, [], label="الملاحق")
    for a in book["appendices"]:
        heading(doc, a["title"], 2, 17, RGBColor(0x1E, 0x1B, 0x4B),
                page_break=True, border=True, before=0, after=12)
        for b in B.render_blocks(a):
            render_lb(doc, b)

    # ---------- الخاتمة ----------
    heading(doc, "خاتمة الكتاب", 1, 21, RGBColor(0x31, 0x26, 0x7B),
            page_break=True, border=True, before=0, after=14)
    para(doc, f"*انتهى كتاب «{meta['title']}» — {meta['edition']}*. أُنجز عبر خط أنابيب المنشور في `tools/`: "
              f"~{st['words']:,} كلمة، {st['code_blocks']} كتلة كود، 33 مخططًا توضيحيًا، و5 صيغ إخراج.",
         size=10, align="center", color=RGBColor(0x64, 0x74, 0x8B), after=6)
    para(doc, f"{author} · {meta['publisher']} · 2026",
         size=9, align="center", color=RGBColor(0x94, 0xA3, 0xB8), after=2)

    out = DIST / "llm-ds.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print("docx:", out.stat().st_size,
          "sections:", len(doc.sections),
          "paras:", len(doc.paragraphs),
          "tables:", len(doc.tables))


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
