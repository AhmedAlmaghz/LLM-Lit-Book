# -*- coding: utf-8 -*-
"""
build_html.py — المخرج HTML (ملف واحد كامل الذواتية) + المكتبة المشتركة لبناء EPUB/PDF.
- dist/llm-ds.html   (صور مضمّنة Base64 — ملف واحد مستقل)
- dist/book_print.html (نسخة الطباعة: صور نسبية + CSS طباعة، مدخل لبناء PDF عبر Edge)
يُستورد من build_epub.py لاستخدام المعالجات نفسها.
"""
import base64, html as _html, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from booklib import load, ensure_dist_images, quote_label, FIGS, DIST
import mathlite
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent

MAIN_CSS = """
:root{
  --bg:#F6F7FB; --paper:#FFFFFF; --ink:#1E293B; --muted:#64748B; --line:#E2E8F0;
  --brand:#4F46E5; --brand2:#0891B2; --soft:#EEF2FF; --accent:#7C3AED;
  --codebg:#F8FAFC; --codeink:#0F172A;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:"Segoe UI","Noto Kufi Arabic",Tahoma,Arial,sans-serif;
  font-size:17px;line-height:1.95;
}
.page{max-width:880px;margin:0 auto;padding:0 22px}
h1,h2,h3,h4{line-height:1.5;font-weight:800;margin:0}
h1{font-size:33px;color:#111827}
h2{font-size:24px;color:#1E1B4B;margin:38px 0 10px}
h3{font-size:19px;color:#312E81;margin:22px 0 6px}
p{margin:10px 0}
a{color:var(--brand);text-decoration:none}
a:hover{text-decoration:underline}
strong{color:#0F172A}
/* الغلاف */
.coverpage{text-align:center;padding:46px 0 8px}
.coverpage img{width:340px;max-width:80vw;border-radius:14px;box-shadow:0 24px 60px rgba(30,27,75,.28)}
.dochead{text-align:center;margin:26px 0 8px}
.dochead h1{font-size:40px;font-weight:900}
.dochead .sub{color:var(--muted);font-size:19px;margin-top:6px}
.dochead .meta{color:#6D28D9;font-size:14.5px;margin-top:10px}
/* الفهرس */
.toc{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:18px 22px;margin:26px 0}
.toc h2{margin:8px 0 12px;font-size:21px;color:var(--brand);border:none;padding:0}
.toc details{border-top:1px dashed var(--line);padding:10px 4px}
.toc details:first-of-type{border-top:none}
.toc summary{cursor:pointer;font-weight:800;color:#312E81;font-size:15.5px;list-style:revert}
.toc .chs{margin:8px 18px 4px 0;display:flex;flex-direction:column;gap:5px}
.toc .chs a{font-size:15px;color:#334155}
.toc .apx{margin:8px 18px 4px 0;display:flex;flex-direction:column;gap:5px}
/* الأجزاء */
.part{margin-top:64px}
.part .pbadge{display:inline-block;background:linear-gradient(90deg,var(--brand),var(--accent));
  color:#fff;font-weight:800;font-size:15px;border-radius:999px;padding:8px 22px;letter-spacing:.3px}
.part h1{font-size:30px;color:#27185E;margin:14px 0 4px}
.part .plight{color:var(--muted);font-size:15px;margin-bottom:10px}
/* الفصول */
.ch{background:var(--paper);border:1px solid var(--line);border-radius:16px;
  padding:26px 30px 30px;margin:26px 0;box-shadow:0 1px 2px rgba(15,23,42,.05)}
.ch .chnum{display:inline-block;background:var(--soft);color:var(--brand);border:1px solid #C7D2FE;
  font-size:13px;font-weight:700;border-radius:999px;padding:4px 16px;margin-bottom:8px}
.ch h2{margin-top:2px;border-bottom:2px solid var(--line);padding-bottom:12px}
/* تنبيهات */
.qa{border:none;border-radius:12px;padding:14px 18px 14px 18px;margin:16px 0;line-height:1.85}
.qa .k{font-weight:800;font-size:14px;display:block;margin-bottom:3px}
.qa.tip{background:#ECFDF5;border-right:4px solid #10B981}.qa.tip .k{color:#047857}
.qa.warn{background:#FFFBEB;border-right:4px solid #F59E0B}.qa.warn .k{color:#B45309}
.qa.danger{background:#FEF2F2;border-right:4px solid #EF4444}.qa.danger .k{color:#B91C1C}
.qa.mistake{background:#FFF1F2;border-right:4px solid #F43F5E}.qa.mistake .k{color:#BE123C}
.qa.note{background:#F0F9FF;border-right:4px solid #0EA5E9}.qa.note .k{color:#0369A1}
.qa.gold{background:#FEFCE8;border-right:4px solid #CA8A04}.qa.gold .k{color:#A16207}
.qa.think{background:#F5F3FF;border-right:4px solid #8B5CF6}.qa.think .k{color:#6D28D9}
.qa.compare{background:#EEF2FF;border-right:4px solid #6366F1}.qa.compare .k{color:#4338CA}
.qa.hook{background:#ECFEFF;border-right:4px solid #06B6D4}.qa.hook .k{color:#0E7490}
.qa.lesson{background:#F0FDFA;border-right:4px solid #14B8A6}.qa.lesson .k{color:#0F766E}
.qa.rule{background:#F8FAFC;border-right:4px solid #475569}.qa.rule .k{color:#334155}
.qa.best{background:#FDF4FF;border-right:4px solid #A855F7}.qa.best .k{color:#7E22CE}
/* الكود */
.codebox{position:relative;background:var(--codebg);border:1px solid #E2E8F0;border-radius:12px;margin:16px 0;
  direction:ltr;text-align:left;overflow:hidden}
.codebox .lang{position:absolute;top:7px;right:12px;font-size:10.5px;letter-spacing:1.5px;
  color:#94A3B8;text-transform:uppercase;font-weight:700}
.codebox pre{margin:0;padding:16px 18px;overflow-x:auto;font-family:Consolas,"Cascadia Code",monospace;
  font-size:13.5px;line-height:1.62;color:var(--codeink);white-space:pre-wrap;word-wrap:break-word}
.codebox code{font-family:inherit;background:none;color:inherit}
/* الأكواد السطرية */
code.ic{direction:ltr;unicode-bidi:isolate;background:#F1F5F9;border:1px solid #E2E8F0;
  border-radius:6px;padding:1px 7px;font-family:Consolas,monospace;font-size:.88em;color:#0F172A}
.mth{direction:ltr;unicode-bidi:isolate;font-family:"Cambria Math",Cambria,serif;
  font-style:italic;background:#F5F3FF;border-radius:5px;padding:0 4px;color:#4338CA;white-space:nowrap}
.mthd{display:block;text-align:center;direction:ltr;unicode-bidi:isolate;
  font-family:"Cambria Math",Cambria,serif;font-style:italic;background:#F8F7FF;
  border:1px solid #E0E7FF;border-radius:8px;padding:8px 14px;margin:10px 0;color:#4338CA;
  font-size:1.02em}
/* الجداول */
.tblwrap{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;background:var(--paper);font-size:14.8px}
th{background:var(--soft);color:#312E81;font-weight:800;text-align:right;padding:10px 12px;border:1px solid var(--line)}
td{padding:9px 12px;border:1px solid var(--line);vertical-align:top;text-align:right}
tr:nth-child(even) td{background:#FAFBFE}
/* قوائم */
ul,ol{margin:10px 0;padding-right:26px;padding-left:0}
li{margin:5px 0}
/* الأشكال */
figure{margin:26px auto;text-align:center;max-width:100%}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:12px;
  box-shadow:0 10px 30px rgba(30,27,75,.10)}
figcaption{margin-top:10px;color:var(--muted);font-size:14.5px;font-weight:600}
.chfig{background:linear-gradient(180deg,#F8FAFF,#F2F5FF);border:1px dashed #C7D2FE;border-radius:14px;
  padding:16px 16px 8px;margin:22px 0}
.chfig h2{border:none;margin:4px 0 0;font-size:17px;color:var(--brand2);text-align:center}
.kicker{display:block;text-align:center;color:var(--muted);font-size:13px;margin-bottom:6px;letter-spacing:.4px}
/* شرائح الجزء fig */
.partfig{margin:20px auto 0;max-width:820px}
.partfig img{border-radius:14px;box-shadow:0 16px 44px rgba(30,27,75,.16)}
/* خاتمة */
.colophon{margin:70px 0 30px;text-align:center;color:var(--muted);font-size:14px}
.colophon .rule{width:160px;height:4px;border-radius:99px;margin:0 auto 16px;
  background:linear-gradient(90deg,#22D3EE,#818CF8,#C084FC)}
.toTop{position:fixed;bottom:22px;left:22px;background:#312E81;color:#fff;width:44px;height:44px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 8px 20px rgba(49,46,129,.35);z-index:9}
.hr{border:none;border-top:1px dashed var(--line);margin:22px 0}
"""

PRINT_CSS = """
@page{size:A4;margin:15mm 14mm}
html{font-size:13.4px;scroll-behavior:auto !important}
body{background:#fff;font-size:13.4px;line-height:1.78}
.page{max-width:none;padding:0}
.coverpage{text-align:center;padding:0;margin-bottom:6mm}
.coverpage img{width:88mm;border-radius:6px;box-shadow:none}
.dochead h1{font-size:26px}
.ch,.toc{box-shadow:none;border-radius:6px;border-color:#D8DEE9}
.ch{padding:16px 18px 18px;margin:20px 0;break-inside:auto}
.part{break-before:page;margin-top:0}
h2{font-size:17.5px;margin:20px 0 8px}
.ch h2{font-size:19px}
h3{font-size:15px}
p{margin:7px 0}
figure{margin:14px auto;break-inside:avoid}
figcaption{font-size:11.5px}
.qa{break-inside:avoid;border-radius:8px;padding:9px 14px;margin:10px 0}
table{font-size:11.4px}th,td{padding:6px 8px}
.codebox{border-radius:8px;break-inside:avoid}
.codebox pre{font-size:10.6px;line-height:1.5}
.part{margin-top:26px}
.chflix{}
a{color:#1D4ED8}
"""

_KIND_ICON = {"tip": "💡", "warn": "⚠️", "note": "📌", "gold": "🏆", "think": "🤔",
              "compare": "⚖️", "mistake": "❌", "danger": "🛑", "hook": "❓",
              "lesson": "🧭", "rule": "📏", "best": "⭐"}


# ------------------------------------------------------------------ معالجات النص
def inline(text):
    """نص إنشائي: escape + bold/italic/inline-code + math."""
    t = text
    # inline code حماية
    spans = []
    def _coderep(m):
        spans.append(m.group(1))
        return f"\x00{len(spans)-1}\x00"
    t = re.sub(r"`([^`\n]+)`", _coderep, t)
    t = _html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]{1,120})\*(?![\w*])", r"<em>\1</em>", t)
    # math \[ \] display ثم \( \) سطرية
    t = re.sub(r"\\\[(.*?)\\\]",
               lambda m: '<span class="mthd">' + _html.escape(mathlite.convert(m.group(1))) + "</span>",
               t, flags=re.DOTALL)
    t = re.sub(r"\\\((.*?)\\\)",
               lambda m: '<span class="mth">' + _html.escape(mathlite._convert(m.group(1), cleanup=True)) + "</span>",
               t, flags=re.DOTALL)
    t = re.sub(r"\x00(\d+)\x00", lambda m: f'<code class="ic">{_html.escape(spans[int(m.group(1))])}</code>', t)
    return t


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode("ascii")


def img_attr(key, resolver=None):
    if resolver is not None:
        return resolver(key)
    return "data:image/png;base64," + _b64(B.fig_path(key))


def code_html(lang, code):
    if lang is None:
        lang = "text"
    try:
        lexer = get_lexer_by_name({"txt": "text", "prompt": "text"}.get(lang, lang))
    except Exception:
        lexer = TextLexer()
    body = highlight(code.rstrip("\n"), lexer, HtmlFormatter(noclasses=False, nowrap=True))
    return (f'<div class="codebox"><span class="lang">{_html.escape(lang)}</span>'
            f"<pre>{body}</pre></div>")


def block_html(b, resolver=None):
    """بلوك → HTML. يعيد سلسلة (قابلة للدمج)."""
    t = b["t"]
    if t == "h2":
        return f"<h3>{inline(b['text'])}</h3>"  # المقاطع داخل الفصل
    if t == "h3":
        return f"<h4>{inline(b['text'])}</h4>"
    if t == "p":
        return f"<p>{inline(b['text'])}</p>"
    if t == "quote":
        kind = b.get("kind", "note")
        icon = _KIND_ICON.get(kind, "📌")
        body = inline(quote_label(b["text"], kind))
        return f'<blockquote class="qa {kind}"><span class="k">{icon}</span>{body}</blockquote>'
    if t == "code":
        return code_html(b.get("lang"), b["code"])
    if t in ("ul", "ol"):
        tag = "ul" if t == "ul" else "ol"
        items = "".join(f"<li>{inline(it)}</li>" for it in b["items"])
        return f"<{tag}>{items}</{tag}>"
    if t == "table":
        esc = lambda s: inline(s)
        head = "".join(f"<th>{esc(h)}</th>" for h in b["header"])
        rows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in b["rows"])
        return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    if t == "hr":
        return '<hr class="hr">'
    if t == "h1":
        return f"<h2>{inline(b['text'])}</h2>"
    raise ValueError(t)


def figure_html(key, label, resolver=None):
    m = FIGS[key]
    return (f'<figure><img src="{img_attr(key, resolver)}" alt="{_html.escape(m["title"])}"/>'
            f"<figcaption>{label} — {inline(m['title'])}</figcaption></figure>")


# ------------------------------------------------------------------ أقسام كاملة
def chapter_html(c, resolver=None, part_start=False):
    ks = B.figs_for_chapter(c["num"])
    out = [f'<section class="ch" id="ch{c["num"]}">']
    out.append(f'<span class="chnum">الفصل {c["num"]} — الجزء {c["part"]}</span>')
    out.append(f"<h2>{inline(c['title'])}</h2>")
    blocks = B.render_blocks(c)
    for b in blocks:
        if b["t"] == "hr":
            continue
        out.append(block_html(b, resolver))
    if ks:
        out.append('<div class="chfig"><span class="kicker">المخطط التوضيحي للفصل</span>')
        for j, key in enumerate(ks, 1):
            out.append(figure_html(key, f"شكل {c['num']}.{j}", resolver))
        for key in ks:
            m = FIGS[key]
            if m["page_extra"]:
                extra = "، ".join(str(x) for x in m["page_extra"])
                out.append(f'<p style="text-align:center;color:#64748B;font-size:13.5px">'
                           f"⤴️ يتكامل هذا المخطط مع المادة العملية في الفصل {extra}.</p>")
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def appendices_html(a, resolver=None):
    out = [f'<section class="ch" id="apx-{a.get("ord", "")}">']
    out.append('<span class="chnum">ملحق</span>')
    out.append(f"<h2>{inline(a['title'])}</h2>")
    for b in B.render_blocks(a):
        if b["t"] == "hr":
            continue
        out.append(block_html(b, resolver))
    out.append("</section>")
    return "\n".join(out)


def front_html(fr, resolver=None):
    out = []
    # الهداء (أول 9 بلوكات الكلمة فقط؛ ما عدا hr الأخير)
    ded = [b for b in fr["dedication"][0:9]]
    out.append('<section class="ch" id="fr-ded"><span class="chnum">تأسيسي</span><h2>الهداء</h2>')
    for b in ded:
        if b["t"] == "p":
            out.append(f"<p>{inline(b['text'])}</p>")
    out.append("</section>")
    for fkey, title in (("preface", "المقدمة"), ("summary", "خطة الكتاب")):
        out.append(f'<section class="ch" id="fr-{fkey}"><span class="chnum">تأسيسي</span><h2>{title}</h2>')
        for b in B.render_blocks(fr[fkey]):
            if b["t"] == "hr":
                continue
            out.append(block_html(b, resolver))
        out.append("</section>")
    return "\n".join(out)


# ------------------------------------------------------------------ التجمّع
def build(resolver=None, print_mode=False, title="المحوّلات والنماذج اللغوية الكبيرة"):
    mathlite.last_failures.clear()
    book = load()
    meta = book["meta"]
    parts = book["part_titles"]
    st = meta["stats"]

    css = MAIN_CSS + (PRINT_CSS if print_mode else "")

    # TOC
    toc = ['<div class="toc">', "<h2>📑 فهرس الكتاب</h2>"]
    for pn, pt in enumerate(parts, 1):
        chs = [c for c in book["chapters"] if c["part"] == pt["num"]]
        toc.append(f"<details{' open' if pn == 1 else ''}><summary>الجزء {pt['ordinal']}: {pt['title']}</summary>")
        toc.append('<div class="chs">')
        toc.extend(f'<a href="#ch{c["num"]}">الفصل {c["num"]}: {_html.escape(c["title"])}</a>' for c in chs)
        toc.append("</div></details>")
    toc.append("<details><summary>الملاحق (ملحق أ — ط)</summary><div class=\"chs\">")
    toc.extend(f'<a href="#apx-{i}">{_html.escape(a["title"])}</a>' for i, a in enumerate(book["appendices"]))
    toc.append("</div></details></div>")
    toc_html = "\n".join(toc)

    body = []
    # الغلاف + الرأس
    cover_src = DIST / "images" / "cover.png"
    cover_html = ""
    if cover_src.exists():
        if resolver is None:
            cover_src_b64 = "data:image/png;base64," + _b64(cover_src)
        else:
            cover_src_b64 = resolver("cover")
        cover_html = (f'<div class="coverpage"><img src="{cover_src_b64}" alt="غلاف الكتاب"></div>')
    body.append(cover_html)
    body.append(f"""<div class="dochead">
  <h1>{inline(meta['title'])}</h1>
  <div class="sub">{inline(meta['subtitle'])}</div>
  <div class="meta">{inline(meta['edition'])} · {meta['publisher']} ·
   ⏱ ~{st['minutes']} دقيقة قراءة · 📄 ~{st['pages_est']} صفحة ·
   48 فصلًا · 10 أجزاء · 9 ملاحق · 33 مخططًا</div>
</div>""")
    body.append(toc_html)
    body.append(front_html(book["front"], resolver))

    for pt in parts:
        pn = pt["num"]
        chs = [c for c in book["chapters"] if c["part"] == pn]
        body.append(f'<section class="part" id="part{pn}">')
        body.append(f'<span class="pbadge">الجزء {pt["ordinal"]}</span>')
        body.append(f"<h1>{inline(pt['title'])}</h1>")
        body.append(f'<div class="plight">الفصول {chs[0]["num"]}–{chs[-1]["num"]}</div>')
        for key in B.figs_for_part(pn):
            body.append(f'<div class="chfig" style="margin-bottom:0"><span class="kicker">شريحة الجزء</span>')
            body.append(figure_html(key, f"شكل ج{pn}", resolver))
            body.append("</div>")
        body.append("</section>")
        for c in chs:
            body.append(chapter_html(c, resolver))

    body.append('<section class="part" id="apx"><span class="pbadge">الملاحق</span>'
                f"<h1>الملاحق التسعة</h1><div class=\"plight\">ملحق أ — ط</div></section>")
    for i, a in enumerate(book["appendices"]):
        ap = dict(a)
        ap["ord"] = i
        body.append(appendices_html(ap, resolver))

    body.append(f"""<div class="colophon"><div class="rule"></div>
انتهى كتاب «{inline(meta['title'])}» — {inline(meta['edition'])} · {meta['publisher']}
<br>~{st['words']:,} كلمة · {st['code_blocks']} كتلة كود · 33 مخططًا توضيحيًا · صيغ إخراج MD/HTML/PDF/DOCX/EPUB
</div><a class="toTop" href="#top" title="إلى الأعلى">↑</a>""")

    html_doc = f"""<!doctype html>
<html dir="rtl" lang="ar" id="top">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(meta['title'])} — {_html.escape(meta['subtitle'])}</title>
<style>{css}</style>
{PYG_CSS_INJECT}
</head>
<body>
<div class="page">
{''.join(body)}
</div>
</body></html>"""
    return html_doc


PYG_CSS = HtmlFormatter(style="default").get_style_defs(".codebox")
PYG_CSS_INJECT = f"<style>{PYG_CSS}\n.codebox .highlight, .codebox .hll{{background:transparent}}</style>"


def write_main():
    ensure_dist_images()
    out = DIST / "llm-ds.html"
    out.write_text(build(resolver=None, print_mode=False), encoding="utf-8")
    print("html:", out.stat().st_size)
    outp = DIST / "book_print.html"
    outp.write_text(build(resolver=lambda k: f"images_print/{k}.png", print_mode=True), encoding="utf-8")
    print("print html:", outp.stat().st_size)


if __name__ == "__main__":
    write_main()
