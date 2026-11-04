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
  --bg:#EDF0F7; --paper:#FFFFFF; --ink:#1F2937; --muted:#64748B; --line:#E2E8F0;
  --brand:#4338CA; --brand2:#0891B2; --soft:#EEF2FF; --accent:#7C3AED;
  --codebg:#F8FAFC; --codeink:#0F172A;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:86px}
body{
  margin:0;color:var(--ink);
  background:radial-gradient(1200px 620px at 50% -12%,#E4E8FF 0%,var(--bg) 55%,#EEF1F7 100%);
  font-family:"Segoe UI","Noto Naskh Arabic","Noto Kufi Arabic",Tahoma,Arial,sans-serif;
  font-size:17.5px;line-height:1.95;text-rendering:optimizeLegibility;
  font-feature-settings:"kern" 1,"liga" 1;
}
/* شريط القراءة */
.sitebar{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.86);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(226,232,240,.9)}
.sitebar .in{max-width:1000px;margin:0 auto;padding:9px 22px;display:flex;align-items:center;gap:12px}
.brandmark{width:38px;height:38px;border-radius:11px;flex:0 0 auto;
  background:linear-gradient(135deg,#06B6D4,#4338CA 55%,#A855F7);color:#fff;
  font-weight:900;font-size:10.5px;display:flex;align-items:center;justify-content:center;
  letter-spacing:.4px;box-shadow:0 8px 18px rgba(67,56,202,.3)}
.bt{font-weight:800;font-size:14.5px;color:#171245;line-height:1.3}
.ba{font-size:11.5px;color:#64748B;line-height:1.3}
.sitebar nav{margin-inline-start:auto;display:flex;gap:6px}
.sitebar nav a{font-size:12.5px;font-weight:700;color:#3730A3;padding:7px 12px;
  border-radius:999px;background:#EEF2FF;border:1px solid #E0E7FF;text-decoration:none;white-space:nowrap}
.sitebar nav a:hover{background:#E0E7FF}
#pbar{height:3px;width:0;background:linear-gradient(90deg,#06B6D4,#4338CA,#A855F7);transition:width .12s linear}
@media(max-width:720px){.sitebar .ba{display:none}.sitebar nav a:last-child{display:none}}
.page{max-width:1000px;margin:0 auto;padding:26px 22px 60px}
h1,h2,h3,h4{line-height:1.5;font-weight:800;margin:0}
h1{font-size:33px;color:#111827}
h2{font-size:24px;color:#1E1B4B;margin:38px 0 10px}
h3{font-size:19px;color:#312E81;margin:22px 0 6px}
p{margin:10px 0}
a{color:var(--brand);text-decoration:none}
a:hover{text-decoration:underline}
strong{color:#0F172A}
/* الغلاف */
.coverpage{text-align:center;padding:26px 0 6px}
.coverpage img{width:min(340px,74vw);border-radius:18px;
  box-shadow:0 30px 80px rgba(30,27,75,.32);border:1px solid rgba(255,255,255,.6)}
.dochead{text-align:center;margin:30px auto 10px;max-width:760px}
.dochead .eyebrow{display:inline-block;font-size:12.5px;font-weight:800;letter-spacing:.5px;
  color:#4338CA;background:#EEF2FF;border:1px solid #C7D2FE;border-radius:999px;
  padding:5px 16px;margin-bottom:14px}
.dochead h1{font-size:clamp(32px,5vw,46px);font-weight:900;color:#141033;
  line-height:1.25;letter-spacing:-.5px}
.dochead .sub{color:#475569;font-size:clamp(17px,2.4vw,21px);margin-top:10px}
.dochead .author{color:#3730A3;font-weight:800;font-size:17px;margin-top:14px}
.dochead .meta{color:#6D28D9;font-size:13.5px;margin-top:8px;line-height:1.8}
/* الفهرس */
.toc{background:var(--paper);border:1px solid var(--line);border-radius:22px;
  padding:24px 28px;margin:30px 0;box-shadow:0 18px 50px rgba(30,27,75,.08)}
.toc h2{margin:0 0 14px;font-size:21px;color:var(--brand);border:none;padding:0}
.tocgrid{column-count:2;column-gap:30px}
@media(max-width:780px){.tocgrid{column-count:1}}
.tocpart{break-inside:avoid;margin:0 0 16px}
.tocptitle{font-weight:900;color:#312E81;font-size:14px;padding:4px 0;
  border-bottom:1px solid #E0E7FF;margin-bottom:4px;line-height:1.6}
.tocptitle .rng{color:#94A3B8;font-weight:600;font-size:11.5px;margin-inline-start:6px}
.tocrow{display:flex;align-items:baseline;gap:7px;padding:1.5px 0;font-size:13.5px;line-height:1.7}
.tocrow a{color:#334155;flex:0 1 auto;min-width:0}
.tocrow a:hover{color:#4338CA}
.tocdots{flex:1 1 auto;border-bottom:1px dotted #CBD5E1;transform:translateY(-4px);min-width:10px}
.tocpage{display:none}
.runmark{display:none}
/* الأجزاء */
.part{margin:56px 0 30px;border-radius:24px;overflow:hidden;
  box-shadow:0 30px 80px rgba(30,27,75,.28);break-inside:avoid}
.part-inner{position:relative;background:linear-gradient(140deg,#141033 0%,#312E81 45%,#6D28D9 100%);
  color:#fff;padding:70px 52px 54px;min-height:540px;display:flex;flex-direction:column;
  justify-content:center;overflow:hidden}
.part-inner::after{content:"";position:absolute;inset-inline-start:-90px;bottom:-120px;
  width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.28),transparent 65%)}
.pghost{position:absolute;top:-28px;inset-inline-start:24px;font-size:190px;font-weight:900;
  line-height:1;color:rgba(255,255,255,.07);letter-spacing:-10px;pointer-events:none}
.pbadge{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(199,210,254,.4);
  color:#E0E7FF;font-weight:800;font-size:13px;border-radius:999px;padding:7px 20px;
  backdrop-filter:blur(6px);width:fit-content}
.part h1{font-size:clamp(30px,4.4vw,46px);color:#fff;margin:18px 0 8px;max-width:24ch;
  line-height:1.3;position:relative;z-index:1}
.plight{color:#C7D2FE;font-size:16px;font-weight:600}
.pmeta{color:#A5B4FC;font-size:13px;margin-top:6px}
.partfig{background:#fff;border-radius:16px;padding:12px 12px 6px;margin:28px 0 0;
  position:relative;z-index:1}
.partfig figcaption{color:#475569}
/* الفصول */
.ch{background:var(--paper);border:1px solid var(--line);border-radius:22px;
  margin:30px 0;box-shadow:0 10px 30px rgba(30,27,75,.06);overflow:hidden}
.ch::before{content:"";display:block;height:5px;
  background:linear-gradient(90deg,#06B6D4,#4338CA 55%,#A855F7)}
.chead{display:flex;align-items:flex-start;gap:20px;padding:30px 34px 24px;
  background:linear-gradient(180deg,#FAFBFF 0%,#FFFFFF 100%);border-bottom:1px solid #EEF2F7}
.cnum-ghost{font-size:54px;font-weight:900;line-height:.85;color:#E0E7FF;
  letter-spacing:-3px;min-width:78px;text-align:center;padding-top:4px}
.chead-txt{flex:1;min-width:0}
.chnum{display:inline-block;background:var(--soft);color:var(--brand);border:1px solid #C7D2FE;
  font-size:12px;font-weight:800;border-radius:999px;padding:4px 14px;margin-bottom:7px;
  letter-spacing:.2px}
.ch h2{margin:0;font-size:clamp(24px,3.2vw,33px);color:#171245;line-height:1.35;
  border:none;padding:0}
.cbody{padding:22px 34px 34px}
/* تنبيهات */
.qa{border:none;border-radius:14px;padding:14px 18px;margin:16px 0;line-height:1.85}
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
.codebox{position:relative;background:var(--codebg);border:1px solid #E2E8F0;border-radius:14px;
  margin:16px 0;direction:ltr;text-align:left;overflow:hidden;box-shadow:0 6px 20px rgba(30,27,75,.05)}
.codebox .lang{position:absolute;top:7px;right:12px;font-size:10.5px;letter-spacing:1.5px;
  color:#94A3B8;text-transform:uppercase;font-weight:700}
.codebox pre{margin:0;padding:16px 18px;overflow-x:auto;font-family:Consolas,"Cascadia Code",monospace;
  font-size:13.5px;line-height:1.62;color:var(--codeink);white-space:pre-wrap;word-wrap:break-word}
.codebox code{font-family:inherit;background:none;color:inherit}
code.ic{direction:ltr;unicode-bidi:isolate;background:#F1F5F9;border:1px solid #E2E8F0;
  border-radius:6px;padding:1px 7px;font-family:Consolas,monospace;font-size:.88em;color:#0F172A}
.mth{direction:ltr;unicode-bidi:isolate;font-family:"Cambria Math",Cambria,serif;
  font-style:italic;background:#F5F3FF;border-radius:5px;padding:0 4px;color:#4338CA;white-space:nowrap}
.mthd{display:block;text-align:center;direction:ltr;unicode-bidi:isolate;
  font-family:"Cambria Math",Cambria,serif;font-style:italic;background:#F8F7FF;
  border:1px solid #E0E7FF;border-radius:10px;padding:8px 14px;margin:12px 0;color:#4338CA;
  font-size:1.02em}
/* الجداول */
.tblwrap{overflow-x:auto;margin:18px 0;border-radius:14px;border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;background:var(--paper);font-size:14.8px}
th{background:var(--soft);color:#312E81;font-weight:800;text-align:right;padding:10px 12px;
  border:1px solid var(--line)}
td{padding:9px 12px;border:1px solid var(--line);vertical-align:top;text-align:right}
tr:nth-child(even) td{background:#FAFBFE}
ul,ol{margin:10px 0;padding-right:26px;padding-left:0}
li{margin:5px 0}
/* الأشكال */
figure{margin:26px auto;text-align:center;max-width:100%}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:14px;
  box-shadow:0 10px 30px rgba(30,27,75,.10)}
figcaption{margin-top:10px;color:var(--muted);font-size:14.5px;font-weight:600}
.chfig{background:linear-gradient(180deg,#F8FAFF,#F2F5FF);border:1px dashed #C7D2FE;
  border-radius:16px;padding:16px 16px 8px;margin:24px 0}
.chfig h2{border:none;margin:4px 0 0;font-size:17px;color:var(--brand2);text-align:center}
.kicker{display:block;text-align:center;color:var(--muted);font-size:13px;margin-bottom:6px;
  letter-spacing:.4px}
.partfig{margin:28px 0 0;max-width:100%}
.partfig img{box-shadow:none;border:none;border-radius:8px}
.colophon{margin:70px 0 30px;text-align:center;color:var(--muted);font-size:14px}
.colophon .rule{width:160px;height:4px;border-radius:99px;margin:0 auto 16px;
  background:linear-gradient(90deg,#22D3EE,#818CF8,#C084FC)}
.toTop{position:fixed;bottom:22px;left:22px;background:#312E81;color:#fff;width:44px;height:44px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 8px 20px rgba(49,46,129,.35);z-index:9}
.hr{border:none;border-top:1px dashed var(--line);margin:22px 0}
@media(max-width:720px){
  .page{padding:18px 14px 44px}
  .chead{padding:22px 18px 18px;gap:12px}
  .cnum-ghost{font-size:40px;min-width:52px}
  .cbody{padding:18px 18px 26px}
  .part-inner{padding:46px 24px 36px;min-height:430px}
  .pghost{font-size:120px}
  .toc{padding:18px}
}
"""

PRINT_CSS = """
@page{size:A4;margin:16mm 15mm 18mm}
html{font-size:10.7pt;scroll-behavior:auto !important}
body{background:#fff;color:#1F2937;font-size:10.7pt;line-height:1.72;padding:0}
.sitebar{display:none !important}
.page{max-width:none;margin:0;padding:0}
.runmark{display:inline;font-size:.5pt;color:#FFFFFF;line-height:0;letter-spacing:0}
.tocpage{display:inline-block;flex:0 0 26pt;width:26pt;min-width:26pt;
  text-align:left;direction:ltr;font-size:8.6pt;color:#FFFFFF;font-weight:700;
  white-space:nowrap}
.coverpage{min-height:250mm;display:flex;align-items:center;justify-content:center;
  padding:0;margin:0;break-after:auto}
.coverpage img{width:120mm;max-width:120mm;border-radius:3mm;box-shadow:none;
  border:.5pt solid #E2E8F0}
.dochead{break-before:page;min-height:247mm;display:flex;flex-direction:column;
  justify-content:center;align-items:center;text-align:center;margin:0;padding:0;max-width:none}
.dochead .eyebrow{font-size:9pt;padding:1.5mm 5mm;margin-bottom:6mm}
.dochead h1{font-size:29pt;line-height:1.3;letter-spacing:0}
.dochead .sub{font-size:14pt;margin-top:5mm}
.dochead .author{font-size:13pt;margin-top:8mm}
.dochead .meta{font-size:9.2pt;color:#6D28D9;margin-top:4mm}
.toc{break-before:page;background:none;border:none;border-radius:0;box-shadow:none;
  padding:0;margin:0}
.toc h2{font-size:19pt;color:#31267B;margin:0 0 5mm;border-bottom:2pt solid #4F46E5;
  padding-bottom:2.5mm}
.tocgrid{column-count:2;column-gap:11mm}
.tocpart{break-inside:avoid;margin:0 0 3.4mm}
.tocptitle{font-size:9.4pt;padding:1mm 0;border-bottom:.5pt solid #E0E7FF;margin-bottom:0}
.tocptitle .rng{font-size:7.8pt}
.tocrow{font-size:8.55pt;padding:.5mm 0;line-height:1.42;gap:4pt}
.tocdots{transform:translateY(-2pt)}
.part{break-before:page;margin:0;border-radius:0;box-shadow:none;overflow:hidden}
.part-inner{background:linear-gradient(140deg,#141033 0%,#312E81 45%,#6D28D9 100%);
  min-height:260mm;padding:23mm 18mm;display:flex;flex-direction:column;
  justify-content:center;border-radius:0;overflow:hidden;
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
.pghost{font-size:150pt;top:-12mm;inset-inline-start:8mm;color:rgba(255,255,255,.07)}
.pbadge{font-size:10pt;padding:2mm 6mm}
.part h1{font-size:30pt;margin:6mm 0 3mm;max-width:23ch;line-height:1.28}
.plight{font-size:11pt}
.pmeta{font-size:9pt}
.partfig{background:#fff;padding:3mm;border-radius:3mm;margin-top:9mm}
.partfig figcaption{font-size:8.4pt;color:#475569}
.ch{break-before:page;background:#fff;border:none;border-radius:0;box-shadow:none;
  margin:0;padding:0;overflow:visible}
.ch::before{height:3pt;margin-bottom:7mm;
  background:linear-gradient(90deg,#06B6D4,#4338CA 55%,#A855F7)}
.chead{padding:0 0 5mm;border-bottom:1pt solid #E2E8F0;background:none;
  display:flex;align-items:flex-start;gap:5mm}
.cnum-ghost{font-size:42pt;line-height:.85;color:#E0E7FF;min-width:20mm;
  letter-spacing:-2pt;padding-top:1mm}
.chnum{font-size:8.6pt;padding:1mm 4mm;margin-bottom:2.5mm;border-radius:99pt}
.ch h2{font-size:21pt;line-height:1.3}
.cbody{padding:6mm 0 0}
h2,h3,h4{break-after:avoid}
h3{font-size:13.4pt;margin:6mm 0 2mm;color:#312E81}
h4{font-size:11.4pt;margin:5mm 0 1.5mm}
p{margin:3.2mm 0;orphans:3;widows:3}
figure{margin:6mm auto;break-inside:avoid}
figure img{max-height:178mm;border-radius:2mm;box-shadow:none;border:.5pt solid #E2E8F0}
figcaption{font-size:8.4pt;margin-top:2mm}
.chfig{border-radius:3mm;padding:3mm 3mm 1mm;margin:5mm 0;break-inside:avoid}
.chfig h2{font-size:12pt;margin:1mm 0 0}
.kicker{font-size:8pt;margin-bottom:1mm}
.qa{break-inside:avoid;border-radius:2.5mm;padding:3mm 4mm;margin:4mm 0;line-height:1.65}
.qa .k{font-size:8.6pt;margin-bottom:1mm}
.codebox{border-radius:2.5mm;break-inside:auto;margin:4mm 0;box-shadow:none}
.codebox pre{font-size:8.1pt;line-height:1.45;padding:3mm 3.5mm;
  overflow:visible;orphans:4;widows:4}
.codebox .lang{font-size:6.4pt;top:2mm;right:3mm}
code.ic{font-size:.82em;padding:0 1mm;border-radius:2pt}
.mth,.mthd{background:none;border:none;color:#312E81;padding:0}
.mthd{font-size:10.5pt;margin:3mm 0;padding:2mm 3mm;border:.5pt solid #E0E7FF;border-radius:2mm}
.tblwrap{margin:4mm 0;border-radius:2mm;overflow:visible}
table{font-size:8.15pt}
th{padding:1.8mm 2mm}
td{padding:1.5mm 2mm}
ul,ol{margin:3mm 0;padding-right:6mm}
li{margin:1mm 0}
.colophon{margin:20mm 0 0;font-size:9pt}
.colophon .rule{width:40mm;height:2.5pt;margin-bottom:5mm}
.toTop{display:none}
.hr{margin:5mm 0}
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
    # نعيد فكّ تهريب HTML داخل الرياضيات قبل التحويل، ثم نهرب الناتج مرة واحدة
    t = re.sub(r"\\\[(.*?)\\\]",
               lambda m: '<span class="mthd">' + _html.escape(
                   mathlite.convert(_html.unescape(m.group(1)))) + "</span>",
               t, flags=re.DOTALL)
    t = re.sub(r"\\\((.*?)\\\)",
               lambda m: '<span class="mth">' + _html.escape(
                   mathlite._convert(_html.unescape(m.group(1)), cleanup=True)) + "</span>",
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
def runmark(token):
    """يحفظ الموضع المنطقي للقسم (لا علامات نصية في الملف النهائي)."""
    return ""


def _chunk(kicker, title, num_ghost, body, section_id, token, cls="ch"):
    return "\n".join([
        f'<section class="{cls}" id="{section_id}">',
        runmark(token),
        f'<header class="chead">',
        f'<div class="cnum-ghost">{num_ghost}</div>',
        f'<div class="chead-txt"><span class="chnum">{kicker}</span>'
        f"<h2>{inline(title)}</h2></div>",
        "</header>",
        f'<div class="cbody">{body}</div>',
        "</section>",
    ])


def chapter_html(c, resolver=None, part_start=False):
    ks = B.figs_for_chapter(c["num"])
    body = []
    for b in B.render_blocks(c):
        if b["t"] == "hr":
            continue
        body.append(block_html(b, resolver))
    if ks:
        body.append('<div class="chfig"><span class="kicker">المخطط التوضيحي للفصل</span>')
        for j, key in enumerate(ks, 1):
            body.append(figure_html(key, f"شكل {c['num']}.{j}", resolver))
        for key in ks:
            m = FIGS[key]
            if m["page_extra"]:
                extra = "، ".join(str(x) for x in m["page_extra"])
                body.append(f'<p style="text-align:center;color:#64748B;font-size:13.5px">'
                            f"⤴️ يتكامل هذا المخطط مع المادة العملية في الفصل {extra}.</p>")
        body.append("</div>")
    return _chunk(f"الفصل {c['num']} · الجزء {c['part']}", c["title"],
                  f"{c['num']:02d}", "\n".join(body), f"ch{c['num']}",
                  f"C{c['num']}")


def appendices_html(a, resolver=None):
    body = []
    for b in B.render_blocks(a):
        if b["t"] == "hr":
            continue
        body.append(block_html(b, resolver))
    return _chunk("ملحق", a["title"], "أ", "\n".join(body),
                  f"apx-{a.get('ord', '')}", f"X{a.get('ord', '')}")


def front_html(fr, resolver=None):
    out = []
    # الهداء (أول 9 بلوكات الكلمة فقط؛ ما عدا hr الأخير)
    ded = [b for b in fr["dedication"][0:9]]
    body = "\n".join(f"<p>{inline(b['text'])}</p>" for b in ded if b["t"] == "p")
    out.append(_chunk("تأسيسي", "الهداء", "✦", body, "fr-ded", "F0"))
    for i, (fkey, title) in enumerate((("preface", "المقدمة"), ("summary", "خطة الكتاب")), 1):
        body = []
        for b in B.render_blocks(fr[fkey]):
            if b["t"] == "hr":
                continue
            body.append(block_html(b, resolver))
        out.append(_chunk("تأسيسي", title, "✦", "\n".join(body),
                          f"fr-{fkey}", f"F{i}"))
    return "\n".join(out)


# ------------------------------------------------------------------ التجمّع
def build(resolver=None, print_mode=False, title="المحوّلات والنماذج اللغوية الكبيرة"):
    mathlite.last_failures.clear()
    book = load()
    meta = book["meta"]
    parts = book["part_titles"]
    st = meta["stats"]

    css = MAIN_CSS + (PRINT_CSS if print_mode else "")

    # TOC — صفوف بخط منقّط وحقل رقم صفحة يُملأ بعد طباعة PDF
    def trow(href, label, token=None):
        # النقطة علامة موضع فقط (بيضاء في الطباعة) — تُستبدل برقم الصفحة بعد الطباعة
        return (f'<div class="tocrow"><a href="{href}">{label}</a>'
                f'<span class="tocdots"></span><span class="tocpage">.</span></div>')

    toc = ['<div class="toc" id="toc">', runmark("T"),
           "<h2>📑 فهرس الكتاب</h2>",
           '<div class="tocgrid">', '<div class="tocpart">',
           '<div class="tocptitle">أعمال تأسيسية</div>']
    toc.append(trow("#fr-ded", "الهداء", "F0"))
    toc.append(trow("#fr-preface", "المقدمة", "F1"))
    toc.append(trow("#fr-summary", "خطة الكتاب", "F2"))
    toc.append("</div>")
    for pt in parts:
        chs = [c for c in book["chapters"] if c["part"] == pt["num"]]
        toc.append('<div class="tocpart">')
        part_link = trow(f"#part{pt['num']}", f"الجزء {pt['ordinal']}: {pt['title']}", f"P{pt['num']}")
        toc.append('<div class="tocptitle">' + part_link +
                   f'<span class="rng">الفصول {chs[0]["num"]}–{chs[-1]["num"]}</span></div>')
        for c in chs:
            toc.append(trow(f"#ch{c['num']}", f"الفصل {c['num']}: {_html.escape(c['title'])}", f"C{c['num']}"))
        toc.append("</div>")
    toc.append('<div class="tocpart"><div class="tocptitle">الملاحق</div>')
    for i, a in enumerate(book["appendices"]):
        toc.append(trow(f"#apx-{i}", _html.escape(a["title"]), f"X{i}"))
    toc.append("</div></div></div>")
    toc_html = "\n".join(toc)

    body = []
    # الغلاف + صفحة العنوان
    cover_src = DIST / "images" / "cover.png"
    cover_html = ""
    if cover_src.exists():
        if resolver is None:
            cover_src_b64 = "data:image/png;base64," + _b64(cover_src)
        else:
            cover_src_b64 = resolver("cover")
        cover_html = (f'<div class="coverpage"><img src="{cover_src_b64}" alt="غلاف الكتاب"></div>')
    body.append(cover_html)
    body.append(f"""<div class="dochead">{runmark("I")}
  <div class="eyebrow">كتاب عملي · إصدار 2026</div>
  <h1>{inline(meta['title'])}</h1>
  <div class="sub">{inline(meta['subtitle'])}</div>
  <div class="author">{inline(meta.get('author_display', meta.get('author', '')))}</div>
  <div class="meta">{inline(meta['edition'])} · {meta['publisher']} ·
   ⏱ ~{st['minutes']} دقيقة قراءة ·
   48 فصلًا · 10 أجزاء · 9 ملاحق · 33 مخططًا</div>
</div>""")
    body.append(toc_html)
    body.append(front_html(book["front"], resolver))

    for pt in parts:
        pn = pt["num"]
        chs = [c for c in book["chapters"] if c["part"] == pn]
        fig = "".join(
            f'<div class="partfig">{figure_html(key, f"شكل ج{pn}", resolver)}</div>'
            for key in B.figs_for_part(pn))
        body.append(f"""<section class="part" id="part{pn}">
{runmark(f"P{pn}")}
<div class="part-inner">
  <div class="pghost">{pt['num']:02d}</div>
  <span class="pbadge">الجزء {pt['ordinal']}</span>
  <h1>{inline(pt['title'])}</h1>
  <div class="plight">الفصول {chs[0]["num"]}–{chs[-1]["num"]} · {len(chs)} فصولًا</div>
  <div class="pmeta">المحوّلات والنماذج اللغوية الكبيرة · من النظرية إلى بناء أنظمة ذكية</div>
  {fig}
</div>
</section>""")
        for c in chs:
            body.append(chapter_html(c, resolver))

    body.append(f"""<section class="part" id="apx">
{runmark("X")}
<div class="part-inner">
  <div class="pghost">أ</div>
  <span class="pbadge">الملاحق</span>
  <h1>الملاحق التسعة</h1>
  <div class="plight">ملحق أ — ط · قوالب وChecklists ومراجع عملية</div>
  <div class="pmeta">مرجع سريع يُستخدم أثناء العمل لا بعد قراءة الكتاب فقط</div>
</div>
</section>""")
    for i, a in enumerate(book["appendices"]):
        ap = dict(a)
        ap["ord"] = i
        body.append(appendices_html(ap, resolver))

    body.append(f"""<div class="colophon">{runmark("Z")}<div class="rule"></div>
انتهى كتاب «{inline(meta['title'])}» — {inline(meta['edition'])} · {meta['publisher']}
<br>{inline(meta.get('author_display', meta.get('author', '')))} · ~{st['words']:,} كلمة ·
{st['code_blocks']} كتلة كود · 33 مخططًا توضيحيًا · صيغ إخراج MD/HTML/PDF/DOCX/EPUB
</div><a class="toTop" href="#top" title="إلى الأعلى">↑</a>""")

    sitebar = f"""<header class="sitebar"><div class="in">
  <div class="brandmark">LLM</div>
  <div><div class="bt">{inline(meta['title'])}</div>
  <div class="ba">{inline(meta.get('author_display', meta.get('author', '')))} · {meta['publisher']}</div></div>
  <nav><a href="#toc">الفهرس</a><a href="#part1">ابدأ القراءة</a></nav>
</div><div id="pbar"></div></header>"""

    html_doc = f"""<!doctype html>
<html dir="rtl" lang="ar" id="top">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="author" content="{_html.escape(meta.get('author', ''))}">
<meta name="description" content="{_html.escape(meta['subtitle'])}">
<title>{_html.escape(meta['title'])} — {_html.escape(meta['subtitle'])}</title>
<style>{css}</style>
{PYG_CSS_INJECT}
</head>
<body>
{sitebar}
<div class="page">
{''.join(body)}
</div>
<script>
(function(){{var b=document.getElementById('pbar');if(!b)return;
addEventListener('scroll',function(){{var h=document.documentElement;
var m=h.scrollHeight-h.clientHeight;b.style.width=(m>0?(h.scrollTop/m*100):0)+'%';}},{{passive:true}});}})();
</script>
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
