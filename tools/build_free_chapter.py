# -*- coding: utf-8 -*-
"""
build_free_chapter.py — يولّد صفحة الفصل المجاني المستقلة:
landing/chapter-01.html

مصدر المحتوى build/book.json، وبنفس معالجات build_html.py (رياضيات يونيكود،
كود ملوّن، مخططات مع صور نسبية إلى dist/images).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B
from build_html import MAIN_CSS, PYG_CSS_INJECT, chapter_html

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "landing" / "chapter-01.html"


def main():
    book = B.load()
    chapter = next(c for c in book["chapters"] if c["num"] == 1)
    meta = book["meta"]
    author = meta.get("author_display", meta.get("author", ""))

    def resolver(key):
        src = ROOT / "dist" / "images" / f"{key}.png"
        dst = ROOT / "landing" / "assets" / f"{key}.png"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        # صفحة الهبوط تُنشر كجذر مستقل على GitHub Pages
        return f"assets/{key}.png"

    chapter_html_body = chapter_html(chapter, resolver=resolver)
    html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>الفصل الأول مجانًا — {meta['title']}</title>
<meta name="description" content="اقرأ الفصل الأول كاملًا من \'المحوّلات والنماذج اللغوية الكبيرة\': كيف تتعلم الآلة من الصفر، بالكود والمخططات.">
<meta name="robots" content="index, follow">
<link rel="icon" type="image/webp" href="assets/cover.webp">
<link rel="canonical" href="https://ahmedalmaghz.github.io/LLM-Lit-Book/chapter-01.html">
<style>
{MAIN_CSS}
body{{background:#F7F8FF}}
:root{{--brand-2:#7C3AED;--cyan:#06B6D4}}
.free-bar{{position:sticky;top:0;z-index:30;background:rgba(255,255,255,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}
.reading-progress{{position:fixed;z-index:40;bottom:0;left:0;height:3px;width:0%;background:linear-gradient(90deg,var(--brand),var(--cyan));transition:width .08s linear}}
.free-bar .in{{max-width:1120px;margin:0 auto;padding:10px 22px;display:flex;align-items:center;gap:12px}}
.free-bar .brand{{display:flex;align-items:center;gap:10px;text-decoration:none;color:#141033;font-weight:800}}
.free-bar .mark{{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#111827,#4338CA);display:flex;align-items:center;justify-content:center;color:#7DD3FC;font-family:Consolas,monospace;font-size:12px;letter-spacing:.5px}}
.free-bar a.back{{color:#4338CA;text-decoration:none;font-weight:700;font-size:13.5px}}
.free-bar .cta{{margin-inline-start:auto;display:inline-flex;align-items:center;gap:8px;background:#141033;color:#fff;text-decoration:none;border-radius:12px;padding:9px 14px;font-weight:800;font-size:13.5px;box-shadow:0 10px 24px rgba(17,24,39,.18)}}
.free-badge{{max-width:1000px;margin:18px auto 0;padding:0 22px;display:flex;align-items:center;gap:10px;color:#4338CA;font-size:13px;font-weight:800}}
.free-badge span{{background:#EEF2FF;border:1px solid #E0E7FF;border-radius:999px;padding:4px 12px}}
.page{{padding-top:8px}}
.ch{{box-shadow:0 18px 60px rgba(30,27,75,.09)}}
.free-cta{{max-width:1000px;margin:26px auto 46px;padding:0 22px}}
.free-cta .card{{background:radial-gradient(700px 350px at 12% 0%,#3B1D63 0%,transparent 60%),#141033;color:#fff;border-radius:24px;padding:28px 30px;box-shadow:0 24px 70px rgba(20,16,51,.35);display:grid;grid-template-columns:1.1fr .9fr;gap:24px;align-items:center}}
.free-cta h2{{font-size:26px;line-height:1.4;margin:0 0 10px}}
.free-cta p{{color:#C7CBFF;line-height:1.8;margin:0 0 10px}}
.free-cta a{{display:inline-flex;align-items:center;gap:8px;background:#7C3AED;color:#fff;text-decoration:none;border-radius:14px;padding:12px 18px;font-weight:800}}
.free-cta small{{color:#A5A7D9;display:block;margin-top:10px}}
.free-foot{{border-top:1px solid var(--line);padding:18px 22px;text-align:center;color:#64748B;font-size:12.5px}}
.free-chapter-toc{{max-width:1000px;margin:0 auto 18px;padding:0 22px;display:flex;flex-wrap:wrap;gap:8px}}
.free-chapter-toc a{{font-size:12px;color:#4338CA;background:#EEF2FF;border:1px solid #E0E7FF;border-radius:999px;padding:5px 12px;text-decoration:none}}
@media(max-width:760px){{.free-bar .brand small{{display:none}}.free-cta .card{{grid-template-columns:1fr;padding:22px}}.free-bar .cta{{display:none}}.free-chapter-toc{{display:none}}}}
</style>
{PYG_CSS_INJECT}
<script>document.addEventListener('DOMContentLoaded',function(){{var l=document.querySelector('.reading-progress');var links=[];document.querySelectorAll('.cbody h2').forEach(function(h,i){{var id='s'+(i+1);h.id=id;links.push('<a href="#'+id+'">'+h.textContent.trim()+'</a>');}});var toc=document.querySelector('.free-chapter-toc');if(toc)toc.innerHTML=links.join('');function onScroll(){{var d=document.documentElement;var max=d.scrollHeight-d.clientHeight;if(l)l.style.width=(max>0?Math.min(100,Math.max(0,d.scrollTop/max*100)):0)+'%';}}window.addEventListener('scroll',onScroll,{{passive:true}});onScroll();}});</script>
</head>
<body>
<header class="free-bar">
  <div class="in">
    <a class="brand" href="./index.html"><span class="mark">AI</span><span>المحوّلات <small style="display:block;color:#64748B;font-size:11px;font-weight:600">من النظرية إلى بناء أنظمة ذكية</small></span></a>
    <a class="back" href="./index.html">صفحة الكتاب</a>
    <a class="cta" href="./index.html#lead">احجز نسخة الإطلاق</a>
  </div>
</header>
<div class="free-badge"><span>الفصل الأول مجانًا</span> من {meta['title']} — {author}</div>
<div class="reading-progress" aria-hidden="true"></div>
<nav class="free-chapter-toc" aria-label="فهرس الفصل"></nav>
<main class="page">
{chapter_html_body}
</main>
<section class="free-cta" aria-label="احجز نسخة الإطلاق">
  <div class="card">
    <div>
      <h2>أعجبك الفصل؟ انتظر حتى تصل نسخة الإطلاق.</h2>
      <p>نفس أسلوب الكتاب: فهم عميق، كود يشغّل غدًا، وخارطة عربية للانتقال من خلية عصبية واحدة إلى نظام LLM حقيقي.</p>
      <a href="./index.html#lead">احجز اهتمامك بالنسخة</a>
      <small>PDF · EPUB · DOCX · HTML · Markdown — مع الكود والدفاتر.</small>
    </div>
    <figure style="margin:0;text-align:center"><img src="assets/cover.webp" alt="غلاف كتاب المحوّلات والنماذج اللغوية الكبيرة" style="width:180px;border-radius:16px;box-shadow:0 18px 48px rgba(0,0,0,.45)"></figure>
  </div>
</section>
<footer class="free-foot">{meta['title']} · {meta['edition']} · {author} · <a href="./index.html" style="color:#4338CA;text-decoration:none">صفحة الكتاب</a></footer>
</body>
</html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"free chapter -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
