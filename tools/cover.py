# -*- coding: utf-8 -*-
"""
cover.py — غلاف الكتاب: صفحة HTML/CSS آمنة العرض (تشكيل عربي عبر المتصفح) → لقطة Edge.
الخريطة: art/cover.html → art/cover.png (2048×3072 via scale=2) → dist/images/cover.png
لا حاجة لأي خدمة صور خارجية: تصميم SVG/CSS محلي بالكامل بنفس لغة المخططات.
"""
import sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from svg2png import launch_screenshot

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "art"
DIST_IMG = ROOT / "dist" / "images"

TITLE = "المحوّلات والنماذج اللغوية الكبيرة"
SUBTITLE = "من النظرية إلى بناء أنظمة ذكية"


def svg_net():
    """شبكة عصبية زخرفية: عقد + حواف + مصفوفة انتباه."""
    import random
    random.seed(42)
    edges = []
    # ثلاث طبقات من العقد لطبقات الشبكة
    layers = [(160, [(90, 300), (90, 430), (90, 560), (90, 690), (90, 820)]),
              (330, [(170, 330), (170, 470), (170, 610), (170, 750)]),
              (500, [(250, 380), (250, 540), (250, 700)])]
    for a in range(len(layers) - 1):
        for j, (nx1, y1) in enumerate(layers[a][1]):
            for i, (nx2, y2) in enumerate(layers[a + 1][1]):
                r = random.random()
                if r > .45:
                    op = .85 if r > .78 else .28
                    edges.append(f'<line x1="{nx1+300}" y1="{y1}" x2="{nx2+300}" y2="{y2}" stroke="#22D3EE" stroke-opacity="{op}" stroke-width="{2.2 if op>.7 else .9}" stroke-linecap="round"/>')
    circles = "".join(
        f'<circle cx="{x+300}" cy="{y}" r="{9 if li != 2 else 7}" fill="#0EA5E9" fill-opacity="{.5 if li==1 else .34}" stroke="#67E8F9" stroke-opacity=".55" stroke-width="1.4"/>'
        for li, (_, nl) in enumerate(layers) for x, y in nl)
    # مصفوفة انتباه (مثلثية) أسفل يسار التكوين
    cells = []
    n = 8
    cell = 30
    x0, y0 = 110, 860
    for r in range(n):
        for c in range(r + 1):
            import random as _r
            r2 = _r.random()
            if r2 < .26:
                continue
            op = .5 + .45 * r2
            col = "#A78BFA" if (r + c) % 3 == 0 else ("#22D3EE" if (r + c) % 3 == 1 else "#7DD3FC")
            cells.append(f'<rect x="{x0+c*cell}" y="{y0+r*cell}" width="{cell-5}" height="{cell-5}" rx="5" fill="{col}" fill-opacity="{op*.6:.2f}"/>')
    return "".join(edges) + circles + "".join(cells)


def build_html():
    net = svg_net()
    html = f"""<!doctype html>
<html dir="rtl" lang="ar"><head><meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0}}
  .cover{{
    width:1024px;height:1536px;position:relative;overflow:hidden;
    background:linear-gradient(165deg,#070B22 0%,#101743 34%,#1E1B5E 58%,#0B1030 100%);
    color:#fff;font-family:"Segoe UI","Noto Kufi Arabic",Tahoma,Arial,sans-serif;
  }}
  .glow{{position:absolute;border-radius:50%;filter:blur(90px);opacity:.5}}
  .g1{{width:520px;height:520px;background:#4F46E5;left:-140px;top:-120px}}
  .g2{{width:420px;height:420px;background:#0EA5E9;right:-100px;top:360px;opacity:.35}}
  .g3{{width:460px;height:460px;background:#7C3AED;left:180px;bottom:-160px;opacity:.4}}
  .grid{{position:absolute;inset:0;background:
    repeating-linear-gradient(0deg,rgba(148,163,184,.055) 0 1px,transparent 1px 96px),
    repeating-linear-gradient(90deg,rgba(148,163,184,.055) 0 1px,transparent 1px 96px)}}
  .net{{position:absolute;inset:0;opacity:.9}}
  .vign{{position:absolute;inset:0;background:linear-gradient(180deg,rgba(3,6,20,.15) 0%,rgba(3,6,20,0) 26%,rgba(3,6,20,0) 58%,rgba(3,6,20,.55) 100%)}}
  .band{{position:absolute;left:96px;right:96px;top:96px;bottom:96px;border:1.6px solid rgba(165,180,252,.28);border-radius:26px}}
  .head{{position:absolute;left:96px;right:96px;top:170px;text-align:center}}
  .mini{{display:inline-block;font-size:19px;letter-spacing:2px;color:#A5B4FC;border:1.2px solid rgba(165,180,252,.45);
    border-radius:999px;padding:9px 26px;margin-bottom:34px;background:rgba(99,102,241,.14)}}
  .title{{font-size:77px;font-weight:800;line-height:1.33;margin:0;
    background:linear-gradient(180deg,#FFFFFF 28%,#BFDBFE 100%);
    -webkit-background-clip:text;background-clip:text;color:transparent;
    text-shadow:0 6px 40px rgba(30,64,175,.35)}}
  .rule{{width:210px;height:5px;border-radius:99px;margin:36px auto 30px;
    background:linear-gradient(90deg,#22D3EE,#818CF8,#C084FC)}}
  .sub{{font-size:33px;font-weight:600;color:#C7D2FE;margin:0;opacity:.97}}
  .tokens{{position:absolute;left:0;right:0;top:660px;display:flex;justify-content:center;gap:14px}}
  .tok{{font-family:Consolas,monospace;direction:ltr;padding:10px 20px;border-radius:12px;font-size:21px;
    border:1px solid rgba(125,211,252,.4);background:rgba(14,165,233,.13);color:#BAE6FD;letter-spacing:1px}}
  .tokens-en{{direction:ltr}}
  .stats{{position:absolute;left:120px;right:120px;top:1216px;display:flex;justify-content:center;flex-wrap:wrap;gap:12px}}
  .st{{font-size:22px;color:#E0E7FF;background:rgba(99,102,241,.18);border:1px solid rgba(165,180,252,.35);
    border-radius:999px;padding:10px 24px;backdrop-filter:blur(4px)}}
  .foot{{position:absolute;left:120px;right:120px;bottom:126px;text-align:center}}
  .pub{{font-size:26px;font-weight:700;color:#F1F5F9;letter-spacing:.5px}}
  .edi{{font-size:22px;color:#A5B4FC;margin-top:10px}}
  .ln{{text-align:center;color:#64748B;font-size:15px;position:absolute;bottom:88px;left:0;right:0;letter-spacing:4px}}
</style></head><body>
<div class="cover">
  <div class="glow g1"></div><div class="glow g2"></div><div class="glow g3"></div>
  <div class="grid"></div>
  <svg class="net" width="1024" height="1536" viewBox="0 0 1024 1536" fill="none" xmlns="http://www.w3.org/2000/svg">
    {net}
  </svg>
  <div class="vign"></div>
  <div class="band"></div>
  <div class="head">
    <span class="mini">دليل هندسي شامل — من الصفر إلى الإنتاج</span>
    <h1 class="title">{TITLE}</h1>
    <div class="rule"></div>
    <p class="sub">{SUBTITLE}</p>
  </div>
  <div class="tokens tokens-en">
    <span class="tok">Transformer</span><span class="tok">Attention</span><span class="tok">MoE</span>
    <span class="tok">Flash-3</span><span class="tok">VLM</span><span class="tok">DPO/GRPO</span>
  </div>
  <div class="stats">
    <span class="st">48 فصلًا</span><span class="st">10 أجزاء</span><span class="st">9 ملاحق</span>
    <span class="st">33 مخططًا</span><span class="st">364+ كتلة كود</span>
  </div>
  <div class="foot">
    <div class="pub">BookAI</div>
    <div class="edi">الطبعة الأولى — 2026</div>
  </div>
  <div class="ln">ARABIC · ENGINEERING · 2026</div>
</div>
</body></html>"""
    ART.mkdir(exist_ok=True)
    (ART / "cover.html").write_text(html, encoding="utf-8")


def main():
    build_html()
    out = ART / "cover.png"
    launch_screenshot(ART / "cover.html", out, 1024, 1536, scale=2.0)
    DIST_IMG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, DIST_IMG / "cover.png")
    print("cover:", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
