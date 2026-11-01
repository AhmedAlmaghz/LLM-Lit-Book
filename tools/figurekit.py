# -*- coding: utf-8 -*-
"""
figurekit — مكتبة SVG خفيفة لإنتاج مخططات تعليمية عربية/إنجليزية بطابع حديث.
- نظام تصميم موحّد: ألوان، تدرجات، بطاقات، أسهم، شارات، وسائل إيضاح.
- كل المخرجات SVG مستقلة (self-contained) بخطوط نظام قياسية.
"""
import re, math, html
from pathlib import Path

AR_RE = re.compile(r"[\u0621-\u064A]")
MONO_F  = "Consolas, 'Cascadia Mono', 'Liberation Mono', monospace"
SANS_F  = "'Segoe UI', 'Cairo', 'Tahoma', 'Noto Naskh Arabic', sans-serif"

C = dict(
    ink="#1E293B", mute="#64748B", line="#CBD5E1", line2="#94A3B8",
    bg="#FFFFFF", panel="#F8FAFC", panel2="#F1F5F9",
    indigo="#4F46E5", indigo2="#6366F1", indigoL="#EEF2FF",
    cyan="#0891B2", cyanL="#ECFEFF",
    violet="#7C3AED", violetL="#F5F3FF",
    emerald="#059669", emeraldL="#ECFDF5",
    amber="#D97706", amberL="#FFFBEB",
    rose="#E11D48", roseL="#FFF1F2",
    slate="#334155", white="#FFFFFF",
)

class Fig:
    def __init__(self, w=1040, h=585, pad=28):
        self.w, self.h, self.pad = w, h, pad
        self.parts = []
        self.defs = []
        self._ids = 0

    def arrow_def(self, cid, color):
        self.defs.append(
            f'<marker id="{cid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        )

    def _dir(self, t):
        return ' dir="rtl"' if AR_RE.search(t or "") else ""

    def text(self, x, y, t, size=15, fill=None, anchor="middle", weight=400, mono=False, italic=False, opacity=1, spacing=None):
        fill = fill or C["ink"]
        fam = MONO_F if mono else SANS_F
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        it = " font-style=\"italic\"" if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}" opacity="{opacity}"{sp}{it}{self._dir(t)} '
            f'dominant-baseline="middle">{html.escape(t)}</text>'
        )

    def rrect(self, x, y, w, h, fill=None, stroke=None, r=10, sw=1.4, dash=None, opacity=1, grad=None):
        f = f"url(#{grad})" if grad else (fill or "none")
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{f}" '
            f'stroke="{stroke or "none"}" stroke-width="{sw}"{extra} opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=None, sw=1.5, dash=None, marker=None, opacity=1):
        st = stroke or C["line2"]
        mk = f' marker-end="url(#{marker})"' if marker else ""
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{st}" stroke-width="{sw}"{extra}{mk} opacity="{opacity}"/>')

    def curve(self, d, stroke=None, sw=1.6, dash=None, marker=None, fill="none", opacity=1):
        st = stroke or C["indigo"]
        mk = f' marker-end="url(#{marker})"' if marker else ""
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{st}" stroke-width="{sw}"{extra}{mk} opacity="{opacity}"/>')

    def circle(self, x, y, r, fill=None, stroke=None, sw=1.5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or C["white"]}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def grad(self, gid, c1, c2, x2="1", y2="0", op1=1, op2=1):
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="0" x2="{x2}" y2="{y2}">'
            f'<stop offset="0" stop-color="{c1}" stop-opacity="{op1}"/>'
            f'<stop offset="1" stop-color="{c2}" stop-opacity="{op2}"/></linearGradient>'
        )

    # ——— مركبات ———
    def panel(self, x, y, w, h, title=None, tint="panel", stroke_t=0.9, grad=None, accent=None, r=14):
        fills = dict(panel=C["panel"], white=C["white"], indigoL=C["indigoL"], cyanL=C["cyanL"],
                     violetL=C["violetL"], emeraldL=C["emeraldL"], amberL=C["amberL"], roseL=C["roseL"], panel2=C["panel2"])
        self.rrect(x, y, w, h, fill=fills.get(tint, C["panel"]), stroke=C["line"], r=r, opacity=0.95, grad=grad)
        if accent:
            self.parts.append(f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2.5" ry="2.5" fill="{accent}"/>')
        if title:
            self.text(x + w / 2, y + 20, title, size=15.5, weight=700, anchor="middle")
        return y + 30

    def card(self, cx, cy, w, h, header, lines, tint="white", accent=C["indigo"], ha=0.62, steps=True, header_fill=None, sub=None):
        """بطاقة برأس ملوّن وأسطر شرح، الوسط (cx, cy)."""
        x, y = cx - w / 2, cy - h / 2
        self.rrect(x, y, w, h, C["white"], stroke=C["line"], r=12)
        hh = 30
        hf = accent
        self.parts.append(f'<path d="M{int(x)+10},{int(y)} H{int(x+w)-10} a10,10 0 0 1 10,10 v{hh-10} h-{int(w)} v-{int(hh-10)} a10,10 0 0 1 10,-10 z" fill="{hf}"/>')
        self.parts.append(f'<rect x="{x}" y="{y+hh}" width="{w}" height="6" fill="{hf}" opacity="0.18"/>')
        self.text(cx, y + hh / 2 + 1, header, size=15, fill="#fff", weight=700)
        step = ha if len(lines) == 1 else (h - hh - 12) / max(1, len(lines) - 1 if len(lines) > 1 else 1)
        if sub:
            self.text(cx, y + hh + 14, sub, size=12.5, fill=C["mute"])
        for k, ln in enumerate(lines):
            ly = y + hh + 12 + (0 if len(lines) == 1 else k * step)
            self.text(cx, ly, ln, size=13.2, fill=C["slate"])
        return (x, y, w, h)

    def flow_h(self, xs, y, labels, gap_w=86, h=52, colors=None, r=10, size=13.5, header=None):
        n = len(labels)
        colors = colors or ([C["indigo"], C["violet"], C["cyan"], C["emerald"], C["amber"], C["rose"]] * 3)
        bw = (self.w - 2 * self.pad) / n - 14
        xs = []
        for k, lab in enumerate(labels):
            x = self.pad + k * (bw + 14)
            xs.append(x + bw / 2)
            self.rrect(x, y - h / 2, bw, h, "white", stroke=colors[k % 6], r=r, sw=1.6)
            self.parts.append(f'<rect x="{x}" y="{y - h/2}" width="7" height="{h}" rx="3.5" ry="3.5" fill="{colors[k%6]}"/>')
            for i, seg in enumerate(lab.split("\n")):
                self.text(x + bw / 2 + 3, y - (8 if len(lab.split("\n")) == 2 and i == 0 else 0) + (9 if len(lab.split("\n")) == 2 and i == 1 else 0),
                          seg, size=size, weight=600 if i == 0 else 400, fill=C["slate"] if i else C["ink"])
        for k in range(n - 1):
            x1 = xs[k] + bw / 2; x2 = xs[k + 1] - bw / 2 - 3
            self.line(x1 + 4, y, x2, y, stroke=C["line2"], sw=1.7, marker="agr")
        if header:
            self.text(self.w / 2, y - h / 2 - 14, header, size=12.5, fill=C["mute"])
        return xs, y

    def chips(self, x, y, items, size=11.5):
        cx = x
        for t in items:
            wid = 10 + len(t) * (7.2 if not AR_RE.search(t) else 6.0)
            self.rrect(cx, y, wid, 22, C["panel2"], stroke=C["line"], r=11)
            self.text(cx + wid / 2, y + 11.4, t, size=size, fill=C["slate"])
            cx += wid + 8
        return cx

    def legend(self, x, y, items, gap=26, size=12.5):
        for k, (color, label) in enumerate(items):
            yy = y + k * gap
            self.parts.append(f'<rect x="{x}" y="{yy-8}" width="16" height="9" rx="3" ry="3" fill="{color}"/>')
            self.text(x + 24, yy - 3.4, label, size=size, fill=C["slate"], anchor="start")

    def hbar(self, x, y, wmax, frac, label, value_text, color=C["indigo"], h=13, size=12):
        self.text(x, y - 13, label, size=size, fill=C["slate"], anchor="start")
        self.rrect(x, y, wmax, h, C["panel2"], stroke=None, r=6)
        self.rrect(x, y, max(6, wmax * frac), h, color, stroke=None, r=6)
        self.text(x + wmax - 2, y - 13, value_text, size=size, fill=C["mute"], anchor="end")

    def badge(self, x, y, t, color=C["indigo"], fillL=None, h=24, size=11.5, weight=700, mono=False):
        wid = len(t) * (8.6 if mono else 7.4) + 18
        self.rrect(x, y, wid, h, fillL or C["indigoL"], stroke=color, r=h / 2, sw=1.1)
        self.text(x + wid / 2, y + h / 2 + 0.6, t, size=size, fill=color, weight=weight, mono=mono)
        return wid

    def section_note(self, t="مصدر: فصل الكتاب — مخطط توضيحي", size=10.5):
        self.text(self.w - 12, self.h - 12, t, size=size, fill=C["mute"], anchor="end", opacity=0.75)

    def arrows_bold(self, cid="agr", color=None):
        self.arrow_def(cid, color or C["slate"])

    def svg(self):
        defs = "\n".join(self.defs)
        body = "\n".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'font-family="{SANS_F}" direction="ltr">\n<defs>{defs}</defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{C["bg"]}"/>\n{body}\n</svg>')

def write_all(figures, outdir):
    """figures: dict name -> Fig ; يكتب ملفات SVG"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, f in figures.items():
        (outdir / f"{name}.svg").write_text(f.svg(), encoding="utf-8")
        print(f"  fg {name}.svg")
