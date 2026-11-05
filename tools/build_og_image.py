# -*- coding: utf-8 -*-
"""build_og_image.py — يولّد صورة مشاركة اجتماعية (Open Graph) لكتاب LLM-DS:
landing/assets/og.jpg — 1200×630، بألوان الغلاف وهوية الكتاب.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COVER = ROOT / "dist" / "images" / "cover.png"
OUT = ROOT / "landing" / "assets" / "og.jpg"

W, H = 1200, 630
BG0 = (11, 16, 32)
BG1 = (28, 19, 61)


def shaped(s):
    return get_display(arabic_reshaper.reshape(s))


def rounded(im, radius):
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius, fill=255)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    return out


def vgrad(size, top, bottom):
    w, h = size
    base = Image.new("RGB", size, top)
    top_c = top
    bot_c = bottom
    for y in range(h):
        t = y / max(1, h - 1)
        color = tuple(int(top_c[i] + (bot_c[i] - top_c[i]) * t) for i in range(3))
        ImageDraw.Draw(base).line([(0, y), (w, y)], fill=color)
    return base


def main():
    img = vgrad((W, H), BG0, BG1).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # أضواء ولمعان
    glow1 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow1)
    gd.ellipse([760, -130, 1235, 320], fill=(124, 58, 237, 105))
    gd.ellipse([-180, 300, 350, 730], fill=(6, 182, 212, 70))
    gd.ellipse([430, 190, 890, 650], fill=(67, 56, 202, 85))
    img = Image.alpha_composite(img, glow1.filter(ImageFilter.GaussianBlur(85)))

    # خطوط شبكية هادئة
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 6), width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 6), width=1)

    # الغلاف بظل
    cover = Image.open(COVER).convert("RGB").resize((300, 450), Image.LANCZOS)
    cover = rounded(cover, 24)
    shadow = Image.new("RGBA", (cover.width + 90, cover.height + 90), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([30, 30, 30 + cover.width, 30 + cover.height], 24, fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    img.alpha_composite(shadow, (755, 92))
    img.alpha_composite(cover, (800, 88))
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([800, 88, 1100, 538], 24, outline=(255, 255, 255, 42), width=2)

    f_kicker = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 21)
    f_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 48)
    f_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 27)
    f_meta = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 22)
    f_author = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 25)

    x = 755  # الحد الأيمن للنص (يمين الغلاف)
    d.text((x, 110), shaped("كتاب عملي · طبعة أولى 2026"), font=f_kicker, fill=(103, 232, 249), anchor="ra")
    d.text((x, 155), shaped("المحوّلات والنماذج"), font=f_title, fill=(245, 246, 255), anchor="ra")
    d.text((x, 215), shaped("اللغوية الكبيرة"), font=f_title, fill=(245, 246, 255), anchor="ra")
    d.text((x, 285), shaped("من النظرية إلى بناء أنظمة ذكية"), font=f_sub, fill=(196, 181, 253), anchor="ra")
    d.text((x, 350), shaped("48 فصلًا · 10 أجزاء · 364 كتلة كود · 33 مخططًا"), font=f_meta, fill=(165, 243, 252), anchor="ra")
    d.text((x, 432), shaped("م. أحمد صالح المغز"), font=f_author, fill=(245, 246, 255), anchor="ra")
    d.text((x, 468), shaped("BookAI · طبعة أولى"), font=f_meta, fill=(138, 143, 184), anchor="ra")

    # شعار AI
    d.rounded_rectangle([70, 78, 122, 130], 14, fill=(67, 56, 202))
    d.text((96, 94), "AI", font=f_kicker, fill=(165, 243, 252), anchor="mm")
    d.text((132, 96), shaped("كتاب LLM بالعربي"), font=f_meta, fill=(199, 203, 239), anchor="lm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(OUT, "JPEG", quality=92, optimize=True, progressive=True)
    print(f"og -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
