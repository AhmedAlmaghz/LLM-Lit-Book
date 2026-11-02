# مخرجات الكتاب — «المحوّلات والنماذج اللغوية الكبيرة»

هذا المجلد يحتوي النسخ النهائية الخمس للكتاب (48 فصلًا · 10 أجزاء · 9 ملاحق · 33 مخططًا توضيحيًا).
جميع الصيغ بالعربية ومن اليمين إلى اليسار، والصور مضمّنة داخل الملفات الكبيرة.

## الملفات

| الملف | الحجم | الصيغة | متى تستخدمه |
|---|---|---|---|
| `llm-ds.pdf` | ~37 MB | PDF/A4 — 626 صفحة | القراءة والطباعة والنسخ الرسمي |
| `llm-ds.html` | ~11.5 MB | HTML5 مستقل | القراءة على الشاشة بمتصفح (صور مضمّنة base64) |
| `book_print.html` | ~2.9 MB | HTML نسخة الطباعة | مصدر توليد الـPDF، صور خارجية `images_print/` |
| `llm-ds.epub` | ~8 MB | EPUB3 (RTL) | القارئات الإلكترونية والهواتف (Kindle/Apple Books/Calibre) |
| `llm-ds.docx` | ~7.2 MB | Word (RTL) | التحرير والمراجعة في Word |
| `llm-ds.md` | ~1.4 MB | Markdown | النسخ الاحتياطي النصي، Git، إعادة البناء |

## المجلدات

- **`images/`** — 33 مخططًا توضيحيًا + الغلاف بدقة كاملة (مراجعتها في `llm-ds.md` و`llm-ds.html`).
- **`images_print/`** — نسخة القراءة/الطباعة من الصور (بعرض 1560px) المستخدمة في `book_print.html` والـEPUB.
- **`../code/`** — 47 ملف بايثون خطيًا (فصل/ملف) + `requirements.txt` + `README.md`.
- **`../code/notebooks/`** — 10 دفاتر Jupyter (دفتر لكل جزء) جاهزة لـColab/Kaggle.
- **`../figures/`** — مصادر المخططات (SVG + PNG) وأدوات توليدها.

## ملاحظات الجودة

- **الاتجاه:** كل الصيغ RTL؛ Word يستخدم `w:bidi` و`w:rtl` وخطوط complex-script، وEPUB يستخدم `page-progression-direction="rtl"`.
- **الرياضيات:** حُوّلت الصيغ اللاتكسية (239 سطرية + 70 عرضية) إلى يونيكود مقروء (كسور، جذور، مؤشرات، رموز) — لا بقايا لاتكس خام في أي صيغة.
- **الكود:** 364 كتلة بايثون ملوّنة في Word بتلوين Pygments، وكلها تجتاز `py_compile`.
- **المخططات:** 33 صورة (فهرسها في مقدمة كل صيغة) موزّعة على الفصول، مع مخططات افتتاح الأجزاء (4 و9).
- **الفهرس:** Word فيه حقل TOC آلي (حدّثه بـF9 عند الفتح)، وEPUB فيه `nav.xhtml` + `toc.ncx`.

## إعادة البناء

```bash
python tools/build_md.py       # dist/llm-ds.md
python tools/build_html.py     # dist/llm-ds.html + dist/book_print.html
python tools/build_pdf.py      # dist/llm-ds.pdf (Edge headless، ~4 دقائق)
python tools/build_docx.py     # dist/llm-ds.docx
python tools/build_epub.py     # dist/llm-ds.epub
python tools/make_code.py      # code/*.py + code/README.md
python tools/make_notebooks.py # code/notebooks/*.ipynb
```

المصدر الموحّد للمحتوى: `build/book.json` (يُولَّد من `LLM-ds.md` عبر `extract_manuscript.py` ثم `clean_book.py`).
