# صفحة هبوط الكتاب — «المحوّلات والنماذج اللغوية الكبيرة»

صفحة إعلانية RTL إعلانية تسويقية فيروسية للكتاب، مبنية كـ **HTML/CSS/JS ثابت بدون أطر**، ومُعدة للنشر عبر GitHub Pages.

## ما داخل المجلد

```text
landing/
├── index.html          # الصفحة الرئيسية (RTL · عربية · بدون أطر)
├── chapter-01.html     # الفصل الأول كاملًا — مولّد من build/book.json
├── sitemap.xml         # خريطة الروابط
├── robots.txt          # السماح لمحركات البحث
├── site.webmanifest    # اسم وأيقونات للهاتف
├── favicon.svg         # شعار SVG أصلي للعلامة
└── assets/
    ├── site.css        # التصميم (RTL · Container · Responsive)
    ├── site.js         # التفاعل: النموذج، المشاركة، الاختبار، التمرير
    ├── config.js       # الروابط القابلة للتحرير (شراء · تحميل · بريد)
    ├── cover.webp      # غلاف آمن للويب (~40KB)
    ├── cover.jpg       # غلاف احتياطي للـ OG/Apple Touch
    ├── og.jpg          # صورة المشاركة الاجتماعية (1200×630)
    ├── f01_neuron.png  # صور الفصل المجاني
    ├── f02_backprop.png# صور الفصل المجاني
    └── transformer.png # مخطط من داخل الكتاب (شكل 8.1)
```

## التشغيل محليًا

```powershell
python -m http.server 8080 --directory landing
Start-Process "http://127.0.0.1:8080/"
```

أو استخدم أي خادم ثابت — الصفحة كاملة ذاتية، بلا قاعدة بيانات.

## النشر على GitHub Pages

المسار `landing/` يُنشر **كجذر الموقع** عبر GitHub Actions (الملف `.github/workflows/pages.yml`).

1. ادخل إلى **Settings → Pages** في المستودع `AhmedAlmaghz/LLM-Lit-Book`.
2. في **Build and deployment**، اختر **Source: GitHub Actions**.
3. ادفع فرع `main`؛ ستعمل مهمة `Deploy book landing page` تلقائيًا.
4. العنوان النهائي للمشروع:

   ```text
   https://ahmedalmaghz.github.io/LLM-Lit-Book/
   ```

> ملاحظة: لا حاجة لجعل `dist/` عامًا. الصفحة تعتمد على `landing/assets/` فقط.

## تفعيل جمع البريد فعليًا

النموذج ثابت ولا يرسل افتراضيًا؛ يحفظ الطلبات **محليًا في متصفح الزائر** حتى تربطه بخدمة:

1. أنشئ نموذجًا في **Formspree** (مجاني) أو Mailchimp أو Google Apps Script.
2. انسخ نقطة استقبال HTTP/POST (مثال Formspree: `https://formspree.io/f/xxxxxxx`).
3. ضعها في `landing/assets/config.js`:

   ```js
   formEndpoint: "https://formspree.io/f/xxxxxxx",
   ```

4. عاود النشر. تُرسل الحقول: الاسم، البريد، الاهتمامات، التاريخ، الصفحة.

### بدائل
- **Mailchimp**: استخدم رابط نموذج Mailchimp أو نقطة embed proxy.
- **Cloudflare Workers / Apps Script**: انشر دالة POST→Email/Sheet.

## روابط الشراء/التحميل

عدّل `landing/assets/config.js` قبل الإطلاق:

```js
purchaseUrl: "https://store.lemonsqueezy.com/buy/xxxxx", // أو رابط Stripe Payment Link
downloadUrl: "https://example.com/llm-ds.pdf",          // اختياري
formEndpoint: "https://formspree.io/f/xxxxxxx",          // جمع البريد
```

إن تُركت `purchaseUrl` فارغة، يوجّه زر الشراء إلى نموذج الحجز — لا ينكسر شيء.

## الفصل الأول المجاني

أُعيد توليده من المصدر الموحّد:

```powershell
python tools/build_free_chapter.py
```

ولتوليد صورة المشاركة الاجتماعية:

```powershell
python tools/build_og_image.py
```

## قائمة الجودة قبل النشر

- [ ] `python -m http.server 8080 --directory landing` ثم افتح الصفحة
- [ ] لا أخطاء Console، وصور الغلاف والفصل تُحمّل
- [ ] الروابط: الفصل المجاني، GitHub، الشراء (بعد تفعيلها) تعمل
- [ ] النموذج يرسل فعليًا بعد ضبط `formEndpoint`
- [ ] `sitemap.xml` و`robots.txt` يطابقان المجال النهائي
- [ ] المشاركة تظهر بشكل صحيح على WhatsApp / X / LinkedIn

## مبادئ المحتوى المعتمدة

- لا ادعاءات وهمية: لا مراجعات مزيفة، لا عدادات مؤقتة، لا وعود مالية.
- الأرقام كلها قابلة للتحقق من المستودع: 48 فصلًا، 10 أجزاء، 9 ملاحق، 364 كتلة Python، 33 مخططًا.
- «الفصل الأول مجاني» حقيقة مثبتة على الصفحة، لا حملة مؤقتة.
