/* إعدادات صفحة الهبوط — املأ الروابط عند نشرها أو عند توفر الشراء */
window.BOOKLANDING = {
  // الروابط العامة للمشروع
  githubUrl: "https://github.com/AhmedAlmaghz/LLM-Lit-Book",
  issuesUrl: "https://github.com/AhmedAlmaghz/LLM-Lit-Book/issues",
  discussionsUrl: "https://github.com/AhmedAlmaghz/LLM-Lit-Book/discussions",
  freeChapterUrl: "./chapter-01.html",

  // رابط الشراء العام (LemonSqueezy أو Stripe Payment Link أو أي رابط متجر)
  // اترك القيمة فارغة إن لم تكن جاهزة؛ زر الشراء سيوجّه إلى نموذج الحجز.
  purchaseUrl: "",
  purchaseLabel: "احجز نسختك",

  // رابط التحميل (اختياري): اترك فارغًا لإخفاء زر التحميل
  downloadUrl: "",

  // نقطة استقبال البريد (Formspree أو Mailchimp أو Apps Script أو أي خدمة POST/JSON)
  // مثال Formspree: "https://formspree.io/f/abcdwxyz"
  // اترك فارغة في هذه النسخة؛ سيتم تسجيل الطلبات محليًا في المتصفح فقط.
  formEndpoint: "",

  // اختياري: خصائص التحليلات (ضع النطاق عند التفعيل)
  plausibleDomain: "",
};
