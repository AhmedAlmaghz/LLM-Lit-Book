/*
 * site.js — تفاعل صفحة الهبوط:
 * - الروابط تُملأ من assets/config.js (شراء / تحميل / GitHub / الفصل المجاني)
 * - نموذج الحجز: تحقق + حفظ محلي + إرسال اختياري إلى formEndpoint
 * - مشاركة الرابط، ظهور تدريجي، حالة الشريط العلوي، سنة التذييل
 */
(function () {
  "use strict";

  const cfg = window.BOOKLANDING || {};
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function setHref(els, url, fallback) {
    els.forEach((el) => {
      if (!url) {
        el.setAttribute("href", fallback || "#lead");
        return;
      }
      el.setAttribute("href", url);
      if (/^https?:/i.test(url)) {
        el.setAttribute("target", "_blank");
        el.setAttribute("rel", "noopener noreferrer");
      }
    });
  }

  /* ---------- تهيئة الروابط من الإعدادات ---------- */
  function initLinks() {
    setHref($$("[data-purchase]"), cfg.purchaseUrl, "#lead");
    setHref($$("[data-download]"), cfg.downloadUrl, "#lead");
    setHref($$("[data-free-chapter]"), cfg.freeChapterUrl || "./chapter-01.html");
    setHref($$("[data-github]"), cfg.githubUrl || "https://github.com/AhmedAlmaghz/LLM-Lit-Book");
  }

  /* ---------- التحليلات الاختيارية ---------- */
  function initAnalytics() {
    if (!cfg.plausibleDomain) return;
    const s = document.createElement("script");
    s.defer = true;
    s.src = "https://plausible.io/js/script.js";
    s.setAttribute("data-domain", cfg.plausibleDomain);
    document.head.appendChild(s);
  }

  /* ---------- حالة الشريط العلوي ---------- */
  function initTopnav() {
    const bar = $("#topnav");
    if (!bar) return;
    const onScroll = () => bar.classList.toggle("scrolled", window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* ---------- ظهور تدريجي ---------- */
  function initReveal() {
    const items = $$(".reveal");
    if (!items.length) return;
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      items.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    items.forEach((el) => io.observe(el));
  }

  /* ---------- نموذج الحجز ---------- */
  const LEAD_KEY = "llm_ds_leads_v1";

  function readLocalLeads() {
    try {
      return JSON.parse(localStorage.getItem(LEAD_KEY) || "[]");
    } catch (_) {
      return [];
    }
  }

  function storeLocalLead(lead) {
    try {
      const leads = readLocalLeads();
      leads.push(lead);
      localStorage.setItem(LEAD_KEY, JSON.stringify(leads.slice(-50)));
      return true;
    } catch (_) {
      return false;
    }
  }

  function setStatus(el, kind, message) {
    el.textContent = message;
    el.classList.remove("ok", "err");
    if (kind) el.classList.add(kind);
  }

  function initLeadForm() {
    const form = $("#lead-form");
    const status = $("#form-status");
    const button = $("#lead-button");
    if (!form || !status || !button) return;

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      status.classList.remove("ok", "err");

      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      const email = form.elements.email.value.trim().toLowerCase();
      const name = (form.elements.name.value || "").trim();
      const interests = $$('.chips input[type="checkbox"]:checked', form).map((el) => el.value);
      const lead = {
        name,
        email,
        interests,
        consent: true,
        page: window.location.href,
        at: new Date().toISOString(),
      };

      button.disabled = true;
      button.textContent = "جارٍ تسجيل اهتمامك…";

      let sent = false;
      if (cfg.formEndpoint) {
        try {
          const data = new FormData();
          data.append("name", name);
          data.append("email", email);
          data.append("interests", interests.join("، "));
          data.append("consent", "yes");
          data.append("page", lead.page);
          data.append("_subject", "حجز مبكر — المحوّلات والنماذج اللغوية الكبيرة");
          const res = await fetch(cfg.formEndpoint, {
            method: "POST",
            body: data,
            headers: { Accept: "application/json" },
          });
          sent = res.ok;
          if (!res.ok) throw new Error("HTTP " + res.status);
        } catch (err) {
          sent = false;
        }
      }

      storeLocalLead(lead);

      if (cfg.formEndpoint && sent) {
        setStatus(status, "ok", "وصلنا اهتمامك — سنراك في بريد الإطلاق.");
        form.querySelectorAll("input[type=email], input[type=text]").forEach((el) => (el.value = ""));
      } else if (cfg.formEndpoint) {
        setStatus(status, "err", "تعذر الإرسال الآن — تحقق من الاتصال أو حاول لاحقًا.");
      } else {
        setStatus(status, "ok", "سُجّل اهتمامك محليًا في هذا المتصفح. لتفعيل الإرسال الفعلي أضف formEndpoint في assets/config.js.");
      }

      button.disabled = false;
      button.textContent = "احجز مقعدي في قائمة الإطلاق";
    });
  }

  /* ---------- اختبار تفاعلي ---------- */
  function initQuiz() {
    const form = $("#quiz-form");
    if (!form) return;
    const result = $("#quiz-result");
    const scoreEl = $("#quiz-score");
    const titleEl = $("#quiz-title");
    const copyEl = $("#quiz-copy");
    const reset = $("#quiz-reset");
    if (!result || !scoreEl || !titleEl || !copyEl) return;

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const total = $$('input[type="radio"]:checked', form).reduce((sum, el) => sum + Number(el.value || 0), 0);
      const pct = Math.round((total / 12) * 100);
      scoreEl.textContent = String(pct);
      let title, copy;
      if (total <= 4) {
        title = "أنت في بداية الرحلة — وهذا في مصلحتك.";
        copy = "ستستمع من أول خلية عصبية حتى آخر Agent بهدوء وبلا افتراضات سابقة. ابدأ بالفصل الأول، وسترى الفكرة تعمل أمامك قبل أن تقرر أي شيء.";
      } else if (total <= 8) {
        title = "لديك الأساسيات — حان وقت حسم القرارات المكلفة.";
        copy = "أنت تعرف الإجماليات، لكن الكتاب سينقلك من “أظن” إلى “أقرر بثقة”: LoRA أم Full FT؟ DPO أم GRPO؟ RAG هجين أم reranker؟ وهذا ما يوفر أسابيع من التجربة العشوائية.";
      } else {
        title = "مستوى متقدم — هذا الكتاب سيضيف لك مخرجات ملموسة.";
        copy = "في هذا المستوى، القيمة ليست في أن تتعلم ما تعرفه، بل في أن تخرجه بأنضج صورة: مشاريع، نشر، تقييم، وقرارات 2026 التي تفصل “أجيد” عن “أثق أن أنظمتي تعمل”.";
      }
      titleEl.textContent = title;
      copyEl.textContent = copy;
      result.hidden = false;
      const link = result.querySelector("a");
      if (link && cfg.freeChapterUrl) link.setAttribute("href", cfg.freeChapterUrl);
    });

    if (reset) {
      reset.addEventListener("click", () => {
        form.reset();
        result.hidden = true;
        scoreEl.textContent = "0";
      });
    }
  }

  /* ---------- المشاركة ---------- */
  function initShare() {
    const canonical = $('link[rel="canonical"]');
    const url = canonical ? canonical.href : window.location.href;
    const text = "المحوّلات والنماذج اللغوية الكبيرة — كتاب LLMs بالعربية: من الصفر إلى الإنتاج، والفصل الأول مجاني.";
    const share = {
      whatsapp: "https://wa.me/?text=" + encodeURIComponent(text + " " + url),
      x: "https://twitter.com/intent/tweet?text=" + encodeURIComponent(text) + "&url=" + encodeURIComponent(url),
      linkedin: "https://www.linkedin.com/sharing/share-offsite/?url=" + encodeURIComponent(url),
      email:
        "mailto:?subject=" +
        encodeURIComponent("كتاب المحوّلات والنماذج اللغوية الكبيرة") +
        "&body=" +
        encodeURIComponent(text + "\n" + url),
    };
    $$("[data-share]").forEach((el) => {
      const kind = el.getAttribute("data-share");
      if (share[kind]) el.setAttribute("href", share[kind]);
    });

    const copyBtn = $("#copy-link");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(url);
          copyBtn.setAttribute("aria-label", "تم نسخ الرابط");
          copyBtn.style.borderColor = "var(--cyan-2)";
          copyBtn.style.color = "var(--cyan-2)";
          window.setTimeout(() => copyBtn.removeAttribute("style"), 1600);
        } catch (_) {
          window.prompt("انسخ الرابط:", url);
        }
      });
    }
  }

  /* ---------- السنة ---------- */
  function initYear() {
    const el = $("#year");
    if (el) el.textContent = String(new Date().getFullYear());
  }

  document.addEventListener("DOMContentLoaded", () => {
    initLinks();
    initAnalytics();
    initTopnav();
    initReveal();
    initLeadForm();
    initQuiz();
    initShare();
    initYear();
  });
})();
