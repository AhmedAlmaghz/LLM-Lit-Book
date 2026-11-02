# -*- coding: utf-8 -*-
"""
الفصل 29: لماذا لا تبدأ من الصفر؟ — التكلفة والوقت والبيانات: حساب الصفقة

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 29 · عدد كتل بايثون في هذا الملف: 1

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch29_build-vs-adapt-cost.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch29_build-vs-adapt-cost.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/1] def choose_strategy( ════════


def choose_strategy(
    task_specific: bool,
    has_updated_knowledge: bool,
    data_size_tokens: int,
    budget_usd: float,
    time_weeks: int,
):
    """
    اقتراح استراتيجية بناءً على المعطيات.
    """
    if not task_specific and data_size_tokens > 1e12 and budget_usd > 1e6:
        return "Pre-training من الصفر"

    if has_updated_knowledge:
        if budget_usd < 1000:
            return "RAG (بدون تدريب)"
        return "RAG + LoRA"

    if task_specific:
        if budget_usd < 100:
            return "Prompt Engineering + Few-shot"
        elif budget_usd < 1000:
            return "LoRA"
        else:
            return "LoRA + Distillation"

    if data_size_tokens > 1e10 and budget_usd > 50000:
        return "Continual Pre-training"

    return "Fine-tuning (LoRA)"

# مثال
print(choose_strategy(
    task_specific=True,
    has_updated_knowledge=False,
    data_size_tokens=1e7,
    budget_usd=500,
    time_weeks=2,
))
# LoRA
