# -*- coding: utf-8 -*-
"""
الفصل 11: تسريع العمليات الحسابية — عندما تصبح السرعة قرارًا هندسيًا

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثالث · الفصل 11 · عدد كتل بايثون في هذا الملف: 1

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch11_compute-acceleration.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch11_compute-acceleration.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/1] recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_he ════════


from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# تكميم AWQ (مبسط)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(
    model="meta-llama/Llama-3-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama-3-8b-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)
