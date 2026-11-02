# -*- coding: utf-8 -*-
"""
الفصل 19: التكميم والتقطير — كيف تصنع نموذجًا أصغر لا يخسر عقله

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 19 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch19_quantization-distillation.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch19_quantization-distillation.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] # في vLLM: تشغيل نموذج بـ FP8 مع KV Cache FP8 ════════


# في vLLM: تشغيل نموذج بـ FP8 مع KV Cache FP8
# vllm serve meta-llama/Llama-3-70B --quantization fp8 --kv-cache-dtype fp8


# ════════ [2/5] recipe = AWQModifier( ════════


from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import AWQModifier

# تكميم AWQ (W4A16)
recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

oneshot(
    model="meta-llama/Llama-3.1-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama-3.1-8b-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)


# ════════ [3/5] recipe = QuantizationModifier( ════════


from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# FP8 W8A8 (أوزان وتنشيطات 8 بت)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",   # أو "FP8" للثابت
    ignore=["lm_head"],
)

oneshot(
    model="meta-llama/Llama-3.1-8B",
    recipe=recipe,
    output_dir="./llama-3.1-8b-fp8",
)


# ════════ [4/5] def distillation_loss(student_logits, teacher_logits, labels, alpha=0. ════════


import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0, ignore_index=-100):
    """
    خسارة التقطير: مزيج من Cross-Entropy (مع البيانات الحقيقية) وKL (مع توزيع المعلم).
    """
    # Cross-Entropy مع البيانات الحقيقية
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    # KL divergence مع تنعيم درجة الحرارة
    # نحتاج إلى قناع لتجاهل الحشو
    mask = (labels != ignore_index).float().view(-1, 1)
    student_log_probs = F.log_softmax(student_logits.view(-1, student_logits.size(-1)) / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / T, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1, keepdim=True)

    kl = (kl * mask).sum() / mask.sum()
    kd_loss = kl * (T ** 2)

    return alpha * ce + (1 - alpha) * kd_loss


# ════════ [5/5] def prune_layers(model, layers_to_prune): ════════


def prune_layers(model, layers_to_prune):
    """
    يحذف طبقات محددة من نموذج MiniGPT.
    layers_to_prune: قائمة مؤشرات الطبقات (مثل [4, 5, 6]).
    """
    new_blocks = []
    for i, block in enumerate(model.blocks):
        if i not in layers_to_prune:
            new_blocks.append(block)
    model.blocks = torch.nn.ModuleList(new_blocks)
    print(f"Pruned {len(layers_to_prune)} layers. Remaining: {len(model.blocks)}")
    return model
