# -*- coding: utf-8 -*-
"""
الفصل 30: التقطير (Knowledge Distillation) — كيف تعلّم نموذجًا صغيرًا أن يفكر مثل معلمه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 30 · عدد كتل بايثون في هذا الملف: 5

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch30_knowledge-distillation.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch30_knowledge-distillation.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/5] def distillation_loss( ════════


import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(
    student_logits: torch.Tensor,    # (batch, num_classes)
    teacher_logits: torch.Tensor,    # (batch, num_classes)
    labels: torch.Tensor,             # (batch,) أو (batch, seq)
    alpha: float = 0.5,
    temperature: float = 2.0,
    ignore_index: int = -100,
):
    """
    خسارة التقطير: مزيج من CE (مع البيانات) وKL (مع المعلم).
    """
    # 1. CE مع البيانات الحقيقية
    if student_logits.dim() == 3:
        # للغة: (batch, seq, vocab)
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
        )
        mask = (labels != ignore_index).view(-1, 1).float()
        student_log_probs = F.log_softmax(student_logits.view(-1, student_logits.size(-1)) / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none", log_target=False).sum(dim=-1, keepdim=True)
        kl = (kl * mask).sum() / mask.sum()
    else:
        # للتصنيف: (batch, num_classes)
        ce = F.cross_entropy(student_logits, labels, ignore_index=ignore_index)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)

    kd = kl * (temperature ** 2)
    return alpha * ce + (1 - alpha) * kd


# ════════ [2/5] def train_distillation( ════════


def train_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    optimizer,
    epochs: int = 3,
    alpha: float = 0.5,
    temperature: float = 4.0,
    device: str = "cuda",
):
    """
    تدريب الطالب على تقليد المعلم.
    """
    student = student.to(device)
    teacher = teacher.to(device).eval()

    # جمّد المعلم
    for p in teacher.parameters():
        p.requires_grad = False

    student.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            student_logits = student(input_ids).logits

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=alpha, temperature=temperature,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    return student


# ════════ [3/5] def top_k_distillation_loss( ════════


def top_k_distillation_loss(
    student_logits, teacher_logits, labels,
    k: int = 50, alpha: float = 0.5, temperature: float = 2.0,
):
    # احصل على top-k من المعلم
    teacher_topk = torch.topk(teacher_logits, k, dim=-1)
    teacher_values = teacher_topk.values  # (batch, seq, k)
    teacher_indices = teacher_topk.indices  # (batch, seq, k)

    # احصل على logits الطالب في نفس المواقع
    student_topk = torch.gather(student_logits, -1, teacher_indices)  # (batch, seq, k)

    # توزيع ناعم على top-k فقط
    teacher_probs = F.softmax(teacher_values / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)
    return alpha * ce + (1 - alpha) * kl


# ════════ [4/5] def self_distillation_step( ════════


def self_distillation_step(
    model,
    input_ids,
    constrained_prompt: str,
    tokenizer,
    optimizer,
    temperature: float = 2.0,
):
    """
    خطوة تقطير ذاتي:
    1. ولّد rollouts من النموذج (بدون قيود).
    2. استخدم النموذج نفسه مع تعليمات مقيدة (مثل "كن موجزًا") كمعلم.
    3. درّب النموذج على تقليد التوزيع المقيد.
    """
    model.train()

    # 1. rollout (بدون قيود)
    with torch.no_grad():
        rollout = model.generate(input_ids, max_new_tokens=50, do_sample=True)

    # 2. المعلم الذاتي (مقيد)
    constrained_input = torch.cat([
        tokenizer(constrained_prompt, return_tensors="pt").input_ids.to(input_ids.device),
        rollout,
    ], dim=1)

    with torch.no_grad():
        teacher_logits = model(constrained_input).logits

    # 3. الطالب (بدون قيود)
    student_logits = model(rollout).logits

    # 4. خسارة التقطير
    loss = distillation_loss(
        student_logits, teacher_logits[:, -student_logits.size(1):],
        labels=rollout,  # أو استخدم -100 لتجاهل البادئة
        alpha=0.0, temperature=temperature,  # KL فقط
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ════════ [5/5] def prune_and_distill( ════════


def prune_and_distill(
    original_model,
    pruned_model,
    train_loader,
    optimizer,
    epochs: int = 3,
    temperature: float = 2.0,
):
    """
    التشذيب + التقطير: النموذج الأصلي (المعلم) يوجه النموذج المشذب (الطالب).
    """
    original_model.eval()
    pruned_model.train()

    for p in original_model.parameters():
        p.requires_grad = False

    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = original_model(input_ids).logits

            student_logits = pruned_model(input_ids).logits

            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                alpha=0.5, temperature=temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pruned_model.parameters(), 1.0)
            optimizer.step()

    return pruned_model
