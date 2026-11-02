# -*- coding: utf-8 -*-
"""
الفصل 32: التشذيب (Pruning) واستعادة الجودة — كيف تحذف من النموذج دون أن تفقد عقله

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 32 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch32_pruning.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch32_pruning.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] def unstructured_prune(model, amount=0.3): ════════


def unstructured_prune(model, amount=0.3):
    """
    يحذف أصغر 30% من الأوزان في كل طبقة خطية.
    """
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


# ════════ [2/8] def magnitude_prune_heads(model, num_heads_to_keep): ════════


def magnitude_prune_heads(model, num_heads_to_keep):
    """
    يحذف رؤوس الانتباه ذات المعيار الأصغر (L2 norm).
    """
    for name, module in model.named_modules():
        if hasattr(module, "num_heads"):
            # احسب معيار كل رأس
            # (تبسيط: يعتمد على البنية)
            pass


# ════════ [3/8] def movement_prune(model, initial_state_dict, amount=0.3): ════════


def movement_prune(model, initial_state_dict, amount=0.3):
    """
    يحذف الأوزان التي لم تتحرك كثيرًا عن التهيئة الأصلية.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name + ".weight" in initial_state_dict:
                delta = (module.weight - initial_state_dict[name + ".weight"]).abs()
                # احذف أصغر delta
                threshold = delta.flatten().topk(int(delta.numel() * (1 - amount))).values.min()
                mask = delta >= threshold
                module.weight.data *= mask.float()
    return model


# ════════ [4/8] def prune_layers(model, layers_to_prune: list[int]): ════════


def prune_layers(model, layers_to_prune: list[int]):
    """
    يحذف طبقات محددة من نموذج.
    """
    new_blocks = []
    for i, block in enumerate(model.blocks):
        if i not in layers_to_prune:
            new_blocks.append(block)
    model.blocks = torch.nn.ModuleList(new_blocks)
    print(f"Pruned {len(layers_to_prune)} layers. Remaining: {len(model.blocks)}")
    return model


# ════════ [5/8] def self_data_distillation_step( ════════


def self_data_distillation_step(
    teacher, student, tokenizer, prompts, optimizer, temperature=2.0
):
    """
    خطوة Self-Data Distillation:
    1. المعلم يولد إجابات.
    2. الطالب يتدرب على تقليد توزيع المعلم.
    """
    teacher.eval()
    student.train()

    # 1. توليد من المعلم
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(teacher.device)
        teacher_outputs = teacher.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.8
        )

    # 2. الطالب يتدرب على نفس المدخلات
    student_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(student.device)
    student_labels = teacher_outputs  # الإجابات المولدة هي "التسميات"

    # 3. توزيع المعلم على نفس الإجابات
    with torch.no_grad():
        teacher_logits = teacher(teacher_outputs).logits

    student_logits = student(teacher_outputs).logits

    loss = distillation_loss(
        student_logits, teacher_logits, student_labels,
        alpha=0.5, temperature=temperature,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ════════ [6/8] def prune_and_recover( ════════


def prune_and_recover(
    teacher_model,  # النموذج الأصلي (المعلم)
    student_model,  # النموذج المشذب (الطالب)
    unlabeled_prompts: list[str],
    tokenizer,
    epochs: int = 3,
    temperature: float = 2.0,
    lr: float = 1e-4,
):
    """
    تشذيب + Self-Data Distillation.
    """
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(unlabeled_prompts), 4):  # batch size 4
            batch_prompts = unlabeled_prompts[i:i+4]
            loss = self_data_distillation_step(
                teacher_model, student_model, tokenizer,
                batch_prompts, optimizer, temperature,
            )
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss:.4f}")

    return student_model


# ════════ [7/8] def measure_recovery(teacher, student, eval_loader, device="cuda"): ════════


def measure_recovery(teacher, student, eval_loader, device="cuda"):
    """
    يقيس نسبة استعادة الجودة.
    """
    teacher.eval()
    student.eval()

    teacher_loss = evaluate_loss(teacher, eval_loader, device)
    student_loss = evaluate_loss(student, eval_loader, device)

    # Perplexity
    teacher_ppl = torch.exp(torch.tensor(teacher_loss))
    student_ppl = torch.exp(torch.tensor(student_loss))

    recovery = (teacher_ppl / student_ppl) * 100  # تقريبي
    return {
        "teacher_ppl": teacher_ppl.item(),
        "student_ppl": student_ppl.item(),
        "recovery_pct": recovery.item(),
    }


# ════════ [8/8] def progressive_prune( ════════


def progressive_prune(
    model, teacher, tokenizer, prompts,
    target_sparsity: float = 0.3,
    steps: int = 3,
    epochs_per_step: int = 1,
):
    """
    تشذيب تدريجي: 10% في كل خطوة، مع إعادة تدريب.
    """
    current_sparsity = 0.0
    sparsity_per_step = target_sparsity / steps

    for step in range(steps):
        # 1. شذب
        prune_amount = sparsity_per_step / (1 - current_sparsity)
        unstructured_prune(model, amount=prune_amount)
        current_sparsity += sparsity_per_step

        # 2. أعد التدريب (تقطير)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(epochs_per_step):
            for i in range(0, len(prompts), 4):
                self_data_distillation_step(
                    teacher, model, tokenizer, prompts[i:i+4], optimizer
                )

        # 3. قيّم
        print(f"Step {step+1}/{steps} | Sparsity: {current_sparsity*100:.1f}%")

    return model
