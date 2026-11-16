#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — لماذا لا تبدأ من الصفر؟ — التكلفة والوقت والبيانات: حساب الصفقة
Transformers and Large Language Models — Build Vs Adapt Cost

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-29-llm-ds (lesson 29)
Code files: 34 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part07_build-on-existing-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ خلية الإعداد — شغّلها أولًا (Colab / Kaggle / محليًا)
# تثبّت ما ينقص فقط، وتتجاهل ما لا يدعم جهازك.
import sys, subprocess, importlib

PKGS = [
    ("torch", "torch"), ("transformers", "transformers"),
    ("datasets", "datasets"), ("accelerate", "accelerate"),
    ("peft", "peft"), ("trl", "trl"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"), ("pandas", "pandas"),
    ("numpy", "numpy"), ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"), ("evaluate", "evaluate"),
]
todo = []
for mod, pipname in PKGS:
    try:
        importlib.import_module(mod)
    except Exception:
        todo.append(pipname)
if todo:
    print("تثبيت الحزم الناقصة:", todo)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
    except Exception as e:
        print("تعذّر تثبيت بعض الحزم (ثبّتها يدويًا إن احتجتها):", e)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print("device:", DEVICE)
print("جاهز ✅")

# ────────────────────────────────────────────────────────────────────────
# [002] code/notebooks/part07_build-on-existing-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part07_build-on-existing-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part07_build-on-existing-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part07_build-on-existing-models.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part07_build-on-existing-models.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part07_build-on-existing-models.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part07_build-on-existing-models.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM

def model_soup(model_paths: list[str], output_path: str):
    """
    يدمج عدة نماذج (نفس المعمارية) بمتوسط الأوزان.
    """
    # حمل النماذج
    models = [AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16) for p in model_paths]

    # تحقق من التطابق
    state_dicts = [m.state_dict() for m in models]
    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == set(keys), "State dicts must have the same keys"

    # متوسط
    merged_sd = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        # للطبقات العددية (int)، لا نأخذ متوسطًا (مثل num_batches_tracked)
        if tensors[0].dtype in [torch.int64, torch.int32, torch.bool]:
            merged_sd[key] = tensors[0]
        else:
            merged_sd[key] = torch.stack(tensors).mean(dim=0)

    # حمل في نموذج جديد
    merged_model = AutoModelForCausalLM.from_pretrained(model_paths[0], torch_dtype=torch.bfloat16)
    merged_model.load_state_dict(merged_sd)
    merged_model.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")

# مثال
model_soup(["./model_seed1", "./model_seed2", "./model_seed3"], "./soup_model")

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part07_build-on-existing-models.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from peft import PeftModel

def merge_loras(base_model, lora_paths: list[str], weights: list[float] = None):
    """
    يدمج عدة LoRAs بأوزان محددة.
    """
    if weights is None:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)

    base_sd = base_model.state_dict()
    merged_sd = {k: v.clone() for k, v in base_sd.items()}

    for path, w in zip(lora_paths, weights):
        lora = PeftModel.from_pretrained(base_model, path)
        lora_sd = lora.state_dict()
        for k in lora_sd:
            if "lora" in k:
                # أضف المساهمة المرجحة
                if k in merged_sd:
                    merged_sd[k] = merged_sd[k] + w * (lora_sd[k] - base_sd.get(k, 0))
                else:
                    merged_sd[k] = w * lora_sd[k]

    merged_model = AutoModelForCausalLM.from_pretrained(base_model.config._name_or_path)
    merged_model.load_state_dict(merged_sd)
    return merged_model

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part07_build-on-existing-models.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def ties_merge(
    base_state_dict: dict,
    finetuned_state_dicts: list[dict],
    k: float = 0.2,  # احتفظ بأعلى 20% من التغييرات
    weights: list[float] = None,
):
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge.
    """
    if weights is None:
        weights = [1.0] * len(finetuned_state_dicts)
    assert len(weights) == len(finetuned_state_dicts)

    merged = {}
    for key in base_state_dict:
        base = base_state_dict[key]
        if base.dtype in [torch.int64, torch.int32, torch.bool]:
            merged[key] = base
            continue

        # 1. Trim: احتفظ بأكبر k% من |delta|
        deltas = []
        for sd in finetuned_state_dicts:
            delta = sd[key] - base
            # احتفظ بأكبر k% فقط
            flat = delta.flatten()
            kth = int(len(flat) * k)
            if kth > 0:
                threshold = flat.abs().topk(kth).values.min()
                delta = torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))
            deltas.append(delta)

        # 2. Elect Sign: التصويت المرجح
        stacked = torch.stack(deltas)  # (N, ...)
        weighted = stacked * torch.tensor(weights).view(-1, *([1] * (stacked.dim() - 1)))
        sign = torch.sign(weighted.sum(dim=0))

        # 3. Disjoint Merge: ادمج فقط ما يتفق مع الإشارة
        merged_delta = torch.zeros_like(base)
        count = torch.zeros_like(base)
        for delta, w in zip(deltas, weights):
            agree = (torch.sign(delta) == sign) & (delta != 0)
            merged_delta += torch.where(agree, delta * w, torch.zeros_like(delta))
            count += agree.float() * w
        merged_delta = merged_delta / count.clamp(min=1e-8)

        merged[key] = base + merged_delta

    return merged

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part07_build-on-existing-models.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def dare_merge(
    base_state_dict: dict,
    finetuned_state_dicts: list[dict],
    drop_rate: float = 0.9,  # أسقط 90% من التغييرات الصغيرة
    weights: list[float] = None,
):
    """
    DARE-Merge: Drop, Rescale, ثم دمج.
    """
    if weights is None:
        weights = [1.0] * len(finetuned_state_dicts)

    merged = {}
    for key in base_state_dict:
        base = base_state_dict[key]
        if base.dtype in [torch.int64, torch.int32, torch.bool]:
            merged[key] = base
            continue

        total_delta = torch.zeros_like(base)
        for sd, w in zip(finetuned_state_dicts, weights):
            delta = sd[key] - base
            # Drop: قناع عشوائي
            mask = (torch.rand_like(delta) > drop_rate).float()
            # Rescale
            delta = delta * mask / (1 - drop_rate)
            total_delta += w * delta

        merged[key] = base + total_delta / sum(weights)

    return merged

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part07_build-on-existing-models.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
def task_arithmetic(
    base_state_dict: dict,
    task_state_dicts: list[dict],
    lambdas: list[float] = None,
):
    """
    Task Arithmetic: ادمج متجهات المهام.
    """
    if lambdas is None:
        lambdas = [1.0] * len(task_state_dicts)

    merged = {}
    for key in base_state_dict:
        base = base_state_dict[key]
        if base.dtype in [torch.int64, torch.int32, torch.bool]:
            merged[key] = base
            continue

        delta = torch.zeros_like(base)
        for sd, lam in zip(task_state_dicts, lambdas):
            delta += lam * (sd[key] - base)

        merged[key] = base + delta

    return merged

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part07_build-on-existing-models.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
def task_negation(
    base_state_dict: dict,
    positive_state_dict: dict,   # النموذج الذي نريد "إضافته"
    negative_state_dict: dict,   # النموذج الذي نريد "طرحه"
    lambda_pos: float = 1.0,
    lambda_neg: float = 1.0,
):
    """
    Task Arithmetic: أضف مهمة، واطرح أخرى.
    """
    merged = {}
    for key in base_state_dict:
        base = base_state_dict[key]
        if base.dtype in [torch.int64, torch.int32, torch.bool]:
            merged[key] = base
            continue

        pos_delta = positive_state_dict[key] - base
        neg_delta = negative_state_dict[key] - base
        merged[key] = base + lambda_pos * pos_delta - lambda_neg * neg_delta

    return merged

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part07_build-on-existing-models.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5, eps: float = 1e-8):
    """
    SLERP بين متجهين.
    """
    # تطبيع
    v0_norm = v0 / (v0.norm() + eps)
    v1_norm = v1 / (v1.norm() + eps)

    # زاوية
    dot = (v0_norm * v1_norm).sum()
    dot = dot.clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)

    # إذا كانت الزاوية صغيرة، استخدم الاستيفاء الخطي
    if omega.abs() < 1e-3:
        return (1 - t) * v0 + t * v1

    sin_omega = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / sin_omega) * v0 + (torch.sin(t * omega) / sin_omega) * v1

def slerp_merge(sd1: dict, sd2: dict, t: float = 0.5):
    """
    SLERP على كل المعاملات.
    """
    merged = {}
    for key in sd1:
        if sd1[key].dtype in [torch.int64, torch.int32, torch.bool]:
            merged[key] = sd1[key]
        else:
            merged[key] = slerp(sd1[key].float(), sd2[key].float(), t).to(sd1[key].dtype)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part07_build-on-existing-models.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
def unstructured_prune(model, amount=0.3):
    """
    يحذف أصغر 30% من الأوزان في كل طبقة خطية.
    """
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part07_build-on-existing-models.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
def magnitude_prune_heads(model, num_heads_to_keep):
    """
    يحذف رؤوس الانتباه ذات المعيار الأصغر (L2 norm).
    """
    for name, module in model.named_modules():
        if hasattr(module, "num_heads"):
            # احسب معيار كل رأس
            # (تبسيط: يعتمد على البنية)
            pass

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part07_build-on-existing-models.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part07_build-on-existing-models.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part07_build-on-existing-models.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part07_build-on-existing-models.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part07_build-on-existing-models.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part07_build-on-existing-models.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part07_build-on-existing-models.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class MultiLoRAServer:
    def __init__(self, base_model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.adapters = {}  # name -> PeftModel

    def load_adapter(self, name: str, adapter_path: str):
        """تحميل LoRA وتخزينها."""
        if name in self.adapters:
            return
        model = PeftModel.from_pretrained(
            self.base_model, adapter_path, adapter_name=name
        )
        self.adapters[name] = model
        print(f"Loaded adapter: {name}")

    def switch_adapter(self, name: str):
        """تبديل المحول النشط."""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not loaded")
        self.base_model.set_adapter(name)

    def generate(self, prompt: str, adapter_name: str, **kwargs):
        """توليد باستخدام محول محدد."""
        if adapter_name not in self.adapters:
            self.load_adapter(adapter_name, f"./adapters/{adapter_name}")

        self.switch_adapter(adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.base_model.generate(**inputs, **kwargs)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# مثال
server = MultiLoRAServer("Qwen/Qwen2.5-1.5B")
server.load_adapter("customer_a", "./adapters/customer_a")
server.load_adapter("customer_b", "./adapters/customer_b")

print(server.generate("مرحبًا، كيف حالك؟", adapter_name="customer_a"))
print(server.generate("مرحبًا، كيف حالك؟", adapter_name="customer_b"))

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part07_build-on-existing-models.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# استخدم LoRA محدد عبر model
response = client.chat.completions.create(
    model="customer_a",  # اسم LoRA
    messages=[{"role": "user", "content": "مرحبًا"}],
)
print(response.choices[0].message.content)

# ────────────────────────────────────────────────────────────────────────
# [025] code/notebooks/part07_build-on-existing-models.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def weighted_sum_loras(base_model, lora_paths: list[str], weights: list[float]):
    """
    دمج عدة LoRAs بأوزان محددة.
    """
    from peft import PeftModel
    import torch

    base_sd = base_model.state_dict()
    merged = {k: v.clone() for k, v in base_sd.items()}

    for path, w in zip(lora_paths, weights):
        lora = PeftModel.from_pretrained(base_model, path)
        lora_sd = lora.state_dict()

        for k in lora_sd:
            if "lora_A" in k or "lora_B" in k:
                # أوزان LoRA: أضف المساهمة المرجحة
                if k in merged:
                    merged[k] = merged[k] + w * lora_sd[k]
                else:
                    merged[k] = w * lora_sd[k]

    merged_model = AutoModelForCausalLM.from_pretrained(base_model.config._name_or_path)
    merged_model.load_state_dict(merged)
    return merged_model

# ────────────────────────────────────────────────────────────────────────
# [026] code/notebooks/part07_build-on-existing-models.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
from peft import PeftModel

def stack_loras(base_model, lora_paths: list[str]):
    """
    تكديس LoRAs: كل واحد يعدل النموذج بشكل متسلسل.
    """
    model = base_model
    for path in lora_paths:
        model = PeftModel.from_pretrained(model, path)
    return model

# ────────────────────────────────────────────────────────────────────────
# [027] code/notebooks/part07_build-on-existing-models.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
class LoRARouter:
    def __init__(self, adapter_descriptions: dict[str, str], embedder):
        """
        adapter_descriptions: {name: description}
        embedder: نموذج تضمين
        """
        self.names = list(adapter_descriptions.keys())
        self.descriptions = list(adapter_descriptions.values())
        self.embeddings = embedder.encode(self.descriptions, normalize_embeddings=True)
        self.embedder = embedder

    def route(self, query: str) -> str:
        """يختار LoRA بناءً على تشابه الطلب مع الوصف."""
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_emb
        best_idx = scores.argmax()
        return self.names[best_idx]

# مثال
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

router = LoRARouter(
    {
        "math": "أسئلة رياضية، حسابات، معادلات",
        "code": "برمجة، كود، خوارزميات",
        "arabic": "أسئلة بالعربية الفصحى",
        "customer_a": "أسئلة عن منتجات شركة A",
    },
    embedder,
)

print(router.route("كيف أحل معادلة من الدرجة الثانية؟"))  # math
print(router.route("اكتب دالة بايثون لجمع رقمين"))  # code

# ────────────────────────────────────────────────────────────────────────
# [028] code/notebooks/part07_build-on-existing-models.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
from collections import OrderedDict
import torch

class LoRACache:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.cache = OrderedDict()  # name -> PeftModel

    def get(self, name: str):
        if name in self.cache:
            self.cache.move_to_end(name)  # اجعله الأحدث
            return self.cache[name]
        return None

    def put(self, name: str, model):
        if name in self.cache:
            self.cache.move_to_end(name)
        else:
            if len(self.cache) >= self.max_size:
                # أفرغ الأقدم (LRU)
                oldest, _ = self.cache.popitem(last=False)
                print(f"Evicted: {oldest}")
            self.cache[name] = model

    def __contains__(self, name):
        return name in self.cache

# ────────────────────────────────────────────────────────────────────────
# [029] code/notebooks/part07_build-on-existing-models.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
from openai import OpenAI
import json
import random

client = OpenAI()

def generate_instructions(seed_instructions: list[str], num_new: int = 10) -> list[str]:
    """
    يولّد تعليمات جديدة بناءً على بذور.
    """
    prompt = f"""لديك هذه التعليمات كأمثلة:
{chr(10).join(f'- {s}' for s in seed_instructions)}

ولّد {num_new} تعليمات جديدة، متنوعة، بالعربية، تشبه الأمثلة لكنها مختلفة.
أعد قائمة JSON فقط، دون أي نص آخر: ["تعليمات 1", "تعليمات 2", ...]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return []

def generate_answer(instruction: str) -> str:
    """يولّد إجابة على تعليمات."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# مثال
seeds = [
    "اكتب فقرة عن أهمية القراءة.",
    "لخص هذا النص في جملتين.",
    "اشرح الفرق بين الذكاء الاصطناعي والتعلم الآلي.",
]

new_instructions = generate_instructions(seeds, num_new=5)
for inst in new_instructions:
    answer = generate_answer(inst)
    print(f"Q: {inst}\nA: {answer[:100]}...\n")

# ────────────────────────────────────────────────────────────────────────
# [030] code/notebooks/part07_build-on-existing-models.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
def self_data_distillation_step(
    model, tokenizer, prompts, optimizer, temperature=2.0, max_new_tokens=256
):
    """
    خطوة Self-Data Distillation:
    1. النموذج يولد إجابات (rollouts).
    2. النموذج يتدرب على هذه الإجابات (تقليد الذات).
    """
    model.train()

    # 1. توليد
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    # 2. تدريب على الإجابات المولدة
    # labels = outputs (مع -100 على الـ prompt إذا أردت)
    labels = outputs.clone()
    # اختياري: تجاهل الـ prompt
    labels[:, :inputs["input_ids"].size(1)] = -100

    logits = model(outputs).logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [031] code/notebooks/part07_build-on-existing-models.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def soft_bootstrap(question: str, model, tokenizer, rounds: int = 3):
    """
    يحسن الإجابة تدريجيًا عبر جولات.
    """
    answer = ""
    history = []

    for round in range(rounds):
        if round == 0:
            prompt = question
        else:
            prompt = f"""السؤال: {question}

إجابتك السابقة: {answer}

حاول تحسين إجابتك. كن أكثر دقة، وضوحًا، واكتمالًا.
الإجابة المحسنة:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        history.append(answer)

    # اختر الأفضل (هنا: الأخيرة، لكن يمكنك استخدام LLM-as-a-Judge)
    return history[-1], history

# مثال
best, all_answers = soft_bootstrap("اشرح مفهوم الانتباه في المحوّلات.", model, tokenizer)
print(f"Best answer: {best}")

# ────────────────────────────────────────────────────────────────────────
# [032] code/notebooks/part07_build-on-existing-models.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
import re
import sympy as sp

def generate_math_problem() -> tuple[str, str]:
    """يولّد مسألة رياضية وإجابتها (بسيطة)."""
    import random
    a, b = random.randint(1, 100), random.randint(1, 100)
    op = random.choice(["+", "-", "*"])
    question = f"احسب: {a} {op} {b}"
    answer = eval(f"{a} {op} {b}")
    return question, str(answer)

def verify_math_answer(question: str, answer: str) -> bool:
    """يتحقق من صحة الإجابة باستخدام sympy."""
    try:
        # استخرج التعبير من السؤال
        expr = re.search(r"احسب:\s*(.+)", question).group(1)
        expected = sp.sympify(expr).evalf()
        given = sp.sympify(answer).evalf()
        return abs(expected - given) < 1e-6
    except Exception:
        return False

def generate_verified_math_data(model, tokenizer, num_samples: int = 1000):
    """يولّد بيانات رياضية، ويفلترها بالتحقق."""
    verified = []
    for _ in range(num_samples):
        question, correct_answer = generate_math_problem()
        # النموذج يحاول الإجابة
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        model_answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # تحقق
        if verify_math_answer(question, model_answer):
            verified.append({"question": question, "answer": model_answer})

    print(f"Verified: {len(verified)}/{num_samples}")
    return verified

# ────────────────────────────────────────────────────────────────────────
# [033] code/notebooks/part07_build-on-existing-models.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
from collections import Counter
import re

def distinct_n(texts: list[str], n: int = 2) -> float:
    """
    Distinct-n: نسبة n-grams الفريدة إلى إجمالي n-grams.
    كلما زادت، زاد التنوع.
    """
    ngrams = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

# مثال
texts = ["القطة تجلس على السجادة", "القطة تنام على السجادة", "القطة تأكل"]
print(f"Distinct-1: {distinct_n(texts, 1):.3f}")
print(f"Distinct-2: {distinct_n(texts, 2):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [034] code/ch29_build-vs-adapt-cost.py
# ────────────────────────────────────────────────────────────────────────
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
