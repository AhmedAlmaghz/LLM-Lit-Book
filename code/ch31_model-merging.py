# -*- coding: utf-8 -*-
"""
الفصل 31: الدمج (Model Merging) — عندما يصبح 1+1 أكبر من 2

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 31 · عدد كتل بايثون في هذا الملف: 7

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch31_model-merging.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch31_model-merging.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/7] def model_soup(model_paths: list[str], output_path: str): ════════


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


# ════════ [2/7] def merge_loras(base_model, lora_paths: list[str], weights: list[float ════════


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


# ════════ [3/7] def ties_merge( ════════


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


# ════════ [4/7] def dare_merge( ════════


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


# ════════ [5/7] def task_arithmetic( ════════


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


# ════════ [6/7] def task_negation( ════════


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


# ════════ [7/7] def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5, eps: flo ════════


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
