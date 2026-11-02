# -*- coding: utf-8 -*-
"""
الفصل 3: بنية التدريب الحديثة — من GPU واحدة إلى عنقود

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الأول · الفصل 3 · عدد كتل بايثون في هذا الملف: 9

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch03_modern-training-infrastructure.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch03_modern-training-infrastructure.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/9] device = "cuda" ════════


import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

device = "cuda"
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# BF16: لا يحتاج GradScaler
use_bf16 = torch.cuda.is_bf16_supported()

# FP16: يحتاج GradScaler لمنع التلاشي
scaler = GradScaler(enabled=not use_bf16)

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])

    if use_bf16:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # للـ clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


# ════════ [2/9] accumulation_steps = 4 ════════


accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(dataloader):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss = loss / accumulation_steps  # مهم جدًا!

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# ════════ [3/9] torch.distributed.init_process_group(backend="nccl") ════════


# تشغيل DDP (مبسط)
# كل عملية (process) تعمل على GPU مختلفة
torch.distributed.init_process_group(backend="nccl")
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


# ════════ [4/9] model = FSDP(model, use_orig_params=True) ════════


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, use_orig_params=True)


# ════════ [5/9] total_norm = 0.0 ════════


total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.detach().data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"grad_norm: {total_norm:.4f}")


# ════════ [6/9] with torch.autograd.set_detect_anomaly(True): ════════


with torch.autograd.set_detect_anomaly(True):
    loss.backward()


# ════════ [7/9] assert not torch.isnan(batch["input_ids"]).any(), "NaN in input!" ════════


assert not torch.isnan(batch["input_ids"]).any(), "NaN in input!"
assert not torch.isinf(batch["input_ids"]).any(), "Inf in input!"


# ════════ [8/9] torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ════════


torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


# ════════ [9/9] def save_checkpoint(path, model, optimizer, scheduler, scaler, step, e ════════


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, epoch):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint["step"], checkpoint["epoch"]
