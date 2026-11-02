# -*- coding: utf-8 -*-
"""
الفصل 15: حلقة التدريب (Pre-training) — حيث يتحول النموذج من ورق إلى عقل

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 15 · عدد كتل بايثون في هذا الملف: 16

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch15_pretraining-loop.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch15_pretraining-loop.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/16] def compute_loss(logits, targets, ignore_index=-100): ════════


import torch
import torch.nn.functional as F

def compute_loss(logits, targets, ignore_index=-100):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    ignore_index: قيمة تُستبعد من حساب الخسارة (عادة -100 أو pad_id)
    """
    # نسطح الأبعاد
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


# ════════ [2/16] def compute_loss_with_z(logits, targets, z_coef=1e-4, ignore_index=-10 ════════


def compute_loss_with_z(logits, targets, z_coef=1e-4, ignore_index=-100):
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    log_z = torch.logsumexp(logits, dim=-1)
    z_loss = (log_z ** 2).mean()
    return ce + z_coef * z_loss


# ════════ [3/16] optimizer = torch.optim.AdamW( ════════


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),      # 0.95 هو الافتراضي في LLaMA/GPT-3
    eps=1e-8,
    weight_decay=0.1,        # 0.1 شائع في المحوّلات
)


# ════════ [4/16] def create_optimizer(model, lr=3e-4, weight_decay=0.1): ════════


def create_optimizer(model, lr=3e-4, weight_decay=0.1):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # لا تطبق weight_decay على الأبعاد 1D (LayerNorm, bias)
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    return optimizer


# ════════ [5/16] def get_lr( ════════


import math

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
) -> float:
    """Warmup + Cosine Decay."""
    if step < warmup_steps:
        # خطي من 0 إلى max_lr
        return max_lr * step / max(1, warmup_steps)

    if step > max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ════════ [6/16] def update_lr(optimizer, step, config): ════════


def update_lr(optimizer, step, config):
    lr = get_lr(
        step,
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ════════ [7/16] def clip_gradients(model, max_norm=1.0): ════════


def clip_gradients(model, max_norm=1.0):
    """يقص التدرجات ويعيد المعيار الأصلي (للمراقبة)."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        error_if_nonfinite=True,  # يرفع خطأ إذا كان هناك NaN/Inf
    )
    return total_norm


# ════════ [8/16] use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() ════════


use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
use_fp16 = not use_bf16 and torch.cuda.is_available()


# ════════ [9/16] def train_step(model, batch, optimizer, scaler, config, step): ════════


from torch.amp import autocast, GradScaler

def train_step(model, batch, optimizer, scaler, config, step):
    device = config["device"]
    input_ids = batch[0].to(device, non_blocking=True)
    targets = batch[1].to(device, non_blocking=True)

    # 1. صفّر التدرجات
    optimizer.zero_grad(set_to_none=True)

    # 2. Forward مع Mixed Precision
    dtype = torch.bfloat16 if config["use_bf16"] else torch.float16
    with autocast(device_type="cuda", dtype=dtype, enabled=config["use_amp"]):
        logits, loss = model(input_ids, targets)

    # 3. Backward مع Scaler (FP16 فقط)
    if config["use_fp16"]:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_gradients(model, config["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = clip_gradients(model, config["max_grad_norm"])
        optimizer.step()

    return loss.item(), grad_norm.item()


# ════════ [10/16] def train(model, train_loader, val_loader, config): ════════


import os
from tqdm import tqdm

def train(model, train_loader, val_loader, config):
    device = config["device"]
    model = model.to(device)

    # المحسّن
    optimizer = create_optimizer(
        model,
        lr=config["max_lr"],
        weight_decay=config["weight_decay"],
    )

    # Scaler (FP16 فقط)
    scaler = GradScaler(enabled=config["use_fp16"])

    # إنشاء مجلد الحفظ
    os.makedirs(config["output_dir"], exist_ok=True)

    # استئناف إذا لزم
    start_step = 0
    start_epoch = 0
    if config.get("resume_from") and os.path.exists(config["resume_from"]):
        start_step, start_epoch = load_checkpoint(
            config["resume_from"], model, optimizer, scaler
        )
        print(f"Resumed from step {start_step}, epoch {start_epoch}")

    model.train()
    global_step = start_step
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for micro_step, batch in enumerate(pbar):
            # ضبط معدل التعلم
            lr = update_lr(optimizer, global_step, config)

            # خطوة تدريب
            loss, grad_norm = train_step(
                model, batch, optimizer, scaler, config, global_step
            )

            # هل حان وقت تحديث الأوزان؟
            if (micro_step + 1) % config["accum_steps"] == 0:
                global_step += 1

                # تسجيل
                if global_step % config["log_every"] == 0:
                    pbar.set_postfix({
                        "loss": f"{loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "gnorm": f"{grad_norm:.2f}",
                        "step": global_step,
                    })

                # تقييم
                if global_step % config["eval_every"] == 0:
                    val_loss = evaluate(model, val_loader, config)
                    print(f"\nStep {global_step} | Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
                    model.train()

                    # حفظ أفضل نموذج
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            os.path.join(config["output_dir"], "best.pt"),
                            model, optimizer, scaler, global_step, epoch, config,
                        )

                # حفظ دوري
                if global_step % config["save_every"] == 0:
                    save_checkpoint(
                        os.path.join(config["output_dir"], f"step_{global_step}.pt"),
                        model, optimizer, scaler, global_step, epoch, config,
                    )

                # إيقاف
                if global_step >= config["max_steps"]:
                    print(f"Reached max_steps ({config['max_steps']}). Stopping.")
                    break

        if global_step >= config["max_steps"]:
            break

    # حفظ نهائي
    save_checkpoint(
        os.path.join(config["output_dir"], "final.pt"),
        model, optimizer, scaler, global_step, config["epochs"], config,
    )
    print("Training complete.")


# ════════ [11/16] if (micro_step + 1) % config["accum_steps"] != 0: ════════


# في نهاية الحقبة، إذا كانت هناك تدرجات متراكمة
if (micro_step + 1) % config["accum_steps"] != 0:
    if config["use_fp16"]:
        scaler.unscale_(optimizer)
        clip_gradients(model, config["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
    else:
        clip_gradients(model, config["max_grad_norm"])
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# ════════ [12/16] @torch.no_grad() ════════


@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch[0].to(config["device"], non_blocking=True)
        targets = batch[1].to(config["device"], non_blocking=True)

        dtype = torch.bfloat16 if config["use_bf16"] else torch.float16
        with autocast(device_type="cuda", dtype=dtype, enabled=config["use_amp"]):
            logits, loss = model(input_ids, targets)

        # تراكم الخسارة
        n_tokens = (targets != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ════════ [13/16] def save_checkpoint(path, model, optimizer, scaler, step, epoch, confi ════════


def save_checkpoint(path, model, optimizer, scaler, step, epoch, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # حفظ آمن: احفظ إلى ملف مؤقت، ثم أعد التسمية
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(path, model, optimizer=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return checkpoint["step"], checkpoint["epoch"]


# ════════ [14/16] class MetricsLogger: ════════


class MetricsLogger:
    def __init__(self, log_file="metrics.csv"):
        self.log_file = log_file
        self.history = []
        with open(log_file, "w") as f:
            f.write("step,epoch,train_loss,val_loss,lr,grad_norm,tokens_per_sec\n")

    def log(self, step, epoch, train_loss, val_loss, lr, grad_norm, tokens_per_sec):
        row = f"{step},{epoch},{train_loss},{val_loss},{lr},{grad_norm},{tokens_per_sec}\n"
        with open(self.log_file, "a") as f:
            f.write(row)
        self.history.append({
            "step": step, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "lr": lr, "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
        })

    def plot(self):
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(steps, [h["train_loss"] for h in self.history], label="Train")
        axes[0, 0].plot(steps, [h["val_loss"] for h in self.history], label="Val")
        axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend(); axes[0, 0].set_title("Loss Curves")

        axes[0, 1].plot(steps, [h["lr"] for h in self.history])
        axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("LR")
        axes[0, 1].set_title("Learning Rate Schedule")

        axes[1, 0].plot(steps, [h["grad_norm"] for h in self.history])
        axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("Grad Norm")
        axes[1, 0].set_title("Gradient Norm")
        axes[1, 0].axhline(y=10, color="r", linestyle="--", label="Threshold")
        axes[1, 0].legend()

        axes[1, 1].plot(steps, [h["tokens_per_sec"] for h in self.history])
        axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("Tokens/sec")
        axes[1, 1].set_title("Throughput")

        plt.tight_layout()
        plt.savefig("training_metrics.png", dpi=150)
        print("Saved training_metrics.png")


# ════════ [15/16] assert not torch.isnan(input_ids).any() ════════


# تحقق من المدخلات
assert not torch.isnan(input_ids).any()
assert not torch.isinf(input_ids).any()

# تحقق من الخسارة نفسها
if torch.isnan(loss):
    print("Loss is NaN!")
    print(f"Logits min/max: {logits.min().item()}, {logits.max().item()}")
    print(f"Targets range: {targets.min().item()}, {targets.max().item()}")


# ════════ [16/16] single_batch = next(iter(train_loader)) ════════


# اختبر Overfit a single batch
single_batch = next(iter(train_loader))
for i in range(100):
    loss, _ = train_step(model, single_batch, optimizer, scaler, config, i)
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss:.4f}")
# يجب أن تنخفض الخسارة إلى قريب من 0 خلال 100 خطوة
