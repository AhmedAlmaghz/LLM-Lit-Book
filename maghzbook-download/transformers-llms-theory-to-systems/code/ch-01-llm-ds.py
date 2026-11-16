#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — الشبكات العصبية من الصفر — كيف تتعلم الآلة أن “تخمّن” بشكل أفضل
Transformers and Large Language Models — Neural Networks From Scratch

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-01-llm-ds (lesson 1)
Code files: 31 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-1
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
# [002] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# ===== 1. توليد بيانات اصطناعية (مشكلة XOR) =====
# XOR: مخرج = 1 إذا اختلف المدخلان، و0 إذا اتفقا
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# ===== 2. تهيئة الأوزان =====
np.random.seed(42)
n_input = 2
n_hidden = 4
n_output = 1

# تهيئة Xavier/Glorot: تقلل مشاكل التدرج
W1 = np.random.randn(n_input, n_hidden) * np.sqrt(1.0 / n_input)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * np.sqrt(1.0 / n_hidden)
b2 = np.zeros((1, n_output))

# ===== 3. دوال التنشيط =====
def sigmoid(z):
    # clip لتجنب overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    # a هو المخرج بعد sigmoid
    return a * (1 - a)

# ===== 4. الانتشار الأمامي =====
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# ===== 5. دالة الخسارة (Binary Cross-Entropy) =====
def bce_loss(y_true, y_pred):
    eps = 1e-8  # لمنع log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ===== 6. الانتشار العكسي =====
def backward(X, y, z1, a1, z2, a2, W2):
    m = X.shape[0]  # حجم الدفعة

    # تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة
    dz2 = a2 - y                          # (m, 1)
    dW2 = a1.T @ dz2 / m                  # (n_hidden, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # تدرج الطبقة المخفية
    da1 = dz2 @ W2.T                      # (m, n_hidden)
    dz1 = da1 * sigmoid_derivative(a1)    # (m, n_hidden)
    dW1 = X.T @ dz1 / m                   # (n_input, n_hidden)
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# ===== 7. حلقة التدريب =====
learning_rate = 0.5
epochs = 5000

for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
    loss = bce_loss(y, a2)
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# ===== 8. النتائج =====
_, _, _, preds = forward(X, W1, b1, W2, b2)
print("\nالتنبؤات بعد التدريب:")
for i in range(len(X)):
    print(f"  {X[i]} -> {preds[i][0]:.4f} (المتوقع: {y[i][0]})")

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def numerical_gradient(f, x, eps=1e-6):
    """تقريب التدرج عدديًا باستخدام الفروق المركزية."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fx_plus = f(x)
        x[idx] = old - eps
        fx_minus = f(x)
        x[idx] = old
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        it.iternext()
    return grad

# استخدامه للتحقق من W1
def loss_for_W1(W1_val):
    W1_tmp = W1_val.copy()
    _, _, _, a2 = forward(X, W1_tmp, b1, W2, b2)
    return bce_loss(y, a2)

# التدرج التحليلي (الذي حسبناه في الكود السابق)
z1_, a1_, z2_, a2_ = forward(X, W1, b1, W2, b2)
dW1_analytic, _, _, _ = backward(X, y, z1_, a1_, z2_, a2_, W2)

# التدرج العددي
dW1_numeric = numerical_gradient(loss_for_W1, W1.copy())

diff = np.abs(dW1_analytic - dW1_numeric).max()
print(f"أقصى فرق بين التدرجين: {diff:.2e}")
# إذا كان الفرق < 1e-6، فالتدرج سليم

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim

# البيانات
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# النموذج
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = XORNet()

# الخسارة والمحسّن
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# التدريب
for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()       # Autograd يحسب التدرجات
    optimizer.step()      # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# النتائج
with torch.no_grad():
    print("\nالتنبؤات:")
    print(model(X).detach().numpy().round(3))

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # استخدم نفس المحول!

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# أو
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=epochs * len(dataloader)
)

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):
    # ... حلقة التدريب ...
    train_loss = ...  # متوسط خسارة التدريب
    val_loss, val_acc = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

# رسم
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"], label="Validation")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.set_title("Loss Curves")

ax2.plot(history["val_acc"])
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Validation Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# في PyTorch
self.dropout = nn.Dropout(p=0.3)  # 30% من الخلايا تُعطّل عشوائيًا
# ...
x = self.dropout(x)

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
model.train()   # قبل التدريب
# ... training loop ...

model.eval()    # قبل التقييم
with torch.no_grad():
    pass  # ... evaluation ...

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# في PyTorch: عبر المحسّن
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
x_norm = (x - x.mean(dim=0)) / x.std(dim=0)

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
x_norm = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=total_steps,
    pct_start=0.03,       # 3% warmup
    anneal_strategy="cos",
)

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
lrs = np.logspace(-6, 0, 100)
losses = []
for lr in lrs:
    model = rebuild_model()  # أعد التهيئة
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # درّب لخطوة أو خطوتين
    losses.append(train_a_few_steps(model, optimizer))

# ابحث عن النقطة التي تبدأ فيها الخسارة في الارتفاع
# اختر معدلًا أقل بقليل من تلك النقطة

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
weights = torch.tensor([1.0, 10.0])  # الفئة النادرة أثقل
criterion = nn.CrossEntropyLoss(weight=weights)

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = build_model()
    train_loader = make_loader(X[train_idx], y[train_idx])
    val_loader = make_loader(X[val_idx], y[val_idx])

    train(model, train_loader)
    score = evaluate(model, val_loader)
    scores.append(score)
    print(f"Fold {fold+1}: {score:.4f}")

print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
# تشغيل DDP (مبسط)
# كل عملية (process) تعمل على GPU مختلفة
torch.distributed.init_process_group(backend="nccl")
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# ────────────────────────────────────────────────────────────────────────
# [025] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, use_orig_params=True)

# ────────────────────────────────────────────────────────────────────────
# [026] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.detach().data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"grad_norm: {total_norm:.4f}")

# ────────────────────────────────────────────────────────────────────────
# [027] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
with torch.autograd.set_detect_anomaly(True):
    loss.backward()

# ────────────────────────────────────────────────────────────────────────
# [028] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
assert not torch.isnan(batch["input_ids"]).any(), "NaN in input!"
assert not torch.isinf(batch["input_ids"]).any(), "Inf in input!"

# ────────────────────────────────────────────────────────────────────────
# [029] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ────────────────────────────────────────────────────────────────────────
# [030] code/notebooks/part01_foundations-neural-nets-to-deep-learning.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [031] code/ch01_neural-networks-from-scratch.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
الفصل 1: الشبكات العصبية من الصفر — كيف تتعلم الآلة أن “تخمّن” بشكل أفضل

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الأول · الفصل 1 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch01_neural-networks-from-scratch.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch01_neural-networks-from-scratch.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32) ════════


import numpy as np

# ===== 1. توليد بيانات اصطناعية (مشكلة XOR) =====
# XOR: مخرج = 1 إذا اختلف المدخلان، و0 إذا اتفقا
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# ===== 2. تهيئة الأوزان =====
np.random.seed(42)
n_input = 2
n_hidden = 4
n_output = 1

# تهيئة Xavier/Glorot: تقلل مشاكل التدرج
W1 = np.random.randn(n_input, n_hidden) * np.sqrt(1.0 / n_input)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * np.sqrt(1.0 / n_hidden)
b2 = np.zeros((1, n_output))

# ===== 3. دوال التنشيط =====
def sigmoid(z):
    # clip لتجنب overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    # a هو المخرج بعد sigmoid
    return a * (1 - a)

# ===== 4. الانتشار الأمامي =====
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# ===== 5. دالة الخسارة (Binary Cross-Entropy) =====
def bce_loss(y_true, y_pred):
    eps = 1e-8  # لمنع log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ===== 6. الانتشار العكسي =====
def backward(X, y, z1, a1, z2, a2, W2):
    m = X.shape[0]  # حجم الدفعة

    # تدرج الخسارة بالنسبة لمخرجات الطبقة الأخيرة
    dz2 = a2 - y                          # (m, 1)
    dW2 = a1.T @ dz2 / m                  # (n_hidden, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # تدرج الطبقة المخفية
    da1 = dz2 @ W2.T                      # (m, n_hidden)
    dz1 = da1 * sigmoid_derivative(a1)    # (m, n_hidden)
    dW1 = X.T @ dz1 / m                   # (n_input, n_hidden)
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# ===== 7. حلقة التدريب =====
learning_rate = 0.5
epochs = 5000

for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
    loss = bce_loss(y, a2)
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# ===== 8. النتائج =====
_, _, _, preds = forward(X, W1, b1, W2, b2)
print("\nالتنبؤات بعد التدريب:")
for i in range(len(X)):
    print(f"  {X[i]} -> {preds[i][0]:.4f} (المتوقع: {y[i][0]})")


# ════════ [2/6] def numerical_gradient(f, x, eps=1e-6): ════════


def numerical_gradient(f, x, eps=1e-6):
    """تقريب التدرج عدديًا باستخدام الفروق المركزية."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fx_plus = f(x)
        x[idx] = old - eps
        fx_minus = f(x)
        x[idx] = old
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        it.iternext()
    return grad

# استخدامه للتحقق من W1
def loss_for_W1(W1_val):
    W1_tmp = W1_val.copy()
    _, _, _, a2 = forward(X, W1_tmp, b1, W2, b2)
    return bce_loss(y, a2)

# التدرج التحليلي (الذي حسبناه في الكود السابق)
z1_, a1_, z2_, a2_ = forward(X, W1, b1, W2, b2)
dW1_analytic, _, _, _ = backward(X, y, z1_, a1_, z2_, a2_, W2)

# التدرج العددي
dW1_numeric = numerical_gradient(loss_for_W1, W1.copy())

diff = np.abs(dW1_analytic - dW1_numeric).max()
print(f"أقصى فرق بين التدرجين: {diff:.2e}")
# إذا كان الفرق < 1e-6، فالتدرج سليم


# ════════ [3/6] X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32 ════════


import torch
import torch.nn as nn
import torch.optim as optim

# البيانات
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# النموذج
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = XORNet()

# الخسارة والمحسّن
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# التدريب
for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()       # Autograd يحسب التدرجات
    optimizer.step()      # تحديث الأوزان

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# النتائج
with torch.no_grad():
    print("\nالتنبؤات:")
    print(model(X).detach().numpy().round(3))


# ════════ [4/6] scaler = StandardScaler() ════════


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # استخدم نفس المحول!


# ════════ [5/6] scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoc ════════


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# أو
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=epochs * len(dataloader)
)


# ════════ [6/6] def set_seed(seed=42): ════════


import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
