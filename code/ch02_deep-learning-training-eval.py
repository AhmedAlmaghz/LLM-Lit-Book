# -*- coding: utf-8 -*-
"""
الفصل 2: مفاهيم التعلّم العميق للتدريب والتقييم — كيف تعرف أن نموذجك “يتعلم” فعلًا؟

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الأول · الفصل 2 · عدد كتل بايثون في هذا الملف: 14

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch02_deep-learning-training-eval.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch02_deep-learning-training-eval.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/14] history = {"train_loss": [], "val_loss": [], "val_acc": []} ════════


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


# ════════ [2/14] self.dropout = nn.Dropout(p=0.3)  # 30% من الخلايا تُعطّل عشوائيًا ════════


# في PyTorch
self.dropout = nn.Dropout(p=0.3)  # 30% من الخلايا تُعطّل عشوائيًا
# ...
x = self.dropout(x)


# ════════ [3/14] model.train()   # قبل التدريب ════════


model.train()   # قبل التدريب
# ... training loop ...

model.eval()    # قبل التقييم
with torch.no_grad():
    pass  # ... evaluation ...


# ════════ [4/14] optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01 ════════


# في PyTorch: عبر المحسّن
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


# ════════ [5/14] class EarlyStopping: ════════


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


# ════════ [6/14] x_norm = (x - x.mean(dim=0)) / x.std(dim=0) ════════


x_norm = (x - x.mean(dim=0)) / x.std(dim=0)


# ════════ [7/14] x_norm = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=Tr ════════


x_norm = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)


# ════════ [8/14] class RMSNorm(nn.Module): ════════


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ════════ [9/14] def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5): ════════


import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


# ════════ [10/14] scheduler = optim.lr_scheduler.OneCycleLR( ════════


scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=total_steps,
    pct_start=0.03,       # 3% warmup
    anneal_strategy="cos",
)


# ════════ [11/14] lrs = np.logspace(-6, 0, 100) ════════


lrs = np.logspace(-6, 0, 100)
losses = []
for lr in lrs:
    model = rebuild_model()  # أعد التهيئة
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # درّب لخطوة أو خطوتين
    losses.append(train_a_few_steps(model, optimizer))

# ابحث عن النقطة التي تبدأ فيها الخسارة في الارتفاع
# اختر معدلًا أقل بقليل من تلك النقطة


# ════════ [12/14] weights = torch.tensor([1.0, 10.0])  # الفئة النادرة أثقل ════════


weights = torch.tensor([1.0, 10.0])  # الفئة النادرة أثقل
criterion = nn.CrossEntropyLoss(weight=weights)


# ════════ [13/14] def focal_loss(logits, targets, alpha=0.25, gamma=2.0): ════════


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()


# ════════ [14/14] kf = KFold(n_splits=5, shuffle=True, random_state=42) ════════


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
