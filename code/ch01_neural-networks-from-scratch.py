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
