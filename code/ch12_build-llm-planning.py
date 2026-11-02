# -*- coding: utf-8 -*-
"""
الفصل 12: بناء LLM من الصفر — تخطيط المشروع وإعداد البيئة (مع كود كامل لكل خطوة)

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 12 · عدد كتل بايثون في هذا الملف: 13

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch12_build-llm-planning.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch12_build-llm-planning.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/13] print(f"Python: {sys.version}") ════════


import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU. Training will be slow.")


# ════════ [2/13] def load_arabic_text(max_samples=None): ════════


import os
from datasets import load_dataset

def load_arabic_text(max_samples=None):
    """
    يحمل نصوصًا عربية. استبدل هذا بأي مجموعة بيانات لديك.
    """
    # مثال: تحميل ويكيبيديا العربية
    ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # استخراج النصوص
    texts = [item["text"] for item in ds]
    return texts

def clean_text(text):
    """تنظيف بسيط: إزالة المسافات الزائدة والأسطر الفارغة."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # إزالة المسافات المتعددة
    return text.strip()

# تحميل وتنظيف
texts = load_arabic_text(max_samples=10000)  # ابدأ بعينة صغيرة
texts = [clean_text(t) for t in texts if len(t) > 50]  # تجاهل النصوص القصيرة جدًا

print(f"Number of documents: {len(texts)}")
print(f"Sample: {texts[0][:200]}")


# ════════ [3/13] def train_tokenizer(texts, vocab_size=32000, save_path="arabic_bpe.jso ════════


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(texts, vocab_size=32000, save_path="arabic_bpe.json"):
    # تهيئة tokenizer BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # التطبيع: NFKC (توحيد الأحرف)
    tokenizer.normalizer = Sequence([NFKC()])

    # التقسيم المسبق: على المسافات وعلامات الترقيم
    tokenizer.pre_tokenizer = Whitespace()

    # فك الترميز: تحويل `Ġ` إلى مسافة
    tokenizer.decoder = decoders.BPEDecoder()

    # المدرب
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
        show_progress=True,
    )

    # التدريب
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # الحفظ
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

# تدريب
tokenizer = train_tokenizer(texts, vocab_size=32000)


# ════════ [4/13] def tokens_per_word(text, tokenizer): ════════


def tokens_per_word(text, tokenizer):
    encoded = tokenizer.encode(text)
    words = len(text.split())
    return len(encoded.ids) / words

# اختبر على عينة
sample = texts[0]
print(f"Tokens/word: {tokens_per_word(sample, tokenizer):.2f}")
# إذا كانت النسبة > 2.0، فقد تحتاج إلى تحسين التطبيع أو حجم المفردات


# ════════ [5/13] class LanguageModelDataset(Dataset): ════════


import torch
from torch.utils.data import Dataset, DataLoader

class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, block_size):
        """
        token_ids: قائمة مسطحة بكل الرموز في النص (بعد الترميز).
        block_size: طول كل عينة.
        """
        self.input_ids = []
        self.target_ids = []

        # نقسم إلى كتل متتالية
        for i in range(0, len(token_ids) - block_size, block_size):
            input_chunk = token_ids[i : i + block_size]
            target_chunk = token_ids[i + 1 : i + block_size + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(texts, tokenizer, block_size=512, batch_size=8, shuffle=True):
    # ترميز كل النصوص ودمجها
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(tokenizer.token_to_id("</s>"))  # فاصل بين المستندات

    dataset = LanguageModelDataset(all_ids, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)

# مثال
train_loader = create_dataloader(texts, tokenizer, block_size=512, batch_size=8)
batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape)  # torch.Size([8, 512]) torch.Size([8, 512])


# ════════ [6/13] class RMSNorm(nn.Module): ════════


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        pos = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k, seq_len):
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        return q_rot, k_rot

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_k, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        q, k = self.rope(q, k, L)  # تدوير Q و K

        # استخدام F.scaled_dot_product_attention مع قناع سببي
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.dropout(self.W_o(out))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, max_seq_len=2048):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8,
                 d_ff=None, max_seq_len=2048, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: مشاركة أوزان Embedding والمخرج
        self.lm_head.weight = self.tok_emb.weight

        # تهيئة
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        x = self.tok_emb(idx)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ════════ [7/13] model = MiniGPT(vocab_size=32000, d_model=256, n_layers=4, n_heads=4,  ════════


# اختبار
model = MiniGPT(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, max_seq_len=512)
x = torch.randint(0, 32000, (2, 128))
logits, loss = model(x, x)
print(logits.shape)  # torch.Size([2, 128, 32000])
print(loss)          # خسارة أولية (يجب أن تكون قريبة من log(32000) ≈ 10.37)


# ════════ [8/13] def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5): ════════


import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ════════ [9/13] def train(model, dataloader, config): ════════


from torch.amp import autocast, GradScaler

def train(model, dataloader, config):
    device = config["device"]
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=config["use_fp16"])

    model.train()
    global_step = 0
    accum_steps = config["accum_steps"]

    for epoch in range(config["epochs"]):
        for step, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # ضبط معدل التعلم
            lr = get_lr(global_step, config["warmup_steps"],
                        config["max_steps"], config["max_lr"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward مع Mixed Precision
            with autocast(device_type="cuda", dtype=torch.bfloat16 if config["use_bf16"] else torch.float16):
                logits, loss = model(input_ids, targets)

            # Gradient Accumulation: اقسم الخسارة
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # تسجيل
                if global_step % config["log_every"] == 0:
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item() * accum_steps:.4f}, LR: {lr:.2e}")

                if global_step >= config["max_steps"]:
                    break

        if global_step >= config["max_steps"]:
            break

    return model


# ════════ [10/13] config = { ════════


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 3,
    "max_steps": 5000,
    "max_lr": 3e-4,
    "min_lr": 1e-5,
    "warmup_steps": 200,
    "accum_steps": 4,
    "use_bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "use_fp16": not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
    "log_every": 50,
}


# ════════ [11/13] def save_checkpoint(path, model, optimizer, scaler, step, epoch, confi ════════


def save_checkpoint(path, model, optimizer, scaler, step, epoch, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"Resumed from step {checkpoint['step']}")
    return checkpoint["step"], checkpoint["epoch"]


# ════════ [12/13] @torch.no_grad() ════════


@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits, loss = model(input_ids, targets)
        # loss هو متوسط على كل الرموز
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl

# مثال
val_loader = create_dataloader(texts[-1000:], tokenizer, block_size=512, batch_size=8)
ppl = compute_perplexity(model, val_loader, config["device"])
print(f"Validation Perplexity: {ppl:.2f}")


# ════════ [13/13] @torch.no_grad() ════════


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(config["device"])

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model_max_len:])  # اقتطاع إذا لزم
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

print(generate(model, tokenizer, "كان يا ما كان", max_new_tokens=30))
