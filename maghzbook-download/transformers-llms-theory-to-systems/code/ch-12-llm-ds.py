#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — بناء LLM من الصفر — تخطيط المشروع وإعداد البيئة (مع كود كامل لكل خطوة)
Transformers and Large Language Models — Build Llm Planning

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-12-llm-ds (lesson 12)
Code files: 86 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-1
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
# [002] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def tokens_per_word(text, tokenizer):
    encoded = tokenizer.encode(text)
    words = len(text.split())
    return len(encoded.ids) / words

# اختبر على عينة
sample = texts[0]
print(f"Tokens/word: {tokens_per_word(sample, tokenizer):.2f}")
# إذا كانت النسبة > 2.0، فقد تحتاج إلى تحسين التطبيع أو حجم المفردات

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# اختبار
model = MiniGPT(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, max_seq_len=512)
x = torch.randint(0, 32000, (2, 128))
logits, loss = model(x, x)
print(logits.shape)  # torch.Size([2, 128, 32000])
print(loss)          # خسارة أولية (يجب أن تكون قريبة من log(32000) ≈ 10.37)

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import re
import unicodedata

def normalize_arabic(
    text: str,
    unify_alef: bool = True,
    unify_teh_marbuta: bool = True,
    remove_diacritics: bool = True,
    remove_tatweel: bool = True,
    normalize_numerals: bool = True,
) -> str:
    """
    تطبيع النص العربي بشكل متسق.
    """
    if not text:
        return text

    # 1. تطبيع Unicode NFKC (يوحد الأشكال المركبة)
    text = unicodedata.normalize("NFKC", text)

    # 2. إزالة التطويل
    if remove_tatweel:
        text = re.sub(r"\u0640", "", text)

    # 3. إزالة التشكيل
    if remove_diacritics:
        # harakat: fatha, damma, kasra, sukun, shadda, tanwin
        text = re.sub(r"[\u064B-\u0652]", "", text)

    # 4. توحيد الهمزات
    if unify_alef:
        text = re.sub(r"[إأآٱ]", "ا", text)
        # الهمزة على واو/ياء (مثل: مسئول -> مسؤول)
        text = text.replace("ئ", "ء").replace("ؤ", "ء")
        # الهمزة المفردة تبقى
        text = text.replace("ء", "ء")  # لا تغيير، لكن للتوضيح

    # 5. توحيد التاء المربوطة
    if unify_teh_marbuta:
        text = text.replace("ة", "ه")

    # 6. توحيد الأرقام العربية-الهندية
    if normalize_numerals:
        arabic_indic = "٠١٢٣٤٥٦٧٨٩"
        western = "0123456789"
        trans = str.maketrans(arabic_indic, western)
        text = text.translate(trans)

    # 7. إزالة علامات الترقيم العربية الخاصة وتحويلها
    text = text.replace("؟", "?").replace("؛", ";").replace("،", ",")

    # 8. توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()

    return text

# اختبار
samples = [
    "إِنَّ الْحَمْدَ لِلَّهِ",
    "مسئولية",
    "أحمد في المدرسة",
    "الطالبُ مجتهدٌ",
]
for s in samples:
    print(f"Original: {s}")
    print(f"Normalized: {normalize_arabic(s)}")
    print()

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm

def train_bpe_tokenizer(
    texts,
    vocab_size=32000,
    min_frequency=2,
    save_prefix="arabic_bpe",
):
    """
    تدريب Byte-Level BPE على نصوص عربية (أو مختلطة).
    """
    tokenizer = ByteLevelBPETokenizer()

    def corpus_iterator():
        for text in texts:
            if text and text.strip():
                yield text

    tokenizer.train_from_iterator(
        corpus_iterator(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    tokenizer.save_model(".", save_prefix)
    print(f"Saved: {save_prefix}-vocab.json, {save_prefix}-merges.txt")
    return tokenizer

# مثال: تحميل بيانات عربية (استبدل بالمصدر الخاص بك)
def load_arabic_texts(limit=100_000):
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ar",
        split="train", streaming=True
    )
    texts = []
    for item in tqdm(islice(ds, limit), total=limit):
        text = normalize_arabic(item["text"])
        if len(text) > 50:
            texts.append(text)
    return texts

texts = load_arabic_texts(limit=50_000)
tokenizer = train_bpe_tokenizer(texts, vocab_size=32_000)

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import sentencepiece as spm
import tempfile
import os

def train_sentencepiece_unigram(
    texts,
    vocab_size=32000,
    model_prefix="arabic_sp",
    character_coverage=1.0,
):
    # كتابة النصوص إلى ملف مؤقت (SentencePiece يحتاج إلى ملف)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for text in texts:
            f.write(text + "\n")
        temp_path = f.name

    try:
        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",       # أو "bpe"
            character_coverage=character_coverage,
            byte_fallback=True,         # لتمثيل أي رمز دون <unk>
            normalization_rule_name="identity",  # نطبع نحن مسبقًا
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        print(f"Saved: {model_prefix}.model, {model_prefix}.vocab")
    finally:
        os.unlink(temp_path)

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    return sp

sp = train_sentencepiece_unigram(texts, vocab_size=32_000)

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
def compute_tokenizer_metrics(texts, tokenizer, is_sp=False):
    """
    tokenizer: إما HuggingFace tokenizer أو SentencePieceProcessor.
    is_sp: True إذا كان SentencePiece.
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    oov_count = 0
    total_count = 0

    for text in texts:
        words = text.split()
        if not words:
            continue

        if is_sp:
            ids = tokenizer.encode(text)
            # SentencePiece: unk_id
            oov = sum(1 for i in ids if i == tokenizer.unk_id())
        else:
            encoding = tokenizer.encode(text)
            ids = encoding.ids
            oov = sum(1 for i in ids if i == tokenizer.token_to_id("<unk>"))

        total_tokens += len(ids)
        total_words += len(words)
        total_chars += len(text)
        oov_count += oov
        total_count += 1

    fertility = total_tokens / max(total_words, 1)
    compression = total_chars / max(total_tokens, 1)
    oov_rate = oov_count / max(total_tokens, 1)

    return {
        "fertility": fertility,
        "compression": compression,
        "oov_rate": oov_rate,
        "total_tokens": total_tokens,
        "total_words": total_words,
    }

# مثال
metrics = compute_tokenizer_metrics(texts[:1000], tokenizer)
print(f"Fertility: {metrics['fertility']:.3f} tokens/word")
print(f"Compression: {metrics['compression']:.2f} chars/token")
print(f"OOV rate: {metrics['oov_rate']*100:.2f}%")

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def build_token_stream(texts, tokenizer, eos_id, is_sp=False):
    """يبني مجرى وحدات مسطحًا مع فواصل بين المستندات."""
    all_ids = []
    for text in texts:
        if is_sp:
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(eos_id)  # فاصل
    return all_ids

def split_into_blocks(ids, block_size):
    """يقسم المجرى إلى كتل متتالية."""
    blocks = []
    for i in range(0, len(ids) - block_size, block_size):
        blocks.append(ids[i : i + block_size])
    return blocks

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
def best_fit_pack(documents, block_size, eos_id=None):
    """
    documents: قائمة قوائم من معرفات الوحدات (كل مستند قائمة).
    يجمع المستندات القصيرة في كتل دون تقطيعها.
    المستندات الأطول من block_size تُقطع.
    """
    # افصل المستندات الطويلة
    chunks = []
    for doc in documents:
        if len(doc) <= block_size:
            chunks.append(doc)
        else:
            for i in range(0, len(doc), block_size):
                chunks.append(doc[i : i + block_size])

    # رتب تنازليًا حسب الطول (FFD)
    chunks.sort(key=len, reverse=True)

    bins = []  # قائمة الكتل
    for chunk in chunks:
        placed = False
        # ابحث عن كتلة بها مساحة كافية
        for b in bins:
            if len(b) + len(chunk) + (1 if eos_id else 0) <= block_size:
                if eos_id is not None:
                    b.append(eos_id)
                b.extend(chunk)
                placed = True
                break
        if not placed:
            new_bin = []
            if eos_id is not None:
                new_bin.append(eos_id)
            new_bin.extend(chunk)
            bins.append(new_bin)

    return bins

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
def sliding_window_chunks(doc_ids, block_size, stride=None):
    """تقسيم مستند طويل إلى أجزاء متداخلة."""
    stride = stride or block_size // 2
    chunks = []
    for i in range(0, len(doc_ids), stride):
        chunk = doc_ids[i : i + block_size]
        if len(chunk) < block_size // 2 and chunks:
            break  # تجاهل الجزء الأخير الصغير جدًا
        chunks.append(chunk)
        if i + block_size >= len(doc_ids):
            break
    return chunks

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

class PackedLMDataset(Dataset):
    def __init__(self, blocks):
        """
        blocks: قائمة قوائم معرفات (كل كتلة بطول block_size تقريبًا).
        """
        self.input_ids = []
        self.target_ids = []

        for block in blocks:
            if len(block) < 2:
                continue
            # إذا كانت الكتلة أقصر من block_size، احشها (أو تجاهلها)
            # هنا نفترض أن كل الكتل بنفس الطول تقريبًا.
            input_chunk = block[:-1]
            target_chunk = block[1:]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
def create_dataloader(blocks, batch_size=8, num_workers=4, pin_memory=True):
    dataset = PackedLMDataset(blocks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,      # لتحميل البيانات بالتوازي
        pin_memory=pin_memory,        # لتسريع النقل إلى GPU
        drop_last=True,
        persistent_workers=num_workers > 0,  # يحتفظ بالعمال بين الحقب
    )

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import IterableDataset

class StreamingLMDataset(IterableDataset):
    def __init__(self, text_iterator, tokenizer, block_size):
        self.text_iterator = text_iterator
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        for text in self.text_iterator:
            ids = self.tokenizer.encode(text).ids
            buffer.extend(ids)
            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
                buffer = buffer[self.block_size :]

# ────────────────────────────────────────────────────────────────────────
# [025] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import GroupShuffleSplit
import hashlib

def grouped_split(texts, groups, test_size=0.15, val_size=0.15, seed=42):
    """
    texts: قائمة النصوص.
    groups: قائمة معرفات المجموعة (نفس الطول).
    """
    # أولًا: افصل الاختبار
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(texts, groups=groups))

    # ثانيًا: افصل التحقق من التدريب
    train_val_texts = [texts[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx]

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=seed
    )
    train_idx_rel, val_idx_rel = next(
        gss_val.split(train_val_texts, groups=train_val_groups)
    )

    train_idx = [train_val_idx[i] for i in train_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]

    return train_idx, val_idx, test_idx

# ────────────────────────────────────────────────────────────────────────
# [026] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
import hashlib

def text_fingerprint(text):
    # تطبيع بسيط
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def deduplicate(texts, groups=None):
    seen = set()
    unique_texts = []
    unique_groups = []
    for i, text in enumerate(texts):
        fp = text_fingerprint(text)
        if fp in seen:
            continue
        seen.add(fp)
        unique_texts.append(text)
        if groups:
            unique_groups.append(groups[i])
    return unique_texts, (unique_groups if groups else None)

# ────────────────────────────────────────────────────────────────────────
# [027] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    من LLaMA، Qwen، DeepSeek.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # حساب في FP32 لتجنب عدم الاستقرار في BF16/FP16
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight

# ────────────────────────────────────────────────────────────────────────
# [028] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
norm = RMSNorm(64)
x = torch.randn(2, 10, 64)
out = norm(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
print(f"RMS of output: {out.float().pow(2).mean(-1).sqrt().mean():.4f}")
# يجب أن يكون قريبًا من 1.0 (بسبب weight=1 وعدم وجود انحياز)

# ────────────────────────────────────────────────────────────────────────
# [029] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    يُطبق على Q وK فقط، وليس على V.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len

        # الترددات: 1 / (base^(2i/dim)) لكل زوج
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # حساب cos و sin مسبقًا لكل المواضع
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len).float()
        freqs = torch.outer(pos, self.inv_freq)  # (seq, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        """
        q, k: (batch, n_heads, seq_len, head_dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # تدوير الأزواج: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        return q_rot, k_rot

# ────────────────────────────────────────────────────────────────────────
# [030] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
rope = RoPE(dim=64, max_seq_len=512)
q = torch.randn(2, 8, 128, 64)  # (B, H, L, D)
k = torch.randn(2, 8, 128, 64)
q_rot, k_rot = rope(q, k, seq_len=128)
print(f"q_rot shape: {q_rot.shape}")  # يجب أن يكون (2, 8, 128, 64)

# اختبار خاصية: الضرب النقطي يعتمد على الفرق في الموضع
q_test = torch.zeros(1, 1, 2, 64)
k_test = torch.zeros(1, 1, 2, 64)
q_test[0, 0, 0, :2] = torch.tensor([1.0, 0.0])   # موضع 0
k_test[0, 0, 1, :2] = torch.tensor([1.0, 0.0])   # موضع 1
q_rot, k_rot = rope(q_test, k_test, seq_len=2)
score = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
print(f"Score with relative position 1: {score:.4f}")
# يجب أن يعتمد على الزاوية، وليس على الموضع المطلق

# ────────────────────────────────────────────────────────────────────────
# [031] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        x: (batch, seq_len, d_model)
        attention_mask: (batch, seq_len) - 1 للكلمات الحقيقية، 0 للحشو
        """
        B, L, _ = x.shape

        # الإسقاطات
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # تطبيق RoPE على Q وK
        q, k = self.rope(q, k, seq_len=L)

        # استخدام F.scaled_dot_product_attention (FlashAttention تلقائيًا)
        # is_causal=True يطبق القناع السببي تلقائيًا
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # لا حاجة، is_causal كافٍ
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # out: (B, H, L, head_dim)

        # دمج الرؤوس
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)

# ────────────────────────────────────────────────────────────────────────
# [032] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
attn = CausalSelfAttention(d_model=256, n_heads=8, max_seq_len=512)
x = torch.randn(2, 64, 256)
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # (2, 64, 256)

# اختبار القناع السببي: تغيير رمز مستقبلي لا يجب أن يؤثر على المخرجات السابقة
x1 = x.clone()
x2 = x.clone()
x2[:, 10, :] = torch.randn(256)  # غيّر الرمز رقم 10
out1 = attn(x1)
out2 = attn(x2)
# يجب أن تكون المخرجات للمواضع 0..9 متطابقة
diff_before = (out1[:, :10] - out2[:, :10]).abs().max()
diff_after = (out1[:, 10:] - out2[:, 10:]).abs().max()
print(f"Diff before pos 10: {diff_before:.6f}")  # يجب أن يكون ~0
print(f"Diff after pos 10: {diff_after:.6f}")   # يجب أن يكون > 0

# ────────────────────────────────────────────────────────────────────────
# [033] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU FFN.
    d_ff الافتراضي = 8/3 * d_model (مع تقريب لمضاعف 256).
    """
    def __init__(self, d_model: int, d_ff: int = None, bias: bool = False):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # تقريب لأقرب مضاعف لـ 256 (لتحسين أداء العتاد)
            d_ff = ((d_ff + 255) // 256) * 256

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ────────────────────────────────────────────────────────────────────────
# [034] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
ffn = SwiGLU(d_model=256)
x = torch.randn(2, 64, 256)
out = ffn(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"d_ff: {ffn.gate_proj.out_features}")  # يجب أن يكون ~682 (8/3 * 256)

# ────────────────────────────────────────────────────────────────────────
# [035] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    كتلة محوّل حديثة (Pre-LN): RMSNorm + Attention + Residual + RMSNorm + FFN + Residual.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        d_ff: int = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # التيار المتبقي: نضيف، لا نستبدل
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# ────────────────────────────────────────────────────────────────────────
# [036] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
block = TransformerBlock(d_model=256, n_heads=8)
x = torch.randn(2, 64, 256)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # نفس الشكل

# ────────────────────────────────────────────────────────────────────────
# [037] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
class MiniGPT(nn.Module):
    """
    نموذج لغوي بنمط LLaMA مصغّر.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # الكتل
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, max_seq_len, dropout, d_ff
            )
            for _ in range(n_layers)
        ])

        # التطبيع النهائي
        self.norm_f = RMSNorm(d_model)

        # رأس اللغة
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط الأوزان
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # التهيئة
        self.apply(self._init_weights)

        # تهيئة خاصة للـ o_proj و down_proj (لتقليل الانفجار)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        input_ids: (batch, seq_len)
        targets: (batch, seq_len) - اختياري
        """
        B, L = input_ids.shape

        # Embedding
        x = self.token_embedding(input_ids)  # (B, L, d_model)

        # المرور عبر الكتل
        for block in self.blocks:
            x = block(x, attention_mask)

        # التطبيع النهائي
        x = self.norm_f(x)

        # رأس اللغة
        logits = self.lm_head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None,
    ):
        """توليد متسلسل مع KV cache ضمني (بسيط هنا، بدونه)."""
        self.eval()
        for _ in range(max_new_tokens):
            # اقتطاع إذا تجاوزنا max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits[mask] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

# ────────────────────────────────────────────────────────────────────────
# [038] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
def test_model():
    """اختبار شامل للنموذج."""
    vocab_size = 1000
    batch_size = 2
    seq_len = 64

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=512,
    )

    # اختبار الأشكال
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"Logits shape: {logits.shape}")  # (2, 64, 1000)
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected initial loss (log V): {math.log(vocab_size):.4f}")

    # اختبار التوليد
    generated = model.generate(
        input_ids[:1, :10],
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )
    print(f"Generated shape: {generated.shape}")  # (1, 30)

    # عدد المعاملات
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_emb = n_params - model.token_embedding.weight.numel()
    print(f"Total parameters: {n_params:,}")
    print(f"Parameters (no embedding, due to tying): {n_params_no_emb:,}")

    return model

model = test_model()

# ────────────────────────────────────────────────────────────────────────
# [039] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
def count_parameters(model):
    """حساب دقيق للمعاملات، مع تفصيل."""
    total = 0
    breakdown = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        # تصنيف
        if "embedding" in name:
            key = "embedding"
        elif "attn" in name:
            key = "attention"
        elif "ffn" in name:
            key = "ffn"
        elif "norm" in name:
            key = "norm"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            key = "other"

        breakdown[key] = breakdown.get(key, 0) + n

    print(f"Total parameters: {total:,}")
    print("Breakdown:")
    for key, n in breakdown.items():
        print(f"  {key}: {n:,} ({n/total*100:.1f}%)")

    return total, breakdown

# لـ 124M (مثل GPT-2 small)
model_124m = MiniGPT(
    vocab_size=50257,
    d_model=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=1024,
)
count_parameters(model_124m)

# ────────────────────────────────────────────────────────────────────────
# [040] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [041] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
def compute_loss_with_z(logits, targets, z_coef=1e-4, ignore_index=-100):
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    log_z = torch.logsumexp(logits, dim=-1)
    z_loss = (log_z ** 2).mean()
    return ce + z_coef * z_loss

# ────────────────────────────────────────────────────────────────────────
# [042] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),      # 0.95 هو الافتراضي في LLaMA/GPT-3
    eps=1e-8,
    weight_decay=0.1,        # 0.1 شائع في المحوّلات
)

# ────────────────────────────────────────────────────────────────────────
# [043] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [044] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [045] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [046] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
def clip_gradients(model, max_norm=1.0):
    """يقص التدرجات ويعيد المعيار الأصلي (للمراقبة)."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        error_if_nonfinite=True,  # يرفع خطأ إذا كان هناك NaN/Inf
    )
    return total_norm

# ────────────────────────────────────────────────────────────────────────
# [047] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
use_fp16 = not use_bf16 and torch.cuda.is_available()

# ────────────────────────────────────────────────────────────────────────
# [048] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [049] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [050] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [051] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [052] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [053] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [054] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
# تحقق من المدخلات
assert not torch.isnan(input_ids).any()
assert not torch.isinf(input_ids).any()

# تحقق من الخسارة نفسها
if torch.isnan(loss):
    print("Loss is NaN!")
    print(f"Logits min/max: {logits.min().item()}, {logits.max().item()}")
    print(f"Targets range: {targets.min().item()}, {targets.max().item()}")

# ────────────────────────────────────────────────────────────────────────
# [055] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
# اختبر Overfit a single batch
single_batch = next(iter(train_loader))
for i in range(100):
    loss, _ = train_step(model, single_batch, optimizer, scaler, config, i)
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss:.4f}")
# يجب أن تنخفض الخسارة إلى قريب من 0 خلال 100 خطوة

# ────────────────────────────────────────────────────────────────────────
# [056] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer

def format_sft_example(example, tokenizer):
    """
    example: {"instruction": "...", "response": "..."}
    يُرجع input_ids و labels مع -100 على رموز التعليمات.
    """
    # بناء النص الكامل باستخدام chat template
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # ترميز النص الكامل
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # لبناء labels، نكرر الترميز على التعليمات فقط لمعرفة طولها
    prompt_messages = [{"role": "user", "content": example["instruction"]}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ────────────────────────────────────────────────────────────────────────
# [057] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,          # alpha = r
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",            # لا تدرب bias
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# مثال: trainable params: 2,162,688 || all params: 1,238,468,608 || trainable%: 0.17%

# ────────────────────────────────────────────────────────────────────────
# [058] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4
    bnb_4bit_use_double_quant=True,      # التكميم المزدوج
    bnb_4bit_compute_dtype=torch.bfloat16, # الحساب في BF16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto",
)

# يجب استدعاء هذا قبل get_peft_model!
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)

# ────────────────────────────────────────────────────────────────────────
# [059] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./sft_lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    packing=False,                # ابدأ بدون packing
    dataset_text_field="text",    # أو استخدم formatting_func
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./sft_lora_final")

# ────────────────────────────────────────────────────────────────────────
# [060] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

def train_lora(model, train_dataset, tokenizer, config):
    device = "cuda"
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=0.01,
    )

    def collate_fn(batch):
        # batch: قائمة من dicts مع input_ids و labels
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = x["input_ids"]
            labels[i, :n] = x["labels"]
            attention_mask[i, :n] = 1
        return input_ids, labels, attention_mask

    loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                        shuffle=True, collate_fn=collate_fn)

    total_steps = len(loader) * config["epochs"] // config["accum_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(config["epochs"]):
        for step, (input_ids, labels, attention_mask) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config["accum_steps"]

            loss.backward()

            if (step + 1) % config["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item() * config['accum_steps']:.4f}")

    return model

# ────────────────────────────────────────────────────────────────────────
# [061] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
from peft import PeftModel

# تحميل النموذج الأساسي (بدون تكميم إذا أردت الدمج)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# تحميل المحولات
peft_model = PeftModel.from_pretrained(base_model, "./sft_lora_final")

# دمج
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# ────────────────────────────────────────────────────────────────────────
# [062] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    """
    نموذج مكافأة: يأخذ (prompt + response)، ويعيد درجة عددية.
    """
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B"):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # (batch,)

def reward_model_loss(chosen_rewards, rejected_rewards):
    """
    خسارة Bradley-Terry: نريد chosen_rewards > rejected_rewards.
    """
    return -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

# ────────────────────────────────────────────────────────────────────────
# [063] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (batch,)
    policy_rejected_logps: torch.Tensor,  # (batch,)
    ref_chosen_logps: torch.Tensor,       # (batch,)
    ref_rejected_logps: torch.Tensor,     # (batch,)
    beta: float = 0.1,
):
    """
    حساب خسارة DPO.
    logps: مجموع لوغاريتمات الاحتمالات لتسلسل الإجابة (بعد قناع -100 على التعليمات).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # للتحليل: هامش التفضيل الضمني
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    margin = (chosen_rewards - rejected_rewards).mean()

    return loss, margin

# ────────────────────────────────────────────────────────────────────────
# [064] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
def compute_logps(model, input_ids, labels, attention_mask):
    """
    يحسب مجموع لوغاريتمات الاحتمالات للرموز التي ليست -100.
    """
    with torch.no_grad() if model.training is False else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # logits: (batch, seq, vocab)
    # نزيح: logits[:, :-1] تتنبأ بـ labels[:, 1:]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # log_softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # نجمع فقط حيث labels != -100
    mask = (labels != -100).float()
    # gathered: (batch, seq)
    gathered = log_probs.gather(dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)  # (batch,)

# ────────────────────────────────────────────────────────────────────────
# [065] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# بيانات تفضيل: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="preference_data.jsonl", split="train")

training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # الافتراضي الآمن
    learning_rate=5e-7,          # أقل بكثير من SFT!
    num_train_epochs=1,          # 1-2 حقب فقط
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    loss_type="sigmoid",         # DPO القياسي
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,              # TRL سينشئ مرجعًا تلقائيًا من النموذج الأساسي
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./dpo_final")

# ────────────────────────────────────────────────────────────────────────
# [066] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
# في TRL: استخدم ORPOTrainer
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    output_dir="./orpo_output",
    learning_rate=8e-6,          # أعلى من DPO قليلًا
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
)
trainer = ORPOTrainer(model=model, args=config, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [067] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
from trl import KTOTrainer, KTOConfig

config = KTOConfig(
    output_dir="./kto_output",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
)
trainer = KTOTrainer(model=model, args=config, train_dataset=kto_dataset, tokenizer=tokenizer)
trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [068] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./grpo_output",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,           # عدد الإجابات لكل تعليمات
    bf16=True,
)
trainer = GRPOTrainer(model=model, args=config, train_dataset=grpo_dataset, tokenizer=tokenizer)
trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [069] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
def format_preference_example(example, tokenizer):
    """
    example: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(example["chosen"], add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(example["rejected"], add_special_tokens=False)["input_ids"]

    # input_ids: prompt + chosen (أو rejected)
    # labels: -100 على prompt، ثم رموز الإجابة
    chosen_input = prompt_ids + chosen_ids
    chosen_labels = [-100] * len(prompt_ids) + chosen_ids

    rejected_input = prompt_ids + rejected_ids
    rejected_labels = [-100] * len(prompt_ids) + rejected_ids

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_input,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input,
        "rejected_labels": rejected_labels,
    }

# ────────────────────────────────────────────────────────────────────────
# [070] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def greedy_decode(model, input_ids, max_new_tokens=50, eos_token_id=None):
    """
    Greedy decoding: اختر الرمز الأعلى احتمالًا في كل خطوة.
    """
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]  # (batch, vocab)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return generated

# ────────────────────────────────────────────────────────────────────────
# [071] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-71
# ────────────────────────────────────────────────────────────────────────
def beam_search(model, input_ids, beam_size=4, max_new_tokens=50, eos_token_id=None):
    """
    Beam search مبسط (batch=1).
    """
    model.eval()
    device = input_ids.device

    # كل عنصر: (sequence_ids, score)
    beams = [(input_ids, 0.0)]
    finished = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            candidates = []

            for seq, score in beams:
                logits = model(seq).logits[:, -1, :]  # (1, vocab)
                log_probs = F.log_softmax(logits, dim=-1)

                # احصل على أعلى beam_size رموز
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    token = topk_ids[0, i].unsqueeze(0).unsqueeze(0)  # (1,1)
                    new_seq = torch.cat([seq, token], dim=-1)
                    new_score = score + topk_log_probs[0, i].item()

                    if eos_token_id is not None and token.item() == eos_token_id:
                        finished.append((new_seq, new_score))
                    else:
                        candidates.append((new_seq, new_score))

            if not candidates:
                break

            # احتفظ بأفضل beam_size
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]

    # اختر أفضل مسار منتهي أو غير منتهٍ
    all_seqs = finished + beams
    all_seqs.sort(key=lambda x: x[1], reverse=True)
    return all_seqs[0][0]

# ────────────────────────────────────────────────────────────────────────
# [072] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-72
# ────────────────────────────────────────────────────────────────────────
def apply_temperature(logits, temperature=1.0):
    if temperature == 0:
        return logits  # Greedy
    return logits / temperature

# ────────────────────────────────────────────────────────────────────────
# [073] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-73
# ────────────────────────────────────────────────────────────────────────
def top_k_filtering(logits, top_k=50):
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    # احصل على قيم top-k
    values, _ = torch.topk(logits, top_k, dim=-1)
    # احتفظ فقط بالقيم >= أصغر قيمة في top-k
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

# ────────────────────────────────────────────────────────────────────────
# [074] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-74
# ────────────────────────────────────────────────────────────────────────
def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # ابحث عن أول موضع يتجاوز p
    sorted_mask = cumulative_probs > top_p
    # أزح القناع لليمين، حتى نحتفظ بالرمز الذي تجاوز p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # ضع -inf على الرموز المقنعة
    indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits

# ────────────────────────────────────────────────────────────────────────
# [075] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-75
# ────────────────────────────────────────────────────────────────────────
def min_p_filtering(logits, min_p=0.05):
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))

# ────────────────────────────────────────────────────────────────────────
# [076] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-76
# ────────────────────────────────────────────────────────────────────────
def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    if penalty == 1.0:
        return logits
    for i in range(logits.size(0)):
        for token_id in set(input_ids[i].tolist()):
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty
    return logits

# ────────────────────────────────────────────────────────────────────────
# [077] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-77
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=None,
    min_p=None,
    repetition_penalty=1.0,
    eos_token_id=None,
):
    """
    توليد متسلسل مع دعم: greedy, temperature, top-k, top-p, min-p, repetition penalty.
    """
    model.eval()
    generated = input_ids.clone()
    device = input_ids.device

    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :]  # (batch, vocab)

        # 1. Repetition Penalty
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, generated, repetition_penalty)

        # 2. Greedy (temperature = 0)
        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            continue

        # 3. Temperature
        logits = logits / temperature

        # 4. Min-p (قبل Top-p، لأن كلاهما قد يُطبق)
        if min_p is not None and min_p > 0:
            logits = min_p_filtering(logits, min_p)

        # 5. Top-k
        if top_k is not None and top_k > 0:
            logits = top_k_filtering(logits, top_k)

        # 6. Top-p
        if top_p is not None and top_p < 1.0:
            logits = top_p_filtering(logits, top_p)

        # 7. أعد Softmax، واسحب عينة
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated

# ────────────────────────────────────────────────────────────────────────
# [078] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-78
# ────────────────────────────────────────────────────────────────────────
# افترض أن لدينا model و tokenizer
prompt = "كان يا ما كان"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

configs = [
    {"name": "Greedy", "temperature": 0, "top_k": None, "top_p": None},
    {"name": "Top-k", "temperature": 0.8, "top_k": 50, "top_p": None},
    {"name": "Top-p", "temperature": 0.8, "top_p": 0.9, "top_k": None},
    {"name": "Min-p", "temperature": 0.8, "min_p": 0.05, "top_k": None, "top_p": None},
]

for cfg in configs:
    output = generate(model, input_ids, max_new_tokens=30, **cfg)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[{cfg['name']}] {text}")

# ────────────────────────────────────────────────────────────────────────
# [079] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-79
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens=50, eos_token_id=None):
    """
    توليد مع KV Cache (يستخدم past_key_values في HuggingFace).
    """
    model.eval()
    generated = input_ids.clone()

    # الخطوة الأولى: prefill (معالجة كل الـ prompt)
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        # الخطوات التالية: decode (رمز واحد فقط في كل مرة)
        outputs = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    return generated

# ────────────────────────────────────────────────────────────────────────
# [080] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-80
# ────────────────────────────────────────────────────────────────────────
def prompt_lookup_proposal(input_ids, n=3, num_pred_tokens=3):
    """
    يقترح رموزًا بناءً على تطابق n-gram في الـ prompt.
    """
    input_list = input_ids[0].tolist()
    if len(input_list) < n:
        return []

    # آخر n رموز
    last_n = tuple(input_list[-n:])

    # ابحث عن تطابق في الـ prompt
    for i in range(len(input_list) - n, 0, -1):
        if tuple(input_list[i - n : i]) == last_n:
            # اقترح الرموز التالية
            proposal = input_list[i : i + num_pred_tokens]
            if proposal:
                return proposal
    return []

# ────────────────────────────────────────────────────────────────────────
# [081] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-81
# ────────────────────────────────────────────────────────────────────────
# في vLLM: تشغيل نموذج بـ FP8 مع KV Cache FP8
# vllm serve meta-llama/Llama-3-70B --quantization fp8 --kv-cache-dtype fp8

# ────────────────────────────────────────────────────────────────────────
# [082] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-82
# ────────────────────────────────────────────────────────────────────────
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import AWQModifier

# تكميم AWQ (W4A16)
recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

oneshot(
    model="meta-llama/Llama-3.1-8B",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./llama-3.1-8b-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)

# ────────────────────────────────────────────────────────────────────────
# [083] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-83
# ────────────────────────────────────────────────────────────────────────
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# FP8 W8A8 (أوزان وتنشيطات 8 بت)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",   # أو "FP8" للثابت
    ignore=["lm_head"],
)

oneshot(
    model="meta-llama/Llama-3.1-8B",
    recipe=recipe,
    output_dir="./llama-3.1-8b-fp8",
)

# ────────────────────────────────────────────────────────────────────────
# [084] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-84
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0, ignore_index=-100):
    """
    خسارة التقطير: مزيج من Cross-Entropy (مع البيانات الحقيقية) وKL (مع توزيع المعلم).
    """
    # Cross-Entropy مع البيانات الحقيقية
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    # KL divergence مع تنعيم درجة الحرارة
    # نحتاج إلى قناع لتجاهل الحشو
    mask = (labels != ignore_index).float().view(-1, 1)
    student_log_probs = F.log_softmax(student_logits.view(-1, student_logits.size(-1)) / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / T, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1, keepdim=True)

    kl = (kl * mask).sum() / mask.sum()
    kd_loss = kl * (T ** 2)

    return alpha * ce + (1 - alpha) * kd_loss

# ────────────────────────────────────────────────────────────────────────
# [085] code/notebooks/part04_build-llm-from-scratch.ipynb#cell-85
# ────────────────────────────────────────────────────────────────────────
def prune_layers(model, layers_to_prune):
    """
    يحذف طبقات محددة من نموذج MiniGPT.
    layers_to_prune: قائمة مؤشرات الطبقات (مثل [4, 5, 6]).
    """
    new_blocks = []
    for i, block in enumerate(model.blocks):
        if i not in layers_to_prune:
            new_blocks.append(block)
    model.blocks = torch.nn.ModuleList(new_blocks)
    print(f"Pruned {len(layers_to_prune)} layers. Remaining: {len(model.blocks)}")
    return model

# ────────────────────────────────────────────────────────────────────────
# [086] code/ch12_build-llm-planning.py
# ────────────────────────────────────────────────────────────────────────
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
