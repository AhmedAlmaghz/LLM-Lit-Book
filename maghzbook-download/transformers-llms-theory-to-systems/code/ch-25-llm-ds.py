#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — مدخل إلى النماذج متعددة الوسائط (Multimodal) — عندما يفتح النموذج عينيه
Transformers and Large Language Models — Multimodal Intro

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-25-llm-ds (lesson 25)
Code files: 34 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part06_multimodal-llms.ipynb#cell-1
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
# [002] code/notebooks/part06_multimodal-llms.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from transformers import AutoModel, AutoImageProcessor

def load_vision_encoder(model_name="google/siglip-so400m-patch14-384"):
    """
    يحمل Vision Encoder من HuggingFace.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, processor

def encode_image(image, model, processor, device="cuda"):
    """
    image: PIL.Image
    يُرجع: (num_patches, vision_dim)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (batch, num_patches, vision_dim)
    return outputs.last_hidden_state

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part06_multimodal-llms.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch.nn as nn

class MLPProjector(nn.Module):
    """
    Projector بسيط: طبقتان خطيتان مع GELU.
    """
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        """
        x = self.fc1(vision_features)
        x = self.act(x)
        x = self.fc2(x)
        return x  # (batch, num_patches, llm_dim)

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part06_multimodal-llms.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PerceiverResampler(nn.Module):
    """
    يضغط الرموز البصرية إلى عدد ثابت من الرموز.
    """
    def __init__(self, vision_dim: int, llm_dim: int, num_latents: int = 64, num_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, llm_dim) * 0.02)
        self.proj_in = nn.Linear(vision_dim, llm_dim) if vision_dim != llm_dim else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim),
        )

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        يُرجع: (batch, num_latents, llm_dim)
        """
        x = self.proj_in(vision_features)  # (B, P, D)
        B = x.size(0)

        latents = self.latents.expand(B, -1, -1)  # (B, L, D)

        # Cross-attention: latents (query) تنتبه إلى vision_features (key/value)
        h = self.norm1(latents)
        h, _ = self.cross_attn(h, x, x)
        latents = latents + h

        # FFN
        latents = latents + self.ffn(self.norm2(latents))
        return latents

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part06_multimodal-llms.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleVLM(nn.Module):
    """
    VLM بسيط: Vision Encoder + Projector + LLM.
    """
    def __init__(
        self,
        vision_encoder,
        llm,
        projector: nn.Module,
        image_token_id: int,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector
        self.image_token_id = image_token_id

        # جمّد Vision Encoder و LLM
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        """
        input_ids: (batch, seq) — يحتوي على image_token_id في موضع الصورة
        pixel_values: (batch, 3, H, W)
        """
        # 1. ترميز الصورة
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        # vision_features: (batch, num_patches, vision_dim)

        # 2. إسقاط إلى فضاء LLM
        visual_tokens = self.projector(vision_features)
        # visual_tokens: (batch, num_patches, llm_dim)

        # 3. تضمين النص
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # text_embeds: (batch, seq, llm_dim)

        # 4. استبدال image_token_id بتضمينات الصورة
        # نفترض أن كل صورة لها موضع واحد في input_ids (image_token_id)
        batch_size = input_ids.size(0)
        new_embeds = []
        for i in range(batch_size):
            # ابحث عن موضع image_token
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                new_embeds.append(text_embeds[i])
                continue

            pos = img_pos[0].item()
            # قبل الصورة، الصورة، بعد الصورة
            before = text_embeds[i, :pos]            # (pos, D)
            after = text_embeds[i, pos+1:]           # (seq-pos-1, D)
            combined = torch.cat([before, visual_tokens[i], after], dim=0)
            new_embeds.append(combined)

        # ملاحظة: الأطوال مختلفة الآن! نحتاج إلى حشو أو التعامل مع ذلك.
        # في التطبيق الحقيقي، نستخدم attention_mask ويتم التعامل مع الأطوال.
        # للتبسيط، نفترض أن كل العينات لها نفس العدد من image_tokens.
        inputs_embeds = torch.stack(new_embeds, dim=0)

        # 5. مرر عبر LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part06_multimodal-llms.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

# تحميل نموذج VLM جاهز
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# تحميل صورة
image = Image.open("path/to/image.jpg")

# بناء المحادثة
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "صف هذه الصورة بالتفصيل."},
        ],
    }
]

# معالجة
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# توليد
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# فك الترميز
response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(response)

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part06_multimodal-llms.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

class ImageTextDataset(Dataset):
    """
    بيانات: (صورة، تعليق).
    """
    def __init__(self, pairs, processor, image_token="<image>"):
        self.pairs = pairs
        self.processor = processor
        self.image_token = image_token

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, caption = self.pairs[idx]
        # بناء النص: <image> + التعليق
        text = f"{self.image_token}\n{caption}"
        inputs = self.processor(
            text=text, images=image,
            return_tensors="pt", padding="max_length", truncation=True, max_length=512,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # labels = input_ids، مع -100 على رموز الصورة
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train_projector(vlm, dataset, epochs=1, lr=1e-3, batch_size=4):
    """
    يدرب Projector فقط (LLM و Vision Encoder مجمدان).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # فقط معاملات Projector
    optimizer = torch.optim.AdamW(vlm.projector.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)

    vlm.projector.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(vlm.llm.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = vlm(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return vlm

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part06_multimodal-llms.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def evaluate_vqa(model, processor, dataset, max_samples=100):
    """
    تقييم بسيط على VQA.
    """
    correct = 0
    total = 0
    for i, (image, question, answer) in enumerate(dataset):
        if i >= max_samples:
            break
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        pred = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if pred.lower() == answer.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part06_multimodal-llms.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def test_arabic_ocr(model, processor, image):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "اقرأ كل النص العربي في هذه الصورة، وأعد النص كما هو."},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part06_multimodal-llms.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LLaVAProjector(nn.Module):
    """
    MLP بسيط: Linear → GELU → Linear.
    هذا هو Projector المستخدم في LLaVA-1.5.
    """
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        """
        return self.fc2(self.act(self.fc1(vision_features)))

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part06_multimodal-llms.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class QFormer(nn.Module):
    """
    Q-Former مبسط: 32 استعلامًا + Cross-Attention + Self-Attention.
    """
    def __init__(
        self,
        vision_dim: int = 1024,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries

        # الاستعلامات القابلة للتعلم
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # إسقاط رقع الصورة
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # طبقات Q-Former
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "norm2": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                "norm3": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        يُرجع: (batch, num_queries, hidden_dim)
        """
        B = vision_features.size(0)
        x = self.vision_proj(vision_features)  # (B, P, H)
        q = self.queries.expand(B, -1, -1)      # (B, Q, H)

        for layer in self.layers:
            # Self-Attention على الاستعلامات
            h = layer["norm1"](q)
            h, _ = layer["self_attn"](h, h, h)
            q = q + h

            # Cross-Attention: استعلامات تنتبه إلى رقع الصورة
            h = layer["norm2"](q)
            h, _ = layer["cross_attn"](h, x, x)
            q = q + h

            # FFN
            q = q + layer["ffn"](layer["norm3"](q))

        return q

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part06_multimodal-llms.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class FlamingoCrossAttention(nn.Module):
    """
    Cross-Attention بين LLM والذاكرة البصرية.
    """
    def __init__(self, llm_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(llm_dim)
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # بوابة تان

    def forward(self, x, visual_memory, visual_mask=None):
        """
        x: (batch, seq, llm_dim) - تمثيل النص
        visual_memory: (batch, num_visual_tokens, llm_dim)
        """
        h = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=h, key=visual_memory, value=visual_memory,
            key_padding_mask=visual_mask,
        )
        # بوابة تان: في البداية، 0، فالنموذج لا يتأثر
        return x + torch.tanh(self.gate) * attn_out

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part06_multimodal-llms.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

def ask_qwen(image: Image.Image, question: str) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# مثال: OCR عربي
image = Image.open("arabic_sign.jpg")
print(ask_qwen(image, "اقرأ كل النص العربي في هذه الصورة، وأعد النص كما هو."))

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part06_multimodal-llms.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_name = "microsoft/Florence-2-large"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda")

def florence_caption(image: Image.Image) -> str:
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    return processor.batch_decode(out, skip_special_tokens=True)[0]

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part06_multimodal-llms.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModel, AutoImageProcessor,
)
import torch.nn as nn

def load_components(device="cuda"):
    # 1. LLM
    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.bfloat16, device_map=device
    )

    # 2. Vision Encoder
    vision_name = "google/siglip-so400m-patch14-384"
    vision_processor = AutoImageProcessor.from_pretrained(vision_name)
    vision_encoder = AutoModel.from_pretrained(
        vision_name, torch_dtype=torch.bfloat16
    ).to(device)

    return llm, tokenizer, vision_encoder, vision_processor

llm, tokenizer, vision_encoder, vision_processor = load_components()
print(f"LLM hidden dim: {llm.config.hidden_size}")          # 1536
print(f"Vision hidden dim: {vision_encoder.config.hidden_size}")  # 1152

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part06_multimodal-llms.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part06_multimodal-llms.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
class PooledProjector(nn.Module):
    """
    Projector مع AvgPool: يقلل الرموز من 729 إلى 144 (عبر pool_size=3).
    """
    def __init__(self, vision_dim, llm_dim, pool_size=3):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.pool = nn.AvgPool2d(pool_size)  # 27x27 -> 9x9

    def forward(self, x):
        # x: (B, 729, 1152) -> (B, 27, 27, 1152) -> (B, 1152, 27, 27)
        B, N, D = x.shape
        side = int(N ** 0.5)
        x = x.view(B, side, side, D).permute(0, 3, 1, 2)
        x = self.pool(x)  # (B, D, 9, 9)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, D)  # (B, 81, D)
        return self.fc2(self.act(self.fc1(x)))

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part06_multimodal-llms.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    """
    المرحلة 1: (صورة، تعليق).
    """
    def __init__(self, pairs, processor, tokenizer, image_token="<image>", max_length=512):
        self.pairs = pairs  # list of (image_path, caption)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length

        # أضف <image> كرمز خاص
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if self.image_token_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, caption = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")

        # النص: <image>\n{تعليق}
        text = f"{self.image_token}\n{caption}"

        # معالجة الصورة
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # ترميز النص
        text_inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length,
            truncation=True, padding="max_length"
        )
        input_ids = text_inputs["input_ids"][0]
        attention_mask = text_inputs["attention_mask"][0]

        # labels: -100 على الحشو، و-100 على <image> (لا نتنبأ بالصورة)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part06_multimodal-llms.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class ArabicVLM(nn.Module):
    def __init__(self, vision_encoder, llm, projector):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector

        # جمّد Vision Encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        # جمّد LLM (سنستخدم LoRA لاحقًا)
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        # 1. ترميز الصورة
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=pixel_values)
            vision_features = vision_out.last_hidden_state  # (B, 729, 1152)

        # 2. إسقاط
        visual_tokens = self.projector(vision_features)  # (B, 729, llm_dim)

        # 3. تضمين النص
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq, llm_dim)

        # 4. استبدال <image> برموز بصرية
        B, seq_len, D = text_embeds.shape
        num_visual = visual_tokens.size(1)

        # نفترض أن كل عينة لها <image> واحد
        new_embeds = []
        new_attention = []
        new_labels = []

        for i in range(B):
            # موضع <image>
            img_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_positions) == 0:
                new_embeds.append(text_embeds[i])
                new_attention.append(attention_mask[i] if attention_mask is not None else torch.ones(seq_len, device=input_ids.device))
                new_labels.append(labels[i] if labels is not None else torch.full((seq_len,), -100, device=input_ids.device))
                continue

            pos = img_positions[0].item()
            before = text_embeds[i, :pos]
            after = text_embeds[i, pos+1:]

            combined = torch.cat([before, visual_tokens[i], after], dim=0)
            new_embeds.append(combined)

            # attention_mask: 1 للصورة، ثم النص
            if attention_mask is not None:
                attn_before = attention_mask[i, :pos]
                attn_visual = torch.ones(num_visual, device=input_ids.device)
                attn_after = attention_mask[i, pos+1:]
                new_attention.append(torch.cat([attn_before, attn_visual, attn_after]))
            else:
                new_attention.append(torch.ones(combined.size(0), device=input_ids.device))

            # labels: -100 للصورة
            if labels is not None:
                lab_before = labels[i, :pos]
                lab_visual = torch.full((num_visual,), -100, device=input_ids.device)
                lab_after = labels[i, pos+1:]
                new_labels.append(torch.cat([lab_before, lab_visual, lab_after]))

        # حشو إلى أقصى طول (لأن الأطوال قد تختلف)
        max_len = max(e.size(0) for e in new_embeds)
        padded_embeds = torch.zeros(B, max_len, D, device=input_ids.device, dtype=text_embeds.dtype)
        padded_attn = torch.zeros(B, max_len, device=input_ids.device, dtype=torch.long)
        padded_labels = torch.full((B, max_len), -100, device=input_ids.device, dtype=torch.long)

        for i in range(B):
            L = new_embeds[i].size(0)
            padded_embeds[i, :L] = new_embeds[i]
            padded_attn[i, :L] = new_attention[i]
            if labels is not None:
                padded_labels[i, :L] = new_labels[i]

        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attn,
            labels=padded_labels if labels is not None else None,
        )
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part06_multimodal-llms.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

def train_alignment(vlm, dataset, epochs=1, batch_size=2, lr=1e-3, device="cuda"):
    """
    المرحلة 1: تدريب Projector فقط.
    """
    vlm = vlm.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(vlm.projector.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)

    vlm.projector.train()
    vlm.llm.eval()
    vlm.vision_encoder.eval()

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = vlm(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    return vlm

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part06_multimodal-llms.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model

def add_lora_to_llm(llm):
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(llm, config)

# استخدم LoRA
vlm.llm = add_lora_to_llm(vlm.llm)
vlm.llm.print_trainable_parameters()

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part06_multimodal-llms.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    المرحلة 2: (صورة، سؤال، إجابة).
    """
    def __init__(self, samples, processor, tokenizer, image_token="<image>", max_length=1024):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, question, answer = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # بناء النص: <image>\nسؤال: {question}\nجواب: {answer}
        prompt = f"{self.image_token}\nسؤال: {question}\nجواب: "
        full = prompt + answer

        # labels: -100 على prompt (بما فيها <image>)، ثم رموز الإجابة
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, max_length=self.max_length, truncation=True)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # تأكد من الطول
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part06_multimodal-llms.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
def train_sft(vlm, dataset, epochs=3, batch_size=1, lr=2e-4, device="cuda", accum_steps=8):
    vlm = vlm.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # فقط معاملات Projector + LoRA
    trainable = list(vlm.projector.parameters()) + [p for p in vlm.llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs // accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, 20, total_steps)

    vlm.projector.train()
    vlm.llm.train()

    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = vlm(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * accum_steps:.4f}")

    return vlm

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part06_multimodal-llms.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answer(vlm, image: Image.Image, question: str, processor, tokenizer, max_new_tokens=100):
    vlm.eval()
    device = next(vlm.parameters()).device

    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt = f"<image>\nسؤال: {question}\nجواب: "
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # احصل على التضمينات
    vision_features = vlm.vision_encoder(pixel_values=pixel_values).last_hidden_state
    visual_tokens = vlm.projector(vision_features)

    text_embeds = vlm.llm.get_input_embeddings()(input_ids)

    # استبدال <image>
    img_pos = (input_ids[0] == vlm.image_token_id).nonzero(as_tuple=True)[0][0].item()
    combined = torch.cat([
        text_embeds[:, :img_pos],
        visual_tokens,
        text_embeds[:, img_pos+1:],
    ], dim=1)

    # توليد
    output_ids = vlm.llm.generate(
        inputs_embeds=combined,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ────────────────────────────────────────────────────────────────────────
# [025] code/notebooks/part06_multimodal-llms.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
def evaluate_ocr(vlm, samples, processor, tokenizer, max_samples=20):
    """
    samples: list of (image_path, expected_text)
    """
    correct = 0
    total = 0
    for image_path, expected in samples[:max_samples]:
        image = Image.open(image_path).convert("RGB")
        pred = generate_answer(vlm, image, "اقرأ كل النص في الصورة.", processor, tokenizer)
        # تنظيف بسيط
        pred = pred.strip()
        if expected.strip() in pred or pred in expected.strip():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

# ────────────────────────────────────────────────────────────────────────
# [026] code/notebooks/part06_multimodal-llms.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
from peft import PeftModel

def merge_and_save(vlm, save_path="./arabic_vlm"):
    # دمج LoRA في LLM
    merged_llm = vlm.llm.merge_and_unload()

    # حفظ LLM
    merged_llm.save_pretrained(f"{save_path}/llm")
    vlm.tokenizer.save_pretrained(f"{save_path}/llm")

    # حفظ Projector
    torch.save(vlm.projector.state_dict(), f"{save_path}/projector.pt")

    # حفظ Vision Encoder (أو استخدم اسم النموذج الأصلي)
    vlm.vision_encoder.save_pretrained(f"{save_path}/vision_encoder")
    vlm.vision_processor.save_pretrained(f"{save_path}/vision_encoder")

    print(f"Saved to {save_path}")

# ────────────────────────────────────────────────────────────────────────
# [027] code/notebooks/part06_multimodal-llms.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# تحميل النموذج المدمج بـ 4-bit
llm_4bit = AutoModelForCausalLM.from_pretrained(
    f"{save_path}/llm",
    quantization_config=bnb_config,
    device_map="auto",
)

# ────────────────────────────────────────────────────────────────────────
# [028] code/notebooks/part06_multimodal-llms.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
def llm_judge(image_path, question, answer, judge_model="gpt-4o-mini"):
    """
    يستخدم GPT-4o لتقييم الإجابة.
    """
    from openai import OpenAI
    import base64

    client = OpenAI()
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""قيّم الإجابة التالية على مقياس 1-5:
- 5: دقيقة، كاملة، مستمدة من الصورة
- 3: صحيحة جزئيًا
- 1: خاطئة أو هلوسة

السؤال: {question}
الإجابة: {answer}

أعد رقمًا واحدًا فقط (1-5)."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=10,
    )
    return int(response.choices[0].message.content.strip())

# ────────────────────────────────────────────────────────────────────────
# [029] code/notebooks/part06_multimodal-llms.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
def evaluate_pope(vlm, samples, processor, tokenizer):
    """
    samples: list of (image_path, object_name, exists: bool)
    """
    correct = 0
    total = 0
    false_positives = 0  # قال "نعم" لغير موجود

    for image_path, obj, exists in samples:
        image = Image.open(image_path).convert("RGB")
        question = f"هل يوجد {obj} في الصورة؟ أجب بـ 'نعم' أو 'لا' فقط."
        answer = generate_answer(vlm, image, question, processor, tokenizer).strip()

        pred_yes = "نعم" in answer
        if pred_yes == exists:
            correct += 1
        elif pred_yes and not exists:
            false_positives += 1
        total += 1

    accuracy = correct / total
    fp_rate = false_positives / total
    return {"accuracy": accuracy, "false_positive_rate": fp_rate}

# ────────────────────────────────────────────────────────────────────────
# [030] code/notebooks/part06_multimodal-llms.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
def detect_hallucination_in_caption(image_path, caption, judge_model="gpt-4o-mini"):
    """
    يستخدم GPT-4o للتحقق من كل جملة في الوصف.
    """
    from openai import OpenAI
    import base64

    client = OpenAI()
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""هل هذا الوصف دقيق بالكامل للصورة؟
إذا كان هناك أي شيء غير موجود في الصورة، اذكر ما هو.
إذا كان دقيقًا، أعد "دقيق".

الوصف: {caption}"""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────────────
# [031] code/notebooks/part06_multimodal-llms.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
def character_error_rate(pred: str, gold: str) -> float:
    """
    CER: نسبة الأخطاء على مستوى الحرف.
    """
    import Levenshtein
    if not gold:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gold) / len(gold)

def evaluate_arabic_ocr(vlm, samples, processor, tokenizer):
    """
    samples: list of (image_path, expected_text)
    """
    cers = []
    for image_path, expected in samples:
        image = Image.open(image_path).convert("RGB")
        pred = generate_answer(vlm, image, "اقرأ كل النص العربي في الصورة.", processor, tokenizer)
        # تنظيف
        pred = pred.strip().replace("النص:", "").strip()
        cer = character_error_rate(pred, expected.strip())
        cers.append(cer)

    return {
        "mean_cer": sum(cers) / len(cers),
        "min_cer": min(cers),
        "max_cer": max(cers),
    }

# ────────────────────────────────────────────────────────────────────────
# [032] code/notebooks/part06_multimodal-llms.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
def adversarial_test(vlm, image, processor, tokenizer):
    """
    اختبارات عدائية على صورة واحدة.
    """
    tests = [
        # أسئلة سلبية
        ("هل يوجد فيل في هذه الصورة؟", "لا"),
        ("هل هذه صورة ليلية؟", None),  # لا نعرف
        # أسئلة عن تفاصيل
        ("ما النص المكتوب في الزاوية العلوية اليمنى؟", None),
        # سؤال محمل
        ("لماذا الشخص في الصورة حزين؟", "قد لا يكون حزينًا"),
    ]

    results = []
    for question, expected in tests:
        answer = generate_answer(vlm, image, question, processor, tokenizer)
        results.append({"question": question, "answer": answer, "expected": expected})

    return results

# ────────────────────────────────────────────────────────────────────────
# [033] code/notebooks/part06_multimodal-llms.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
def human_eval_interface(samples):
    """
    واجهة بسيطة (CLI) لتقييم بشري.
    samples: list of (image_path, question, answer)
    """
    scores = []
    for image_path, question, answer in samples:
        print(f"\n{'='*60}")
        print(f"الصورة: {image_path}")
        print(f"السؤال: {question}")
        print(f"الإجابة: {answer}")
        print(f"{'='*60}")

        while True:
            score = input("قيّم (1-5): ").strip()
            if score in ["1", "2", "3", "4", "5"]:
                scores.append(int(score))
                break
            print("أدخل رقمًا من 1 إلى 5.")

    return {"mean_score": sum(scores) / len(scores), "scores": scores}

# ────────────────────────────────────────────────────────────────────────
# [034] code/ch25_multimodal-intro.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
الفصل 25: مدخل إلى النماذج متعددة الوسائط (Multimodal) — عندما يفتح النموذج عينيه

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السادس · الفصل 25 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch25_multimodal-intro.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch25_multimodal-intro.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] def load_vision_encoder(model_name="google/siglip-so400m-patch14-384") ════════


import torch
from transformers import AutoModel, AutoImageProcessor

def load_vision_encoder(model_name="google/siglip-so400m-patch14-384"):
    """
    يحمل Vision Encoder من HuggingFace.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, processor

def encode_image(image, model, processor, device="cuda"):
    """
    image: PIL.Image
    يُرجع: (num_patches, vision_dim)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state: (batch, num_patches, vision_dim)
    return outputs.last_hidden_state


# ════════ [2/8] class MLPProjector(nn.Module): ════════


import torch.nn as nn

class MLPProjector(nn.Module):
    """
    Projector بسيط: طبقتان خطيتان مع GELU.
    """
    def __init__(self, vision_dim: int, llm_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        """
        x = self.fc1(vision_features)
        x = self.act(x)
        x = self.fc2(x)
        return x  # (batch, num_patches, llm_dim)


# ════════ [3/8] class PerceiverResampler(nn.Module): ════════


class PerceiverResampler(nn.Module):
    """
    يضغط الرموز البصرية إلى عدد ثابت من الرموز.
    """
    def __init__(self, vision_dim: int, llm_dim: int, num_latents: int = 64, num_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, llm_dim) * 0.02)
        self.proj_in = nn.Linear(vision_dim, llm_dim) if vision_dim != llm_dim else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim),
        )

    def forward(self, vision_features):
        """
        vision_features: (batch, num_patches, vision_dim)
        يُرجع: (batch, num_latents, llm_dim)
        """
        x = self.proj_in(vision_features)  # (B, P, D)
        B = x.size(0)

        latents = self.latents.expand(B, -1, -1)  # (B, L, D)

        # Cross-attention: latents (query) تنتبه إلى vision_features (key/value)
        h = self.norm1(latents)
        h, _ = self.cross_attn(h, x, x)
        latents = latents + h

        # FFN
        latents = latents + self.ffn(self.norm2(latents))
        return latents


# ════════ [4/8] class SimpleVLM(nn.Module): ════════


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleVLM(nn.Module):
    """
    VLM بسيط: Vision Encoder + Projector + LLM.
    """
    def __init__(
        self,
        vision_encoder,
        llm,
        projector: nn.Module,
        image_token_id: int,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector
        self.image_token_id = image_token_id

        # جمّد Vision Encoder و LLM
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        """
        input_ids: (batch, seq) — يحتوي على image_token_id في موضع الصورة
        pixel_values: (batch, 3, H, W)
        """
        # 1. ترميز الصورة
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        # vision_features: (batch, num_patches, vision_dim)

        # 2. إسقاط إلى فضاء LLM
        visual_tokens = self.projector(vision_features)
        # visual_tokens: (batch, num_patches, llm_dim)

        # 3. تضمين النص
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # text_embeds: (batch, seq, llm_dim)

        # 4. استبدال image_token_id بتضمينات الصورة
        # نفترض أن كل صورة لها موضع واحد في input_ids (image_token_id)
        batch_size = input_ids.size(0)
        new_embeds = []
        for i in range(batch_size):
            # ابحث عن موضع image_token
            img_pos = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                new_embeds.append(text_embeds[i])
                continue

            pos = img_pos[0].item()
            # قبل الصورة، الصورة، بعد الصورة
            before = text_embeds[i, :pos]            # (pos, D)
            after = text_embeds[i, pos+1:]           # (seq-pos-1, D)
            combined = torch.cat([before, visual_tokens[i], after], dim=0)
            new_embeds.append(combined)

        # ملاحظة: الأطوال مختلفة الآن! نحتاج إلى حشو أو التعامل مع ذلك.
        # في التطبيق الحقيقي، نستخدم attention_mask ويتم التعامل مع الأطوال.
        # للتبسيط، نفترض أن كل العينات لها نفس العدد من image_tokens.
        inputs_embeds = torch.stack(new_embeds, dim=0)

        # 5. مرر عبر LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs


# ════════ [5/8] model_name = "Qwen/Qwen2-VL-2B-Instruct" ════════


from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

# تحميل نموذج VLM جاهز
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# تحميل صورة
image = Image.open("path/to/image.jpg")

# بناء المحادثة
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "صف هذه الصورة بالتفصيل."},
        ],
    }
]

# معالجة
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

# توليد
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# فك الترميز
response = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(response)


# ════════ [6/8] class ImageTextDataset(Dataset): ════════


from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

class ImageTextDataset(Dataset):
    """
    بيانات: (صورة، تعليق).
    """
    def __init__(self, pairs, processor, image_token="<image>"):
        self.pairs = pairs
        self.processor = processor
        self.image_token = image_token

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, caption = self.pairs[idx]
        # بناء النص: <image> + التعليق
        text = f"{self.image_token}\n{caption}"
        inputs = self.processor(
            text=text, images=image,
            return_tensors="pt", padding="max_length", truncation=True, max_length=512,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # labels = input_ids، مع -100 على رموز الصورة
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train_projector(vlm, dataset, epochs=1, lr=1e-3, batch_size=4):
    """
    يدرب Projector فقط (LLM و Vision Encoder مجمدان).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # فقط معاملات Projector
    optimizer = torch.optim.AdamW(vlm.projector.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, total_steps)

    vlm.projector.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(vlm.llm.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = vlm(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return vlm


# ════════ [7/8] def evaluate_vqa(model, processor, dataset, max_samples=100): ════════


def evaluate_vqa(model, processor, dataset, max_samples=100):
    """
    تقييم بسيط على VQA.
    """
    correct = 0
    total = 0
    for i, (image, question, answer) in enumerate(dataset):
        if i >= max_samples:
            break
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        pred = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if pred.lower() == answer.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


# ════════ [8/8] def test_arabic_ocr(model, processor, image): ════════


def test_arabic_ocr(model, processor, image):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "اقرأ كل النص العربي في هذه الصورة، وأعد النص كما هو."},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
