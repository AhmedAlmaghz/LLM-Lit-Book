# -*- coding: utf-8 -*-
"""
الفصل 13: بناء Tokenizer وتجهيز البيانات بعمق — المطبخ الخفي الذي يحدد جودة النموذج

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 13 · عدد كتل بايثون في هذا الملف: 12

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch13_tokenizer-data-pipeline.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch13_tokenizer-data-pipeline.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/12] def normalize_arabic( ════════


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


# ════════ [2/12] def train_bpe_tokenizer( ════════


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


# ════════ [3/12] def train_sentencepiece_unigram( ════════


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


# ════════ [4/12] def compute_tokenizer_metrics(texts, tokenizer, is_sp=False): ════════


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


# ════════ [5/12] def build_token_stream(texts, tokenizer, eos_id, is_sp=False): ════════


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


# ════════ [6/12] def best_fit_pack(documents, block_size, eos_id=None): ════════


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


# ════════ [7/12] def sliding_window_chunks(doc_ids, block_size, stride=None): ════════


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


# ════════ [8/12] class PackedLMDataset(Dataset): ════════


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


# ════════ [9/12] def create_dataloader(blocks, batch_size=8, num_workers=4, pin_memory= ════════


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


# ════════ [10/12] class StreamingLMDataset(IterableDataset): ════════


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


# ════════ [11/12] def grouped_split(texts, groups, test_size=0.15, val_size=0.15, seed=4 ════════


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


# ════════ [12/12] def text_fingerprint(text): ════════


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
