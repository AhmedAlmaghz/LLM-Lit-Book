# -*- coding: utf-8 -*-
"""
الفصل 4: الترميز إلى وحدات (Tokenization) — كيف يقرأ النموذج ما لا تراه عينك

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثاني · الفصل 4 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch04_tokenization.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch04_tokenization.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] def get_vocab(corpus): ════════


from collections import Counter

def get_vocab(corpus):
    """يحول corpus إلى قاموس من الكلمات المقسومة إلى حروف."""
    vocab = Counter()
    for word in corpus:
        # نضيف </w> لنهاية كل كلمة
        chars = tuple(word) + ("</w>",)
        vocab[chars] += 1
    return vocab

def get_stats(vocab):
    """يحسب تكرارات كل زوج متجاور."""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """يدمج pair في كل الكلمات."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab

# مثال
corpus = ["low", "lower", "lowest", "new", "newer", "newest"]
vocab = get_vocab(corpus)
num_merges = 10
merges = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(best)
    print(f"Merge {i+1}: {best} (freq={pairs[best]})")

print("\nالمفردات النهائية:")
for word, freq in vocab.items():
    print(f"  {' '.join(word)}: {freq}")


# ════════ [2/6] tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-case ════════


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
text = "unaffable programming"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['una', '##ff', '##able', 'programming']


# ════════ [3/6] spm.SentencePieceTrainer.train( ════════


import sentencepiece as spm

# تدريب tokenizer على ملف نصي
spm.SentencePieceTrainer.train(
    input="arabic_corpus.txt",
    model_prefix="arabic_sp",
    vocab_size=32000,
    model_type="bpe",  # أو "unigram"
    character_coverage=1.0,  # 1.0 لكل اللغات، 0.9995 للغات الكبيرة
    byte_fallback=True,  # للتعامل مع الرموز غير المعروفة
)

# الاستخدام
sp = spm.SentencePieceProcessor(model_file="arabic_sp.model")
text = "أنا أحب البرمجة"
pieces = sp.encode(text, out_type=str)
print(pieces)  # ['▁أنا', '▁أحب', '▁البرمجة']
ids = sp.encode(text, out_type=int)
print(ids)


# ════════ [4/6] def tokens_per_word(text, tokenizer): ════════


def tokens_per_word(text, tokenizer):
    tokens = tokenizer.encode(text)
    words = len(text.split())
    return len(tokens) / words

# مقارنة
from transformers import AutoTokenizer

texts = {
    "english": "I love programming and building AI systems.",
    "arabic": "أنا أحب البرمجة وبناء أنظمة الذكاء الاصطناعي.",
}

for name, tok_name in [("GPT-4", "cl100k_base"), ("LLaMA", "meta-llama/Llama-2-7b-hf")]:
    print(f"\n=== {name} ===")
    if tok_name == "cl100k_base":
        import tiktoken
        enc = tiktoken.get_encoding(tok_name)
        for lang, text in texts.items():
            tokens = enc.encode(text)
            print(f"  {lang}: {len(tokens)} tokens, {len(text.split())} words, ratio={len(tokens)/len(text.split()):.2f}")
    else:
        tok = AutoTokenizer.from_pretrained(tok_name)
        for lang, text in texts.items():
            tokens = tok.encode(text)
            print(f"  {lang}: {len(tokens)} tokens, {len(text.split())} words, ratio={len(tokens)/len(text.split()):.2f}")


# ════════ [5/6] def normalize_arabic(text, remove_diacritics=True, unify_alef=True, un ════════


import re
import unicodedata

def normalize_arabic(text, remove_diacritics=True, unify_alef=True, unify_teh_marbuta=True):
    # إزالة التطويل
    text = re.sub(r"\u0640", "", text)

    if remove_diacritics:
        # إزالة التشكيل (الفتحة، الضمة، الكسرة، السكون، الشدة، التنوين)
        text = re.sub(r"[\u064B-\u0652]", "", text)

    if unify_alef:
        text = re.sub(r"[إأآا]", "ا", text)

    if unify_teh_marbuta:
        text = re.sub(r"ة", "ه", text)

    # توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ════════ [6/6] batch = tokenizer(texts, padding=True, truncation=True, return_tensors ════════


batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# batch["attention_mask"] موجودة تلقائيًا
