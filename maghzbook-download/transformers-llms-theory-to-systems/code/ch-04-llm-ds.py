#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المحوّلات والنماذج اللغوية الكبيرة — الترميز إلى وحدات (Tokenization) — كيف يقرأ النموذج ما لا تراه عينك
Transformers and Large Language Models — Tokenization

Book slug : transformers-llms-theory-to-systems
Chapter   : ch-04-llm-ds (lesson 4)
Code files: 25 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/notebooks/part02_embeddings-and-representations.ipynb#cell-1
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
# [002] code/notebooks/part02_embeddings-and-representations.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/notebooks/part02_embeddings-and-representations.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
text = "unaffable programming"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['una', '##ff', '##able', 'programming']

# ────────────────────────────────────────────────────────────────────────
# [004] code/notebooks/part02_embeddings-and-representations.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] code/notebooks/part02_embeddings-and-representations.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/notebooks/part02_embeddings-and-representations.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] code/notebooks/part02_embeddings-and-representations.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# batch["attention_mask"] موجودة تلقائيًا

# ────────────────────────────────────────────────────────────────────────
# [008] code/notebooks/part02_embeddings-and-representations.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids):
        # context_ids: (batch, context_size)
        embeds = self.embeddings(context_ids)          # (batch, context_size, dim)
        context_vec = embeds.mean(dim=1)               # (batch, dim)
        logits = self.output(context_vec)              # (batch, vocab_size)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [009] code/notebooks/part02_embeddings-and-representations.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, center_ids, context_ids):
        # center_ids: (batch,)
        # context_ids: (batch, context_size)
        center_embeds = self.embeddings(center_ids)    # (batch, dim)
        logits = self.output(center_embeds)            # (batch, vocab_size)
        # نقارن مع كل كلمة في السياق
        loss = F.cross_entropy(
            logits.unsqueeze(1).expand(-1, context_ids.size(1), -1).reshape(-1, logits.size(-1)),
            context_ids.reshape(-1),
        )
        return loss

# ────────────────────────────────────────────────────────────────────────
# [010] code/notebooks/part02_embeddings-and-representations.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_ids, pos_context_ids, neg_context_ids):
        # center: (batch,)
        # pos: (batch,)
        # neg: (batch, k)
        center = self.center_embeddings(center_ids)         # (batch, dim)
        pos = self.context_embeddings(pos_context_ids)      # (batch, dim)
        neg = self.context_embeddings(neg_context_ids)      # (batch, k, dim)

        # النقاط: كلما زادت، زاد احتمال أن الزوج حقيقي
        pos_score = torch.sum(center * pos, dim=-1)         # (batch,)
        neg_score = torch.bmm(neg, center.unsqueeze(-1)).squeeze(-1)  # (batch, k)

        # خسارة: نريد pos_score عاليًا، neg_score منخفضًا
        loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())
        return loss

# ────────────────────────────────────────────────────────────────────────
# [011] code/notebooks/part02_embeddings-and-representations.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

# ===== 1. تجهيز البيانات =====
def build_vocab(sentences, min_count=1):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    vocab = {w: i for i, (w, c) in enumerate(counter.items()) if c >= min_count}
    return vocab, counter

def generate_pairs(sentences, vocab, window=2):
    pairs = []
    for sent in sentences:
        ids = [vocab.get(w, -1) for w in sent]
        ids = [i for i in ids if i >= 0]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return pairs

# ===== 2. Dataset =====
class SkipGramDataset(Dataset):
    def __init__(self, pairs, vocab_size, n_neg=5, power=0.75):
        self.pairs = pairs
        self.vocab_size = vocab_size
        self.n_neg = n_neg
        # توزيع العينات السالبة
        self.neg_probs = torch.tensor([power], dtype=torch.float32)  # سيُحدّث لاحقًا

    def set_neg_distribution(self, counter, vocab):
        freqs = torch.zeros(self.vocab_size)
        for w, i in vocab.items():
            freqs[i] = counter[w] ** 0.75
        self.neg_probs = freqs / freqs.sum()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, pos = self.pairs[idx]
        negatives = torch.multinomial(self.neg_probs, self.n_neg, replacement=True)
        return torch.tensor(center), torch.tensor(pos), negatives

# ===== 3. النموذج =====
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)
        # تهيئة صغيرة
        nn.init.uniform_(self.center.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, center, pos, neg):
        c = self.center(center)                     # (B, D)
        p = self.context(pos)                       # (B, D)
        n = self.context(neg)                       # (B, K, D)

        pos_score = (c * p).sum(-1)                 # (B,)
        neg_score = torch.bmm(n, c.unsqueeze(-1)).squeeze(-1)  # (B, K)

        loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())
        return loss

# ===== 4. التدريب =====
def train_word2vec(sentences, embed_dim=100, window=2, n_neg=5,
                   epochs=5, batch_size=256, lr=0.025):
    vocab, counter = build_vocab(sentences)
    pairs = generate_pairs(sentences, vocab, window)
    dataset = SkipGramDataset(pairs, len(vocab), n_neg)
    dataset.set_neg_distribution(counter, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Word2Vec(len(vocab), embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for center, pos, neg in loader:
            optimizer.zero_grad()
            loss = model(center, pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    # التضمين النهائي: متوسط المصفوفتين
    embeddings = (model.center.weight + model.context.weight).detach().numpy()
    id2word = {i: w for w, i in vocab.items()}
    return embeddings, vocab, id2word

# ===== 5. البحث عن أقرب الجيران =====
def nearest_neighbors(word, embeddings, vocab, id2word, top_k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    if word not in vocab:
        print(f"'{word}' not in vocab")
        return
    idx = vocab[word]
    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    top = sims.argsort()[::-1][1:top_k+1]
    return [(id2word[i], float(sims[i])) for i in top]

# ===== مثال =====
if __name__ == "__main__":
    sentences = [
        "الملك يحكم البلاد".split(),
        "الملكة تحكم البلاد".split(),
        "الرجل يعمل في الحقل".split(),
        "المرأة تعمل في الحقل".split(),
        # ... (أضف المزيد من الجمل المتنوعة)
    ]
    embeddings, vocab, id2word = train_word2vec(sentences, embed_dim=50, epochs=10)
    for w in ["الملك", "الرجل"]:
        print(w, "→", nearest_neighbors(w, embeddings, vocab, id2word, 3))

# ────────────────────────────────────────────────────────────────────────
# [012] code/notebooks/part02_embeddings-and-representations.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# باستخدام gensim
import gensim.downloader as api

# تحميل GloVe مدرب مسبقًا (100 بُعد)
glove = api.load("glove-wiki-gigaword-100")
print(glove.most_similar("king", topn=5))
# [('prince', 0.77), ('monarch', 0.73), ('throne', 0.72), ...]

# العملية الشهيرة
result = glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(result)  # [('queen', 0.71)]

# ────────────────────────────────────────────────────────────────────────
# [013] code/notebooks/part02_embeddings-and-representations.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

def evaluate_similarity(embeddings, vocab, test_pairs):
    """test_pairs: list of (word1, word2, human_score)"""
    human_scores, model_scores = [], []
    for w1, w2, score in test_pairs:
        if w1 in vocab and w2 in vocab:
            v1 = embeddings[vocab[w1]].reshape(1, -1)
            v2 = embeddings[vocab[w2]].reshape(1, -1)
            model_scores.append(cosine_similarity(v1, v2)[0][0])
            human_scores.append(score)
    return spearmanr(human_scores, model_scores).correlation

# ────────────────────────────────────────────────────────────────────────
# [014] code/notebooks/part02_embeddings-and-representations.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
def evaluate_analogy(embeddings, vocab, id2word, analogies):
    """analogies: list of (a, b, c, expected_d)"""
    correct, total = 0, 0
    for a, b, c, expected in analogies:
        if not all(w in vocab for w in [a, b, c]):
            continue
        target = embeddings[vocab[a]] - embeddings[vocab[b]] + embeddings[vocab[c]]
        sims = cosine_similarity([target], embeddings)[0]
        # استبعد الكلمات المدخلة
        for w in [a, b, c]:
            sims[vocab[w]] = -1
        pred_idx = sims.argmax()
        total += 1
        if id2word[pred_idx] == expected:
            correct += 1
    return correct / total if total > 0 else 0

# ────────────────────────────────────────────────────────────────────────
# [015] code/notebooks/part02_embeddings-and-representations.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
from gensim.models import FastText

# تدريب
sentences = [line.split() for line in open("arabic_corpus.txt", encoding="utf-8")]
model = FastText(sentences, vector_size=100, window=5, min_count=5, epochs=10, sg=1)

# الكلمات المتشابهة
model.wv.most_similar("البرمجة", topn=5)

# كلمة غير موجودة في المفردات: FastText يمثلها من n-grams
model.wv["برمجه"]  # حتى لو لم تكن في corpus، لها متجه

# ────────────────────────────────────────────────────────────────────────
# [016] code/notebooks/part02_embeddings-and-representations.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_hy = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        h = h0
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h))
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.W_hy(outputs)             # (batch, seq_len, output_dim)
        return logits, h

# ────────────────────────────────────────────────────────────────────────
# [017] code/notebooks/part02_embeddings-and-representations.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
def train_rnn(model, data, epochs=10, seq_len=35, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        h = None
        total_loss = 0
        for batch in data:
            optimizer.zero_grad()
            logits, h = model(batch["input"], h0=h.detach() if h is not None else None)
            loss = criterion(logits.view(-1, logits.size(-1)), batch["target"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(data):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [018] code/notebooks/part02_embeddings-and-representations.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # نجمع كل الأوزان في مصفوفة واحدة للكفاءة
        self.W = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=-1)
        gates = self.W(combined)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if state is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            state = [(h, c) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            new_state = []
            for l, cell in enumerate(self.cells):
                h, c = cell(inp, state[l])
                inp = self.dropout(h) if l < self.num_layers - 1 else h
                new_state.append((h, c))
            state = new_state
            outputs.append(inp)

        outputs = torch.stack(outputs, dim=1)
        return outputs, state

# ────────────────────────────────────────────────────────────────────────
# [019] code/notebooks/part02_embeddings-and-representations.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# lengths: (batch,) طول كل تسلسل قبل الحشو
packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)

# ────────────────────────────────────────────────────────────────────────
# [020] code/notebooks/part02_embeddings-and-representations.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

# ────────────────────────────────────────────────────────────────────────
# [021] code/notebooks/part02_embeddings-and-representations.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        return self.fc(out)

# ────────────────────────────────────────────────────────────────────────
# [022] code/notebooks/part02_embeddings-and-representations.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state):
        emb = self.embed(x)
        outputs, state = self.lstm(emb, state)
        logits = self.fc(outputs)
        return logits, state

# ────────────────────────────────────────────────────────────────────────
# [023] code/notebooks/part02_embeddings-and-representations.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
def mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: (batch, seq, hidden)
    attention_mask: (batch, seq) - 1 للكلمات الحقيقية، 0 للحشو
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# ────────────────────────────────────────────────────────────────────────
# [024] code/notebooks/part02_embeddings-and-representations.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
from seqeval.metrics import classification_report, f1_score

y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
y_pred = [["B-PER", "I-PER", "O", "B-LOC"]]
print(classification_report(y_true, y_pred))

# ────────────────────────────────────────────────────────────────────────
# [025] code/ch04_tokenization.py
# ────────────────────────────────────────────────────────────────────────
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
