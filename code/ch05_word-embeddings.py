# -*- coding: utf-8 -*-
"""
الفصل 5: تضمينات الكلمات (Word2Vec وGloVe) — كيف تتحول الكلمة إلى إحداثيات في فضاء المعنى

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثاني · الفصل 5 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch05_word-embeddings.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch05_word-embeddings.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] class CBOW(nn.Module): ════════


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


# ════════ [2/8] class SkipGram(nn.Module): ════════


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


# ════════ [3/8] class SkipGramNegativeSampling(nn.Module): ════════


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


# ════════ [4/8] def build_vocab(sentences, min_count=1): ════════


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


# ════════ [5/8] glove = api.load("glove-wiki-gigaword-100") ════════


# باستخدام gensim
import gensim.downloader as api

# تحميل GloVe مدرب مسبقًا (100 بُعد)
glove = api.load("glove-wiki-gigaword-100")
print(glove.most_similar("king", topn=5))
# [('prince', 0.77), ('monarch', 0.73), ('throne', 0.72), ...]

# العملية الشهيرة
result = glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(result)  # [('queen', 0.71)]


# ════════ [6/8] def evaluate_similarity(embeddings, vocab, test_pairs): ════════


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


# ════════ [7/8] def evaluate_analogy(embeddings, vocab, id2word, analogies): ════════


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


# ════════ [8/8] sentences = [line.split() for line in open("arabic_corpus.txt", encodi ════════


from gensim.models import FastText

# تدريب
sentences = [line.split() for line in open("arabic_corpus.txt", encoding="utf-8")]
model = FastText(sentences, vector_size=100, window=5, min_count=5, epochs=10, sg=1)

# الكلمات المتشابهة
model.wv.most_similar("البرمجة", topn=5)

# كلمة غير موجودة في المفردات: FastText يمثلها من n-grams
model.wv["برمجه"]  # حتى لو لم تكن في corpus، لها متجه
