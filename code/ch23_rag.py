# -*- coding: utf-8 -*-
"""
الفصل 23: التوليد المعزّز بالاسترجاع (RAG) — كيف تجعل النموذج يجيب من كتب لم يقرأها

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الخامس · الفصل 23 · عدد كتل بايثون في هذا الملف: 10

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch23_rag.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch23_rag.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/10] def recursive_chunk(text: str, chunk_size: int = 400, overlap: int = 5 ════════


def recursive_chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    تقسيم النص بشكل تكراري: فقرات -> جمل -> كلمات.
    """
    # 1. جرب تقسيم على فقرات
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        # إذا كانت الفقرة أطول من chunk_size، قسمها على جمل
        if len(para) > chunk_size:
            sentences = para.replace("\n", " ").split(". ")
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent + ". "
                else:
                    if current:
                        chunks.append(current.strip())
                    # ابدأ جديدة مع تداخل
                    current = current[-overlap:] if len(current) > overlap else ""
                    current += sent + ". "
        else:
            if len(current) + len(para) <= chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ════════ [2/10] def semantic_chunk(text: str, embedder, threshold: float = 0.7) -> lis ════════


from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, embedder, threshold: float = 0.7) -> list[str]:
    """
    تقسيم بناءً على تغير التشابه بين الجمل.
    """
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) < 2:
        return [text]

    embeddings = embedder.encode(sentences, normalize_embeddings=True)
    # التشابه بين كل جملتين متتاليتين
    sims = [np.dot(embeddings[i], embeddings[i+1]) for i in range(len(sentences)-1)]

    chunks = []
    current = [sentences[0]]
    for i, sim in enumerate(sims):
        if sim < threshold:
            chunks.append(". ".join(current) + ".")
            current = [sentences[i+1]]
        else:
            current.append(sentences[i+1])
    if current:
        chunks.append(". ".join(current) + ".")

    return chunks


# ════════ [3/10] def embed_chunks(chunks: list[str], model_name: str = "intfloat/multil ════════


from sentence_transformers import SentenceTransformer
import torch

def embed_chunks(chunks: list[str], model_name: str = "intfloat/multilingual-e5-large"):
    """
    يحول القطع النصية إلى متجهات.
    """
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    # e5 يتطلب بادئة "passage: " للقطع
    # و"query: " للأسئلة (مهم جدًا!)
    prefixed = [f"passage: {c}" for c in chunks]
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,   # مهم للتشابه الجيبي
        batch_size=32,
        show_progress_bar=True,
    )
    return embeddings, model


# ════════ [4/10] def build_faiss_index(embeddings: np.ndarray): ════════


import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray):
    """
    بناء فهرس FAISS للبحث بالتشابه الجيبي.
    embeddings: (n, d)، مطبعة (normalized).
    """
    d = embeddings.shape[1]
    # IndexFlatIP: الضرب النقطي = التشابه الجيبي للمتجهات المطبعة
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index

def search_faiss(index, query_embedding: np.ndarray, k: int = 5):
    """
    يبحث عن أعلى k تشابهًا.
    query_embedding: (1, d)، مطبع.
    """
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    return scores[0], indices[0]


# ════════ [5/10] client = QdrantClient(":memory:")  # أو host="localhost", port=6333 ════════


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")  # أو host="localhost", port=6333

# إنشاء collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# إضافة القطع
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": chunks[i], "source": "doc1.pdf"},
    )
    for i in range(len(chunks))
]
client.upsert(collection_name="docs", points=points)

# البحث
results = client.search(
    collection_name="docs",
    query_vector=query_embedding[0].tolist(),
    limit=5,
)
for r in results:
    print(r.score, r.payload["text"][:100])


# ════════ [6/10] def build_bm25(chunks: list[str]): ════════


from rank_bm25 import BM25Okapi

def build_bm25(chunks: list[str]):
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)

def search_bm25(bm25, query: str, k: int = 5):
    scores = bm25.get_scores(query.split())
    top_k = np.argsort(scores)[::-1][:k]
    return scores[top_k], top_k


# ════════ [7/10] def hybrid_search( ════════


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    faiss_index,
    bm25,
    k: int = 5,
    alpha: float = 0.5,  # وزن البحث الكثيف
):
    """
    دمج نتائج FAISS وBM25 باستخدام Reciprocal Rank Fusion (RRF).
    """
    # احصل على نتائج FAISS
    _, faiss_indices = search_faiss(faiss_index, query_embedding, k=k*2)
    # احصل على نتائج BM25
    _, bm25_indices = search_bm25(bm25, query, k=k*2)

    # RRF: دمج الترتيب
    rrf_scores = {}
    for rank, idx in enumerate(faiss_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (60 + rank + 1)
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (60 + rank + 1)

    # رتب حسب النتيجة
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return sorted_indices


# ════════ [8/10] def rerank(query: str, chunks: list[str], top_k: int = 3, model_name:  ════════


from sentence_transformers import CrossEncoder

def rerank(query: str, chunks: list[str], top_k: int = 3, model_name: str = "BAAI/bge-reranker-v2-m3"):
    """
    يعيد ترتيب القطع باستخدام Cross-Encoder.
    """
    model = CrossEncoder(model_name, device="cuda")
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs)
    # رتب تنازليًا
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ════════ [9/10] def build_rag_prompt(query: str, contexts: list[str]) -> str: ════════


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    context_text = "\n\n".join([f"[مصدر {i+1}]: {c}" for i, c in enumerate(contexts)])

    return f"""أنت مساعد يجيب على الأسئلة بناءً على السياق المُقدم فقط.

السياق:
{context_text}

السؤال: {query}

التعليمات:
- أجب بناءً على السياق فقط.
- إذا لم تجد الإجابة في السياق، قل: "لا أجد هذه المعلومة في المصادر المتاحة."
- اذكر رقم المصدر الذي استخدمته (مثل [مصدر 1]).
- كن موجزًا ودقيقًا.

الإجابة:"""


# ════════ [10/10] data = { ════════


# مثال باستخدام RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

data = {
    "question": ["ما هي عاصمة فرنسا؟"],
    "answer": ["باريس."],
    "contexts": [["باريس هي عاصمة فرنسا وأكبر مدنها."]],
    "ground_truth": ["باريس"],
}
dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
