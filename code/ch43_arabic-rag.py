# -*- coding: utf-8 -*-
"""
الفصل 43: RAG للمستندات العربية — عندما تقرأ الآلة ما كتبه البشر بيدهم

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 43 · عدد كتل بايثون في هذا الملف: 11

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch43_arabic-rag.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch43_arabic-rag.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/11] def extract_text_pymupdf(pdf_path: str) -> list[dict]: ════════


import fitz  # PyMuPDF
import pdfplumber

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """
    يستخرج النص من PDF باستخدام PyMuPDF.
    يُرجع قائمة من الصفحات، كل صفحة نص.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages

def extract_text_pdfplumber(pdf_path: str) -> list[dict]:
    """
    يستخرج النص من PDF باستخدام pdfplumber.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": page_num + 1, "text": text})
    return pages


# ════════ [2/11] def is_text_reversed(text: str) -> bool: ════════


import re

def is_text_reversed(text: str) -> bool:
    """
    يكشف إذا كان النص العربي معكوسًا.
    """
    # إذا كان النص يبدأ بحرف عربي وينتهي بحرف عربي، لكن الكلمات معكوسة
    # طريقة بسيطة: ابحث عن كلمات شائعة معكوسة
    common_reversed = ["ةطقلا", "نبا", "نم", "ىلع"]  # القطة، ابن، من، على
    for word in common_reversed:
        if word in text:
            return True
    return False

def reverse_arabic_text(text: str) -> str:
    """يعكس النص العربي (إذا كان معكوسًا)."""
    return text[::-1]


# ════════ [3/11] def extract_text_ocr(image_path: str, lang: str = "ara") -> str: ════════


import pytesseract
from PIL import Image

def extract_text_ocr(image_path: str, lang: str = "ara") -> str:
    """
    يستخرج النص من صورة باستخدام Tesseract.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def extract_text_vlm(image_path: str, vlm, processor, tokenizer) -> str:
    """
    يستخرج النص باستخدام VLM (أدق للعربية).
    """
    image = Image.open(image_path).convert("RGB")
    prompt = "اقرأ كل النص العربي في هذه الصورة. أعد النص كما هو، دون شرح."
    return generate_answer(vlm, image, prompt, processor, tokenizer)


# ════════ [4/11] def extract_tables(pdf_path: str, pages: str = "all") -> list: ════════


import camelot

def extract_tables(pdf_path: str, pages: str = "all") -> list:
    """
    يستخرج الجداول من PDF باستخدام Camelot.
    """
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
    return [table.df.to_dict() for table in tables]


# ════════ [5/11] def split_arabic_sentences(text: str) -> list[str]: ════════


import re

def split_arabic_sentences(text: str) -> list[str]:
    """
    يقسم النص إلى جمل، مع احترام علامات الترقيم العربية.
    """
    # علامات نهاية الجملة في العربية
    sentence_endings = r"[.!?؟۔؛]"
    # قسّم عند علامة نهاية، متبوعة بمسافة أو نهاية النص
    parts = re.split(f"(?<=[{sentence_endings}])\\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    يقسم النص إلى قطع، مع احترام حدود الجمل.
    """
    sentences = split_arabic_sentences(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chunk_size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            # ابدأ جديدة، مع تداخل (آخر overlap حرفًا من السابق)
            overlap_text = current[-overlap:] if len(current) > overlap else ""
            current = overlap_text + sent + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ════════ [6/11] def hierarchical_chunk(text: str, max_chunk_size: int = 500) -> list[d ════════


def hierarchical_chunk(text: str, max_chunk_size: int = 500) -> list[dict]:
    """
    تقسيم هرمي: أقسام -> فقرات -> جمل.
    """
    # افترض أن الأقسام مفصولة بعناوين (مثل "الفصل")
    sections = re.split(r"\n(?=الفصل|المادة|القسم)", text)

    chunks = []
    for section in sections:
        paragraphs = section.split("\n\n")
        for para in paragraphs:
            if len(para) <= max_chunk_size:
                chunks.append({"text": para.strip(), "type": "paragraph"})
            else:
                for sub in chunk_by_sentences(para, max_chunk_size):
                    chunks.append({"text": sub, "type": "sentence"})
    return chunks


# ════════ [7/11] def embed_chunks(chunks: list[dict], model_name: str = "intfloat/multi ════════


from sentence_transformers import SentenceTransformer

def embed_chunks(chunks: list[dict], model_name: str = "intfloat/multilingual-e5-large"):
    """
    يحول القطع إلى متجهات.
    """
    model = SentenceTransformer(model_name)
    texts = [f"passage: {c['text']}" for c in chunks]  # e5 يحتاج بادئة
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return embeddings, model


# ════════ [8/11] def hybrid_retrieval( ════════


from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_retrieval(
    query: str,
    query_embedding: np.ndarray,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    k: int = 5,
):
    """
    بحث هجين: dense + BM25، مع RRF.
    """
    # 1. Dense: تشابه جيبي
    dense_scores = chunk_embeddings @ query_embedding
    dense_top = np.argsort(dense_scores)[::-1][:k*2]

    # 2. BM25
    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())
    bm25_top = np.argsort(bm25_scores)[::-1][:k*2]

    # 3. RRF
    rrf = {}
    for rank, idx in enumerate(dense_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank + 1)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices]


# ════════ [9/11] def rerank(query: str, chunks: list[dict], top_k: int = 3): ════════


from sentence_transformers import CrossEncoder

def rerank(query: str, chunks: list[dict], top_k: int = 3):
    """
    يعيد ترتيب القطع بـ Cross-Encoder.
    """
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [{"chunk": c, "score": s} for c, s in ranked[:top_k]]


# ════════ [10/11] def build_rag_prompt(query: str, contexts: list[dict]) -> str: ════════


def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    """
    يبني أمر RAG مع سياق ومصادر.
    """
    context_text = "\n\n".join([
        f"[مصدر {i+1}] (صفحة {c.get('page', '?')}):\n{c['text']}"
        for i, c in enumerate(contexts)
    ])

    return f"""أنت مساعد خبير يجيب على الأسئلة بناءً على المستندات المُقدمة فقط.

السياق:
{context_text}

السؤال: {query}

التعليمات:
- أجب بناءً على السياق فقط.
- إذا لم تجد الإجابة في السياق، قل: "لا أجد هذه المعلومة في المستندات المتاحة."
- اذكر رقم المصدر ورقم الصفحة (مثل: [مصدر 1، صفحة 5]).
- كن دقيقًا وموجزًا.
- إذا كان السؤال يحتاج إلى أرقام، اذكرها كما هي.

الإجابة:"""


# ════════ [11/11] def hierarchical_retrieval(query, sections_index, chunks_index, embedd ════════


def hierarchical_retrieval(query, sections_index, chunks_index, embeddings, k=5):
    """
    1. ابحث في ملخصات الأقسام.
    2. ثم ابحث في قطع القسم المختار.
    """
    # 1. بحث في الأقسام
    section_embeddings = sections_index["embeddings"]
    query_emb = embeddings.encode(query)
    section_scores = section_embeddings @ query_emb
    best_section = sections_index["names"][section_scores.argmax()]

    # 2. بحث في قطع القسم
    section_chunks = [c for c in chunks_index if c["section"] == best_section]
    # ... بحث dense + BM25 ...
    return section_chunks[:k]
