# -*- coding: utf-8 -*-
"""
الفصل 44: تلخيص المستندات الطويلة — عندما ينسى النموذج ما في منتصف الكتاب

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 44 · عدد كتل بايثون في هذا الملف: 8

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch44_long-document-summarization.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch44_long-document-summarization.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/8] def extractive_summarize(text: str, num_sentences: int = 5, embedder=N ════════


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def extractive_summarize(text: str, num_sentences: int = 5, embedder=None):
    """
    تلخيص استخراجي بـ TextRank (مبسط).
    """
    if embedder is None:
        embedder = SentenceTransformer("intfloat/multilingual-e5-small")

    sentences = split_arabic_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # تضمين الجمل
    embeddings = embedder.encode([f"passage: {s}" for s in sentences], normalize_embeddings=True)

    # مصفوفة التشابه
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)

    # PageRank (مبسط)
    scores = np.ones(len(sentences)) / len(sentences)
    damping = 0.85
    for _ in range(50):
        new_scores = (1 - damping) / len(sentences) + damping * sim_matrix.T @ scores
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    # اختر أعلى الجمل
    top_indices = np.argsort(scores)[::-1][:num_sentences]
    top_indices = sorted(top_indices)  # احتفظ بالترتيب الأصلي

    return " ".join([sentences[i] for i in top_indices])


# ════════ [2/8] def mmr_selection(embeddings, scores, num_sentences: int, lambda_param ════════


def mmr_selection(embeddings, scores, num_sentences: int, lambda_param: float = 0.5):
    """
    Maximal Marginal Relevance: اختر جملًا ذات صلة، لكن متنوعة.
    """
    selected = []
    candidates = list(range(len(scores)))

    for _ in range(num_sentences):
        if not candidates:
            break
        mmr_scores = []
        for i in candidates:
            relevance = scores[i]
            if selected:
                similarity = max(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] for j in selected)
            else:
                similarity = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * similarity
            mmr_scores.append((i, mmr))
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)

    return selected


# ════════ [3/8] def summarize_simple(text: str, llm, max_words: int = 200) -> str: ════════


def summarize_simple(text: str, llm, max_words: int = 200) -> str:
    """تلخيص بسيط بـ LLM."""
    prompt = f"""لخص النص التالي في {max_words} كلمة.
حافظ على الأفكار الرئيسية، والأرقام الأساسية.
لا تضف معلومات غير موجودة في النص.
اكتب بالعربية الفصحى.

النص:
{text}

الملخص:"""
    return llm(prompt)


# ════════ [4/8] def map_reduce_summarize( ════════


def map_reduce_summarize(
    text: str,
    llm,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    summary_words_per_chunk: int = 150,
    final_summary_words: int = 300,
) -> str:
    """
    Map-Reduce: قسّم، لخّص كل جزء، ثم لخّص الملخصات.
    """
    # 1. تقسيم
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])

    print(f"Split into {len(chunks)} chunks")

    # 2. Map: لخّص كل جزء
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""لخص الجزء التالي من مستند طويل في {summary_words_per_chunk} كلمة.
هذا الجزء {i+1} من {len(chunks)}. حافظ على المعلومات المهمة، والأرقام.
لا تضف معلومات غير موجودة.

الجزء:
{chunk}

الملخص:"""
        summary = llm(prompt)
        partial_summaries.append(summary)
        print(f"  Chunk {i+1}/{len(chunks)} summarized")

    # 3. Reduce: لخّص الملخصات
    combined = "\n\n".join([f"[جزء {i+1}]: {s}" for i, s in enumerate(partial_summaries)])

    final_prompt = f"""لديك ملخصات لأجزاء من مستند طويل. ادمجها في ملخص نهائي واحد ({(final_summary_words)} كلمة).
حافظ على التسلسل المنطقي. أزل التكرار. لا تفقد المعلومات الأساسية.
لا تضف معلومات غير موجودة.

الملخصات:
{combined}

الملخص النهائي:"""
    return llm(final_prompt)


# ════════ [5/8] def refine_summarize( ════════


def refine_summarize(
    text: str,
    llm,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    target_words: int = 300,
) -> str:
    """
    Refine: لخّص تدريجيًا، مع تحسين الملخص في كل خطوة.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])

    # 1. الملخص الأولي
    current_summary = llm(f"""لخص النص التالي في {target_words} كلمة.
حافظ على المعلومات المهمة.

النص:
{chunks[0]}

الملخص:""")

    # 2. حسّن بالتدريج
    for i, chunk in enumerate(chunks[1:], start=1):
        prompt = f"""لديك ملخص حالي، وجزء جديد من نفس المستند.
حسّن الملخص بإضافة المعلومات المهمة من الجزء الجديد.
احتفظ بالطول التقريبي ({target_words} كلمة).
أزل التكرار. لا تفقد المعلومات السابقة.

الملخص الحالي:
{current_summary}

الجزء الجديد:
{chunk}

الملخص المحسن:"""
        current_summary = llm(prompt)
        print(f"  Refined with chunk {i+1}/{len(chunks)}")

    return current_summary


# ════════ [6/8] def rag_summarize(query: str, chunks: list[dict], embedder, llm, top_k ════════


def rag_summarize(query: str, chunks: list[dict], embedder, llm, top_k: int = 10):
    """
    تلخيص موجّه بسؤال: استرجع، ثم لخّص.
    """
    # 1. استرجاع
    query_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True)
    chunk_embs = embedder.encode([f"passage: {c['text']}" for c in chunks], normalize_embeddings=True)
    scores = chunk_embs @ query_emb[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]

    # 2. تلخيص
    context = "\n\n".join([c["text"] for c in top_chunks])
    prompt = f"""لخص المعلومات التالية ذات الصلة بالسؤال: "{query}"
حافظ على الأرقام والتفاصيل. لا تضف معلومات غير موجودة.

المعلومات:
{context}

الملخص:"""
    return llm(prompt)


# ════════ [7/8] def evaluate_summary(source: str, summary: str, llm_judge) -> dict: ════════


def evaluate_summary(source: str, summary: str, llm_judge) -> dict:
    """
    تقييم شامل: أمانة، تغطية، سلاسة.
    """
    prompt = f"""قيّم الملخص التالي مقارنةً بالنص الأصلي. أعد JSON.

النص الأصلي (مقتطف):
{source[:3000]}

الملخص:
{summary}

المعايير (1-5):
1. الأمانة: هل كل ما في الملخص موجود في النص؟ (5 = لا هلوسة)
2. التغطية: هل يغطي الأفكار الرئيسية؟ (5 = كل الأفكار)
3. السلاسة: هل يقرأ كأن إنسانًا كتبه؟ (5 = سلس)
4. الإيجاز: هل هو موجز دون فقدان المعنى؟ (5 = مثالي)

أعد JSON:
{{"faithfulness": <1-5>, "coverage": <1-5>, "fluency": <1-5>, "conciseness": <1-5>, "overall": <1-5>, "missing_points": ["..."]}}"""

    import json
    response = llm_judge(prompt)
    return json.loads(response)


# ════════ [8/8] def test_lost_in_middle(llm, needle: str, haystack_size: int = 50000): ════════


def test_lost_in_middle(llm, needle: str, haystack_size: int = 50000):
    """
    يختبر إذا كان النموذج يجد معلومة في منتصف السياق.
    """
    filler = "هذا نص حشو. " * (haystack_size // 15)

    positions = {"beginning": 0, "middle": len(filler) // 2, "end": len(filler) - len(needle) - 100}

    results = {}
    for name, pos in positions.items():
        context = filler[:pos] + needle + filler[pos:]
        prompt = f"""النص التالي يحتوي على معلومة. ابحث عنها وأعدها.

{context}

المعلومة:"""
        response = llm(prompt)
        results[name] = needle.lower() in response.lower()

    return results
