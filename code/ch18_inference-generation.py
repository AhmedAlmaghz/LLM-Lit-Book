# -*- coding: utf-8 -*-
"""
الفصل 18: الاستدلال والتوليد — كيف يختار النموذج كلمته التالية؟

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الرابع · الفصل 18 · عدد كتل بايثون في هذا الملف: 11

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch18_inference-generation.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch18_inference-generation.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/11] def greedy_decode(model, input_ids, max_new_tokens=50, eos_token_id=No ════════


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


# ════════ [2/11] def beam_search(model, input_ids, beam_size=4, max_new_tokens=50, eos_ ════════


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


# ════════ [3/11] def apply_temperature(logits, temperature=1.0): ════════


def apply_temperature(logits, temperature=1.0):
    if temperature == 0:
        return logits  # Greedy
    return logits / temperature


# ════════ [4/11] def top_k_filtering(logits, top_k=50): ════════


def top_k_filtering(logits, top_k=50):
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    # احصل على قيم top-k
    values, _ = torch.topk(logits, top_k, dim=-1)
    # احتفظ فقط بالقيم >= أصغر قيمة في top-k
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


# ════════ [5/11] def top_p_filtering(logits, top_p=0.9): ════════


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


# ════════ [6/11] def min_p_filtering(logits, min_p=0.05): ════════


def min_p_filtering(logits, min_p=0.05):
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))


# ════════ [7/11] def apply_repetition_penalty(logits, input_ids, penalty=1.2): ════════


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


# ════════ [8/11] @torch.no_grad() ════════


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


# ════════ [9/11] prompt = "كان يا ما كان" ════════


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


# ════════ [10/11] @torch.no_grad() ════════


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


# ════════ [11/11] def prompt_lookup_proposal(input_ids, n=3, num_pred_tokens=3): ════════


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
