# -*- coding: utf-8 -*-
"""
الفصل 35: مقاييس التقييم الأساسية — الأرقام التي تخبرك الحقيقة (وأحيانًا تكذب)

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثامن · الفصل 35 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch35_core-evaluation-metrics.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch35_core-evaluation-metrics.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] @torch.no_grad() ════════


import torch
import math
import torch.nn.functional as F

@torch.no_grad()
def compute_perplexity(model, dataloader, device="cuda"):
    """
    حساب Perplexity على مجموعة بيانات.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq, vocab)

        # نزيح: logits[:, :-1] تتنبأ بـ input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # خسارة، مع تجاهل الحشو
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # أو pad_token_id
            reduction="sum",
        )

        # عدد الرموز (بعد الإزاحة)
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": ppl, "tokens": total_tokens}


# ════════ [2/6] def compute_classification_metrics(y_true, y_pred, average="binary"): ════════


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, average="binary"):
    """
    يحسب المقاييس الأساسية.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics

# مثال
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

print(compute_classification_metrics(y_true, y_pred))


# ════════ [3/6] def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusio ════════


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


# ════════ [4/6] def compute_bleu(references: list[list[str]], candidates: list[str]): ════════


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

def compute_bleu(references: list[list[str]], candidates: list[str]):
    """
    references: list of list of tokens (مراجع متعددة لكل جملة)
    candidates: list of tokens (المولدة)
    """
    smoothing = SmoothingFunction().method1
    scores = []
    for ref, cand in zip(references, candidates):
        score = sentence_bleu([ref], cand, smoothing_function=smoothing)
        scores.append(score)
    return sum(scores) / len(scores)

# مثال
references = [
    [["القطة", "تجلس", "على", "السجادة"]],
    [["السماء", "زرقاء", "اليوم"]],
]
candidates = [
    ["القطة", "تجلس", "على", "السجادة"],
    ["السماء", "صافية", "اليوم"],
]
print(f"BLEU: {compute_bleu(references, candidates):.4f}")


# ════════ [5/6] def compute_rouge(reference: str, candidate: str): ════════


from rouge_score import rouge_scorer

def compute_rouge(reference: str, candidate: str):
    """
    يحسب ROUGE-1، ROUGE-2، ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# مثال
reference = "القطة تجلس على السجادة الحمراء في غرفة المعيشة."
candidate = "القطة تجلس على السجادة الحمراء."
print(compute_rouge(reference, candidate))


# ════════ [6/6] def compute_bertscore(references: list[str], candidates: list[str], la ════════


from bert_score import score

def compute_bertscore(references: list[str], candidates: list[str], lang="ar"):
    """
    يحسب BERTScore (Precision, Recall, F1).
    """
    P, R, F1 = score(candidates, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }

# مثال
references = ["القطة تجلس على السجادة الحمراء."]
candidates = ["قطة تجلس على سجادة حمراء."]
print(compute_bertscore(references, candidates))
