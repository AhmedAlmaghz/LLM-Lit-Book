# -*- coding: utf-8 -*-
"""
الفصل 41: تحليل المشاعر بالعربية — عندما تصبح "ما شاء الله" سخرية

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء التاسع · الفصل 41 · عدد كتل بايثون في هذا الملف: 9

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch41_arabic-sentiment.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch41_arabic-sentiment.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/9] def normalize_arabic(text: str) -> str: ════════


import pandas as pd
import re
import unicodedata

def normalize_arabic(text: str) -> str:
    """تطبيع عربي بسيط."""
    if not isinstance(text, str):
        return ""
    # إزالة التطويل
    text = re.sub(r"\u0640", "", text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # توحيد الهمزات
    text = re.sub(r"[إأآٱ]", "ا", text)
    # توحيد التاء المربوطة
    text = text.replace("ة", "ه")
    # توحيد الأرقام العربية
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    # إزالة الروابط
    text = re.sub(r"http\S+|www\S+", "", text)
    # إزالة الإشارات (@mentions)
    text = re.sub(r"@\w+", "", text)
    # توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path: str) -> pd.DataFrame:
    """يحمل بيانات ويطبع إحصاءات."""
    df = pd.read_csv(path)
    # افترض أن الأعمدة: text, label (positive/negative/neutral)
    df["text"] = df["text"].apply(normalize_arabic)
    df = df[df["text"].str.len() > 3]  # إزالة النصوص القصيرة جدًا
    df = df.drop_duplicates(subset=["text"])
    print(f"Loaded {len(df)} samples")
    print(df["label"].value_counts())
    return df

# مثال
# df = load_dataset("astd.csv")


# ════════ [2/9] def emoji_to_text(text: str) -> str: ════════


import emoji

def emoji_to_text(text: str) -> str:
    """يحول الإيموجي إلى وصف نصي."""
    return emoji.demojize(text, language="ar")

# مثال
print(emoji_to_text("الخدمة سيئة 😡"))  # "الخدمة سيئة :وجه_غاضب:"


# ════════ [3/9] def train_tfidf_baseline(df: pd.DataFrame): ════════


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

def train_tfidf_baseline(df: pd.DataFrame):
    """يدرب خط أساس TF-IDF + LR."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            analyzer="word",
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    return pipeline, X_test, y_test


# ════════ [4/9] AutoModelForSequenceClassification, ════════


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import torch

def prepare_dataset(df: pd.DataFrame, tokenizer, label2id: dict, max_length: int = 128):
    """يحول DataFrame إلى Dataset."""
    ds = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.map(lambda x: {"labels": label2id[x["label"]]})
    ds = ds.remove_columns(["text", "label"])
    return ds

def train_small_model(
    df: pd.DataFrame,
    model_name: str = "aubmindlab/bert-base-arabertv02",
    output_dir: str = "./sentiment_model",
):
    """يدرب نموذج تصنيف مشاعر عربي."""
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # تقسيم
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_ds = prepare_dataset(train_df, tokenizer, label2id)
    val_ds = prepare_dataset(val_df, tokenizer, label2id)

    # إعدادات التدريب
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


# ════════ [5/9] def prepare_chat_dataset(df: pd.DataFrame, tokenizer): ════════


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def prepare_chat_dataset(df: pd.DataFrame, tokenizer):
    """يحول البيانات إلى صيغة محادثة."""
    def format_example(row):
        return {
            "messages": [
                {"role": "system", "content": "أنت مصنف مشاعر. صنّف النص إلى: إيجابي، سلبي، محايد. أعد كلمة واحدة فقط."},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["label"]},
            ]
        }

    ds = Dataset.from_pandas(df)
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds

def train_lora_sentiment(df: pd.DataFrame):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = prepare_chat_dataset(df, tokenizer)

    args = SFTConfig(
        output_dir="./sentiment_lora",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./sentiment_lora_final")
    return model, tokenizer


# ════════ [6/9] client = OpenAI() ════════


from openai import OpenAI
client = OpenAI()

def classify_with_gpt(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """أنت مصنف مشاعر عربي خبير. صنّف النص إلى: إيجابي، سلبي، محايد.
انتبه للتهكم: "ما شاء الله، وصل بعد شهر" هي سلبية.
انتبه للهجات: "مبسوط" = إيجابي، "مش بطال" = محايد/إيجابي.
أعد كلمة واحدة فقط."""},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


# ════════ [7/9] def detect_sarcasm_heuristic(text: str) -> bool: ════════


def detect_sarcasm_heuristic(text: str) -> bool:
    """كشف تهكم بسيط ( heuristic)."""
    sarcasm_markers = [
        r"ما\s+شاء\s+الله.*(?:بعد|تأخر|انتظر)",
        r"الله\s+يعطيك\s+العافية.*(?:بعد|تأخر|انتظر)",
        r"(?:ممتاز|رائع|جميل).*!!!!",
        r"😍.*(?:سيء|سيئه|تأخر|فاشل)",
        r"(?:سيء|فاشل).*😍",
    ]
    for pattern in sarcasm_markers:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ════════ [8/9] dialect_mapping = { ════════


dialect_mapping = {
    # خليجي
    "مبسوط": "سعيد",
    "زين": "جيد",
    "ما يخالف": "لا مشكلة",
    "الله يعافيك": "شكرًا",
    # مصري
    "مش بطال": "ليس سيئًا",
    "تحفة": "رائع",
    "زي الفل": "ممتاز",
    # شامي
    "منيح": "جيد",
    "كتير حلو": "جميل جدًا",
}

def normalize_dialect(text: str) -> str:
    """يحول بعض كلمات اللهجة إلى الفصحى."""
    for dialect, fusha in dialect_mapping.items():
        text = text.replace(dialect, fusha)
    return text


# ════════ [9/9] app = FastAPI() ════════


from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# تحميل النموذج
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    confidence: float

@app.post("/classify", response_model=Response)
def classify(req: Request):
    text = normalize_arabic(req.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = probs.argmax().item()
    return Response(
        label=model.config.id2label[pred],
        confidence=probs[pred].item(),
    )

@app.get("/health")
def health():
    return {"status": "ok"}
