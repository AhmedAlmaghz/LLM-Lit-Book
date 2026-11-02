# -*- coding: utf-8 -*-
"""
الفصل 9: التنويعات الكبرى — BERT وGPT وT5: ثلاث فلسفات في بناء العقل

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثالث · الفصل 9 · عدد كتل بايثون في هذا الملف: 3

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch09_bert-gpt-t5.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch09_bert-gpt-t5.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/3] class BertClassifier(nn.Module): ════════


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # نستخدم تمثيل [CLS] (أول رمز) كملخص للجملة
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

# مثال للاستخدام
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["I love this movie", "This film is terrible"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

model = BertClassifier()
logits = model(batch["input_ids"], batch["attention_mask"])
print(logits.softmax(dim=-1))  # احتمالات الفئتين


# ════════ [2/3] class GPTBlock(nn.Module): ════════


class GPTBlock(nn.Module):
    """كتلة GPT: انتباه سببي + FFN، بدون انتباه متقاطع."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ════════ [3/3] tokenizer = T5Tokenizer.from_pretrained("t5-small") ════════


from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "translate English to French: The cat is sitting on the mat."
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# "Le chat est assis sur le tapis."
