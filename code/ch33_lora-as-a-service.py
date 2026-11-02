# -*- coding: utf-8 -*-
"""
الفصل 33: التخصيص السريع (Adapters وLoRA) كنمط بناء — كيف تخدم مئة عميل بنموذج واحد

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء السابع · الفصل 33 · عدد كتل بايثون في هذا الملف: 6

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch33_lora-as-a-service.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch33_lora-as-a-service.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/6] class MultiLoRAServer: ════════


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class MultiLoRAServer:
    def __init__(self, base_model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.adapters = {}  # name -> PeftModel

    def load_adapter(self, name: str, adapter_path: str):
        """تحميل LoRA وتخزينها."""
        if name in self.adapters:
            return
        model = PeftModel.from_pretrained(
            self.base_model, adapter_path, adapter_name=name
        )
        self.adapters[name] = model
        print(f"Loaded adapter: {name}")

    def switch_adapter(self, name: str):
        """تبديل المحول النشط."""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not loaded")
        self.base_model.set_adapter(name)

    def generate(self, prompt: str, adapter_name: str, **kwargs):
        """توليد باستخدام محول محدد."""
        if adapter_name not in self.adapters:
            self.load_adapter(adapter_name, f"./adapters/{adapter_name}")

        self.switch_adapter(adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.base_model.generate(**inputs, **kwargs)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# مثال
server = MultiLoRAServer("Qwen/Qwen2.5-1.5B")
server.load_adapter("customer_a", "./adapters/customer_a")
server.load_adapter("customer_b", "./adapters/customer_b")

print(server.generate("مرحبًا، كيف حالك؟", adapter_name="customer_a"))
print(server.generate("مرحبًا، كيف حالك؟", adapter_name="customer_b"))


# ════════ [2/6] client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy") ════════


from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# استخدم LoRA محدد عبر model
response = client.chat.completions.create(
    model="customer_a",  # اسم LoRA
    messages=[{"role": "user", "content": "مرحبًا"}],
)
print(response.choices[0].message.content)


# ════════ [3/6] def weighted_sum_loras(base_model, lora_paths: list[str], weights: lis ════════


def weighted_sum_loras(base_model, lora_paths: list[str], weights: list[float]):
    """
    دمج عدة LoRAs بأوزان محددة.
    """
    from peft import PeftModel
    import torch

    base_sd = base_model.state_dict()
    merged = {k: v.clone() for k, v in base_sd.items()}

    for path, w in zip(lora_paths, weights):
        lora = PeftModel.from_pretrained(base_model, path)
        lora_sd = lora.state_dict()

        for k in lora_sd:
            if "lora_A" in k or "lora_B" in k:
                # أوزان LoRA: أضف المساهمة المرجحة
                if k in merged:
                    merged[k] = merged[k] + w * lora_sd[k]
                else:
                    merged[k] = w * lora_sd[k]

    merged_model = AutoModelForCausalLM.from_pretrained(base_model.config._name_or_path)
    merged_model.load_state_dict(merged)
    return merged_model


# ════════ [4/6] def stack_loras(base_model, lora_paths: list[str]): ════════


from peft import PeftModel

def stack_loras(base_model, lora_paths: list[str]):
    """
    تكديس LoRAs: كل واحد يعدل النموذج بشكل متسلسل.
    """
    model = base_model
    for path in lora_paths:
        model = PeftModel.from_pretrained(model, path)
    return model


# ════════ [5/6] class LoRARouter: ════════


class LoRARouter:
    def __init__(self, adapter_descriptions: dict[str, str], embedder):
        """
        adapter_descriptions: {name: description}
        embedder: نموذج تضمين
        """
        self.names = list(adapter_descriptions.keys())
        self.descriptions = list(adapter_descriptions.values())
        self.embeddings = embedder.encode(self.descriptions, normalize_embeddings=True)
        self.embedder = embedder

    def route(self, query: str) -> str:
        """يختار LoRA بناءً على تشابه الطلب مع الوصف."""
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_emb
        best_idx = scores.argmax()
        return self.names[best_idx]

# مثال
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

router = LoRARouter(
    {
        "math": "أسئلة رياضية، حسابات، معادلات",
        "code": "برمجة، كود، خوارزميات",
        "arabic": "أسئلة بالعربية الفصحى",
        "customer_a": "أسئلة عن منتجات شركة A",
    },
    embedder,
)

print(router.route("كيف أحل معادلة من الدرجة الثانية؟"))  # math
print(router.route("اكتب دالة بايثون لجمع رقمين"))  # code


# ════════ [6/6] class LoRACache: ════════


from collections import OrderedDict
import torch

class LoRACache:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.cache = OrderedDict()  # name -> PeftModel

    def get(self, name: str):
        if name in self.cache:
            self.cache.move_to_end(name)  # اجعله الأحدث
            return self.cache[name]
        return None

    def put(self, name: str, model):
        if name in self.cache:
            self.cache.move_to_end(name)
        else:
            if len(self.cache) >= self.max_size:
                # أفرغ الأقدم (LRU)
                oldest, _ = self.cache.popitem(last=False)
                print(f"Evicted: {oldest}")
            self.cache[name] = model

    def __contains__(self, name):
        return name in self.cache
