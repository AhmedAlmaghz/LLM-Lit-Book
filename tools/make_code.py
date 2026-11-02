# -*- coding: utf-8 -*-
"""
make_code.py — بناء مجلد code/ من كتاب book.json:
- ملف .py خطي لكل فصل فيه كتلة بايثون واحدة أو أكثر (أسماء ASCII)
- ترويسة ثابتة: تعليمات Colab/Kaggle + تثبيت الحزم + ملاحظة CPU/GPU
- لافتة قبل كل كتلة كود تشرح موقعها في الفصل
- يتحقق من صحة كل ملف بـ py_compile، ويكتب code/README.md
"""
import sys, json, re, py_compile, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import booklib as B

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
CODE.mkdir(exist_ok=True)

SLUGS = {
    1: "neural-networks-from-scratch",
    2: "deep-learning-training-eval",
    3: "modern-training-infrastructure",
    4: "tokenization",
    5: "word-embeddings",
    6: "sentence-embeddings-rnn-lstm",
    7: "self-attention-why",
    8: "transformer-architecture",
    9: "bert-gpt-t5",
    10: "modern-transformer-variants",
    11: "compute-acceleration",
    12: "build-llm-planning",
    13: "tokenizer-data-pipeline",
    14: "build-model-gpt-block",
    15: "pretraining-loop",
    16: "sft-lora-qlora",
    17: "rlhf-dpo",
    18: "inference-generation",
    19: "quantization-distillation",
    20: "prompt-engineering",
    21: "advanced-peft",
    22: "advanced-preference-optimization",
    23: "rag",
    24: "agents-and-tools",
    25: "multimodal-intro",
    26: "vision-language-architectures",
    27: "build-vlm-arabic",
    28: "multimodal-evaluation",
    29: "build-vs-adapt-cost",
    30: "knowledge-distillation",
    31: "model-merging",
    32: "pruning",
    33: "lora-as-a-service",
    34: "self-improvement-synthetic-data",
    35: "core-evaluation-metrics",
    36: "modern-benchmarks",
    37: "llm-as-a-judge",
    38: "multimodal-agent-evaluation",
    39: "safety-privacy-bias",
    40: "deployment-production",
    41: "arabic-sentiment",
    42: "machine-translation",
    43: "arabic-rag",
    44: "long-document-summarization",
    45: "code-generation-analysis",
    46: "interview-questions",
    47: "llm-system-design-patterns",
    48: "engineering-best-practices",
}

PART_SLUGS = {
    1: "foundations-neural-nets-to-deep-learning",
    2: "embeddings-and-representations",
    3: "transformers",
    4: "build-llm-from-scratch",
    5: "advanced-techniques-and-agents",
    6: "multimodal-llms",
    7: "build-on-existing-models",
    8: "evaluation-testing-deployment",
    9: "practical-projects",
    10: "interviews-and-career",
}

HEADER = '''# -*- coding: utf-8 -*-
"""
{title}

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء {part_ord} · الفصل {num} · عدد كتل بايثون في هذا الملف: {n_blocks}

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python {fname}
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python {fname}

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \\
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""
'''


def py_blocks(c):
    return [b for b in c["blocks"] if b["t"] == "code" and b.get("lang") in ("python", "py")]


def block_banner(block, idx, total):
    code = block["code"].strip("\n")
    first = ""
    for ln in code.splitlines():
        s = ln.strip()
        if s and not s.startswith(("#", "import ", "from ")):
            first = s[:70]
            break
    if not first:
        for ln in code.splitlines():
            if ln.strip():
                first = ln.strip()[:70]
                break
    return f"# {'═' * 8} [{idx}/{total}] {first} {'═' * 8}\n"


def build_chapter_file(book, c):
    blocks = py_blocks(c)
    if not blocks:
        return None
    pt = next(p for p in book["part_titles"] if p["num"] == c["part"])
    fname = f"ch{c['num']:02d}_{SLUGS.get(c['num'], 'chapter')}.py"
    title = f"الفصل {c['num']}: {c['title']}"
    parts = [HEADER.format(title=title, part_ord=pt["ordinal"], num=c["num"],
                           n_blocks=len(blocks), fname=fname)]
    for i, b in enumerate(blocks, 1):
        parts.append(block_banner(b, i, len(blocks)))
        parts.append(b["code"].strip("\n") + "\n")
    text = "\n\n".join(parts)
    path = CODE / fname
    path.write_text(text, encoding="utf-8")
    return path


def main():
    book = B.load()
    made = []
    for c in book["chapters"]:
        p = build_chapter_file(book, c)
        if p:
            made.append((c["num"], p, len(py_blocks(c))))

    # فحص الصياغة
    fails = []
    for num, p, _ in made:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            fails.append((p.name, str(e).splitlines()[-1][:160]))

    # requirements.txt (شائعة عبر الفصول)
    req = [
        "torch>=2.3", "transformers>=4.44", "datasets>=2.20", "accelerate>=0.33",
        "peft>=0.12", "trl>=0.9", "bitsandbytes>=0.43", "sentence-transformers>=3.0",
        "faiss-cpu>=1.8", "evaluate>=0.4", "scikit-learn>=1.4", "pandas>=2.2",
        "numpy>=1.26", "matplotlib>=3.8", "tqdm>=4.66", "safetensors>=0.4",
    ]
    (CODE / "requirements.txt").write_text("\n".join(req) + "\n", encoding="utf-8")

    # README
    total_blocks = sum(n for _, _, n in made)
    lines = [
        "# مجلد الأكواد — «المحوّلات والنماذج اللغوية الكبيرة»",
        "",
        f"ملف `.py` خطي لكل فصل فيه كود بايثون ({len(made)} ملفًا، {total_blocks} كتلة). "
        "كل ملف يُقرأ من أعلاه إلى أسفله مثل دفتر Jupyter: الكتل مرتّبة كما وردت في الفصل، "
        "مع لافتة قبل كل كتلة.",
        "",
        "## التشغيل السريع",
        "",
        "```bash",
        "pip install -r code/requirements.txt",
        "python code/ch14_build-model-gpt-block.py",
        "```",
        "",
        "**Colab:** ارفع الملف إلى `/content` ثم `!python chXX_*.py`.",
        "**Kaggle:** أضف الملف كـ Script، أو استخدم الدفاتر الجاهزة في `code/notebooks/`.",
        "",
        "**CPU/GPU:** كل الأكواد تعمل على CPU، وفصول التدريب أسرع على GPU. "
        "اختر GPU من إعدادات Colab/Kaggle عند الحاجة؛ الأكواد تكتشف الجهاز تلقائيًا حيث يلزم.",
        "",
        "## الملفات",
        "",
        "| الملف | الفصل | كتل |",
        "|---|---|---|",
    ]
    for num, p, n in made:
        c = next(c for c in book["chapters"] if c["num"] == num)
        lines.append(f"| `{p.name}` | {num}. {c['title']} | {n} |")
    lines += [
        "",
        "## الدفاتر",
        "",
        "`code/notebooks/` فيه دفتر Jupyter واحد لكل جزء (10 دفاتر): خلية إعداد + "
        "خلية Markdown لكل فصل + خلية كود لكل كتلة بايثون. مناسبة للفتح مباشرة في Colab/Kaggle.",
        "",
    ]
    (CODE / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"files: {len(made)} · blocks: {total_blocks} · compile fails: {len(fails)}")
    for f in fails[:10]:
        print("FAIL:", f)


if __name__ == "__main__":
    main()
