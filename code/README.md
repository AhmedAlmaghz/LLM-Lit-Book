# مجلد الأكواد — «المحوّلات والنماذج اللغوية الكبيرة»

ملف `.py` خطي لكل فصل فيه كود بايثون (47 ملفًا، 364 كتلة). كل ملف يُقرأ من أعلاه إلى أسفله مثل دفتر Jupyter: الكتل مرتّبة كما وردت في الفصل، مع لافتة قبل كل كتلة.

## التشغيل السريع

```bash
pip install -r code/requirements.txt
python code/ch14_build-model-gpt-block.py
```

**Colab:** ارفع الملف إلى `/content` ثم `!python chXX_*.py`.
**Kaggle:** أضف الملف كـ Script، أو استخدم الدفاتر الجاهزة في `code/notebooks/`.

**CPU/GPU:** كل الأكواد تعمل على CPU، وفصول التدريب أسرع على GPU. اختر GPU من إعدادات Colab/Kaggle عند الحاجة؛ الأكواد تكتشف الجهاز تلقائيًا حيث يلزم.

## الملفات

| الملف | الفصل | كتل |
|---|---|---|
| `ch01_neural-networks-from-scratch.py` | 1. الشبكات العصبية من الصفر — كيف تتعلم الآلة أن “تخمّن” بشكل أفضل | 6 |
| `ch02_deep-learning-training-eval.py` | 2. مفاهيم التعلّم العميق للتدريب والتقييم — كيف تعرف أن نموذجك “يتعلم” فعلًا؟ | 14 |
| `ch03_modern-training-infrastructure.py` | 3. بنية التدريب الحديثة — من GPU واحدة إلى عنقود | 9 |
| `ch04_tokenization.py` | 4. الترميز إلى وحدات (Tokenization) — كيف يقرأ النموذج ما لا تراه عينك | 6 |
| `ch05_word-embeddings.py` | 5. تضمينات الكلمات (Word2Vec وGloVe) — كيف تتحول الكلمة إلى إحداثيات في فضاء المعنى | 8 |
| `ch06_sentence-embeddings-rnn-lstm.py` | 6. تضمينات الجُمل — RNN وLSTM وGRU: كيف تتذكر الشبكة ما نسيته؟ | 9 |
| `ch07_self-attention-why.py` | 7. الدافع وراء آلية الانتباه الذاتي — اللحظة التي توقفت فيها الشبكة عن النسيان | 3 |
| `ch08_transformer-architecture.py` | 8. بنية المحوّل — من كتلة واحدة إلى معمارية كاملة | 4 |
| `ch09_bert-gpt-t5.py` | 9. التنويعات الكبرى — BERT وGPT وT5: ثلاث فلسفات في بناء العقل | 3 |
| `ch10_modern-transformer-variants.py` | 10. ما بعد المحوّل الكلاسيكي — FlashAttention وMLA وMoE وMamba | 3 |
| `ch11_compute-acceleration.py` | 11. تسريع العمليات الحسابية — عندما تصبح السرعة قرارًا هندسيًا | 1 |
| `ch12_build-llm-planning.py` | 12. بناء LLM من الصفر — تخطيط المشروع وإعداد البيئة (مع كود كامل لكل خطوة) | 13 |
| `ch13_tokenizer-data-pipeline.py` | 13. بناء Tokenizer وتجهيز البيانات بعمق — المطبخ الخفي الذي يحدد جودة النموذج | 12 |
| `ch14_build-model-gpt-block.py` | 14. بناء النموذج — من Embedding إلى GPT Block (كود كامل، مختبر، خطوة بخطوة) | 13 |
| `ch15_pretraining-loop.py` | 15. حلقة التدريب (Pre-training) — حيث يتحول النموذج من ورق إلى عقل | 16 |
| `ch16_sft-lora-qlora.py` | 16. الضبط الدقيق (SFT) والضبط الفعّال (LoRA/QLoRA) — كيف تعلّم نموذجًا أن يطيع دون أن تعيد بناءه | 6 |
| `ch17_rlhf-dpo.py` | 17. ضبط التفضيلات (RLHF وDPO) — كيف تعلّم النموذج أن يختار "الأفضل" لا "الصحيح" فقط | 8 |
| `ch18_inference-generation.py` | 18. الاستدلال والتوليد — كيف يختار النموذج كلمته التالية؟ | 11 |
| `ch19_quantization-distillation.py` | 19. التكميم والتقطير — كيف تصنع نموذجًا أصغر لا يخسر عقله | 5 |
| `ch20_prompt-engineering.py` | 20. هندسة الأوامر (Prompt Engineering) — فن مخاطبة عقل لا تراه | 9 |
| `ch21_advanced-peft.py` | 21. الضبط الدقيق الفعّال (PEFT) المتقدم — عندما يكون "الأقل" أكثر | 7 |
| `ch22_advanced-preference-optimization.py` | 22. ضبط التفضيلات المتقدم — ORPO وKTO وSimPO وGRPO: عائلة DPO وأبناء العمومة | 6 |
| `ch23_rag.py` | 23. التوليد المعزّز بالاسترجاع (RAG) — كيف تجعل النموذج يجيب من كتب لم يقرأها | 10 |
| `ch24_agents-and-tools.py` | 24. الوكلاء (Agents) واستخدام الأدوات — عندما يتحول النموذج من مُجيب إلى فاعل | 5 |
| `ch25_multimodal-intro.py` | 25. مدخل إلى النماذج متعددة الوسائط (Multimodal) — عندما يفتح النموذج عينيه | 8 |
| `ch26_vision-language-architectures.py` | 26. معماريات Vision-Language المتقدمة — خمس فلسفات في رؤية واحدة | 5 |
| `ch27_build-vlm-arabic.py` | 27. بناء VLM عمليًا من نموذج موجود — من الصفر إلى نموذج يرى بالعربية | 13 |
| `ch28_multimodal-evaluation.py` | 28. تقييم النماذج متعددة الوسائط — كيف تعرف أن النموذج “يرى” فعلًا؟ | 6 |
| `ch29_build-vs-adapt-cost.py` | 29. لماذا لا تبدأ من الصفر؟ — التكلفة والوقت والبيانات: حساب الصفقة | 1 |
| `ch30_knowledge-distillation.py` | 30. التقطير (Knowledge Distillation) — كيف تعلّم نموذجًا صغيرًا أن يفكر مثل معلمه | 5 |
| `ch31_model-merging.py` | 31. الدمج (Model Merging) — عندما يصبح 1+1 أكبر من 2 | 7 |
| `ch32_pruning.py` | 32. التشذيب (Pruning) واستعادة الجودة — كيف تحذف من النموذج دون أن تفقد عقله | 8 |
| `ch33_lora-as-a-service.py` | 33. التخصيص السريع (Adapters وLoRA) كنمط بناء — كيف تخدم مئة عميل بنموذج واحد | 6 |
| `ch34_self-improvement-synthetic-data.py` | 34. Self-Improvement وSynthetic Data — كيف يتفوق النموذج على نفسه | 5 |
| `ch35_core-evaluation-metrics.py` | 35. مقاييس التقييم الأساسية — الأرقام التي تخبرك الحقيقة (وأحيانًا تكذب) | 6 |
| `ch36_modern-benchmarks.py` | 36. المعايير الحديثة — MMLU وBIG-Bench وHELM: كيف نقيس "الذكاء العام"؟ | 5 |
| `ch37_llm-as-a-judge.py` | 37. LLM-as-a-Judge — عندما يصبح النموذج قاضيًا على نفسه | 10 |
| `ch38_multimodal-agent-evaluation.py` | 38. تقييم النماذج متعددة الوسائط والوكلاء — كيف تقيس "الرؤية" و"الفعل"؟ | 8 |
| `ch39_safety-privacy-bias.py` | 39. الأمان والخصوصية والتحيز — عندما يصبح النموذج خطرًا دون أن يدري | 11 |
| `ch40_deployment-production.py` | 40. النشر (Deployment) — من المختبر إلى الإنتاج: كيف تبني نظامًا لا ينهار | 10 |
| `ch41_arabic-sentiment.py` | 41. تحليل المشاعر بالعربية — عندما تصبح "ما شاء الله" سخرية | 9 |
| `ch42_machine-translation.py` | 42. الترجمة الآلية — عندما تخون الكلمة مترجمها | 10 |
| `ch43_arabic-rag.py` | 43. RAG للمستندات العربية — عندما تقرأ الآلة ما كتبه البشر بيدهم | 11 |
| `ch44_long-document-summarization.py` | 44. تلخيص المستندات الطويلة — عندما ينسى النموذج ما في منتصف الكتاب | 8 |
| `ch45_code-generation-analysis.py` | 45. توليد الكود وتحليله — عندما يكتب النموذج كودًا يعمل، لكنه يقتل | 11 |
| `ch47_llm-system-design-patterns.py` | 47. أنماط تصميم أنظمة LLM — عندما يصبح النموذج مجرد لبنة | 9 |
| `ch48_engineering-best-practices.py` | 48. أفضل الممارسات الهندسية والمسار المهني — الرحلة لا تنتهي عند النشر | 5 |

## الدفاتر

`code/notebooks/` فيه دفتر Jupyter واحد لكل جزء (10 دفاتر): خلية إعداد + خلية Markdown لكل فصل + خلية كود لكل كتلة بايثون. مناسبة للفتح مباشرة في Colab/Kaggle.
