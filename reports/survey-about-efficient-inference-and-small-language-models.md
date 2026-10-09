# Efficient Inference and Small Language Models: A Survey
## TL;DR
- Efficient inference for language models is usually framed as reducing time, computation, memory, and storage while preserving quality, and surveys consistently organize methods into compression, decoding, and system/hardware optimization [1][2][3].
- Classic compression methods remain central: ALBERT reduces parameters with weight-sharing and factorization, I-BERT performs integer-only inference with 2.4-4.0× speedup on a T4 GPU, and later surveys still treat quantization, pruning, and distillation as core levers [4][5][6].
- Recent compact-model work shows that pruning plus distillation can be used to derive 8B and 4B models from a 15B teacher with up to 40× fewer training tokens per student than training from scratch [7].
- For serving, speculative decoding targets autoregressive latency, while PowerInfer shows that exploiting neuron-activation locality can yield up to 11.69× better performance than llama.cpp on a consumer RTX 4090 [8][9].
- Small language models are now studied as a distinct category: a recent survey covers 70 open-source models in the 100M–5B range, benchmarks on-device runtime, memory, and energy, and pairs model-quality evaluation with deployment cost analysis [10].
## Background
Efficient inference asks how to make language models faster and cheaper at deployment without losing too much accuracy. The core constraints are latency, throughput, memory footprint, storage, and energy use [1][11]. Early work such as ALBERT showed that parameter sharing and factorization can shrink models substantially while keeping strong GLUE, RACE, and SQuAD results [4]. Subsequent work generalized the toolbox to quantization, pruning, knowledge distillation, early exit, token skipping, and other compression or acceleration methods [1][6].

The topic matters now because deployment has become a first-class benchmark problem. MLPerf Inference specifies dataset and quality targets together with latency and throughput constraints, and also tracks power measurements for server and offline scenarios [12][11]. That puts model design, decoding algorithms, and serving systems into the same optimization problem rather than treating them separately [2][3].

## Compression and Compact Architectures
A long-running theme is reducing the model itself. ALBERT uses parameter-reduction techniques to lower memory consumption and increase training speed, yet still reports state-of-the-art results on GLUE, RACE, and SQuAD [4]. The broader compression literature then expanded to low-rank factorization, pruning, quantization, distillation, early exit, and token skipping, with surveys emphasizing that efficient inference should be measured in time, compute, memory, and disk usage rather than only parameter count [1][6].

Recent surveys are more explicit that many practical compression pipelines avoid expensive retraining. The 2024 survey on LLM compression groups methods into quantization, pruning, distillation, compact architecture design, and dynamic networks, and notes that tuning-free methods are attractive because finetuning or retraining is costly [6]. The companion survey on efficient inference similarly places quantization-aware training, sparsification, early exiting, and knowledge distillation alongside system-level methods [2].

A recent concrete example is Compact Language Models via Pruning and Knowledge Distillation, which combines depth, width, attention, and MLP pruning with distillation-based retraining. It reports that deriving 8B and 4B models from a pretrained 15B model can require up to 40× fewer training tokens per student than training from scratch, while outperforming prior compression techniques [7].

## Quantization and Low-Precision Execution
Quantization is one of the most important deployment techniques because it directly reduces memory traffic and arithmetic cost. I-BERT is a representative early result: it performs integer-only inference with integer approximations for GELU, Softmax, and Layer Normalization, and reports similar or slightly higher GLUE accuracy than full precision together with a 2.4-4.0× speedup for INT8 inference on a T4 GPU [5].

More recent surveys describe quantization as a central inference optimization and highlight tuning-free approaches and quantization-aware training as practical ways to lower cost [6][2]. Those surveys also connect quantization to LoRA-based adaptation after quantization, showing that low-precision execution and lightweight finetuning are now often combined rather than treated as separate design choices [6].

The practical point is that quantization helps not just by shrinking weight storage but also by making the serving stack friendlier to limited-memory devices and cheaper accelerators. That is why it appears in both model-centric compression surveys and system-centric inference surveys [1][2].

## Decoding and Serving Systems
Not all efficiency gains come from smaller weights. A major line of work attacks autoregressive decoding itself. The speculative decoding survey describes the method as drafting multiple future tokens and then verifying them in parallel, with the goal of lowering inter-token latency in memory-bound workloads [8]. The vLLM documentation lists several concrete variants—EAGLE, MTP, draft models, parallel draft models, MLP speculators, n-gram, and suffix decoding—showing that speculative decoding has become a family of implementable serving strategies rather than a single algorithm [13].

At the system level, PowerInfer takes a different route: it exploits a power-law distribution in neuron activation, placing hot neurons on the GPU and cold neurons on the CPU. The paper reports up to 11.69× better performance than llama.cpp on a single RTX 4090, while also reaching 82% of A100 token-generation rate for OPT-30B [9][14]. This is a useful reminder that inference speed is often limited by memory locality and data movement, not only by FLOPs [3].

Roofline-style analysis makes this point explicit. The LLM Inference Unveiled survey frames inference as a balance between parameter reduction, fast decoding, system-level optimization, and hardware-level optimization, and it explicitly includes early exiting, mixture-of-experts, dynamic sparsity, and heterogeneous CPU/GPU serving [3].

## What Makes a Small Language Model
Small language models are now studied as a separate object of analysis rather than just “smaller versions” of large models. The survey A Survey of Small Language Models covers architectures, training methods, compression techniques, benchmark datasets, and evaluation metrics, and it proposes a taxonomy that includes model compression, pruning, and quantization [15].

A recent measurement study sharpens this view by surveying 70 open-source models in the 100M–5B parameter range and testing commonsense reasoning, mathematics, in-context learning, and long-context performance [10]. It also benchmarks prefill and decode speed, memory footprint, and energy consumption on edge devices such as Jetson Orin and smartphones, using llama.cpp to reduce inference-engine confounds [10]. That combination of quality and deployment metrics is important because a model is only “small” in practice if it fits a hardware budget and still performs adequately.

Training data and instruction tuning now matter as much as architecture. SmolLM2 is a 1.7B model trained on about 11T tokens with a multi-stage process that mixes web text with specialized math, code, and instruction-following data; the paper introduces FineMath, Stack-Edu, and SmolTalk to address dataset quality problems [16]. Shakti, a 2.5B model aimed at smartphones and other constrained devices, emphasizes edge AI, low latency, and vernacular-language coverage, and the paper highlights VGQA as a core inference-speed innovation [17].

## Trends and Open Problems
The last two years have shifted the field from “how do we compress one model?” to “how do we manage an entire deployment stack under budget constraints?” Surveys now jointly discuss quantization, pruning, distillation, compact architectures, and system-level serving, while benchmarking studies add latency, memory, and energy to the quality metrics [6][2][10].

One clear trend is that data quality and adaptation are becoming decisive for small models. SmolLM2’s data-centric training, the Baby Llama distillation result, and the continued-pretraining study on a 125M model all suggest that high-quality or domain-specific data can compensate for smaller parameter counts [16][18][19].

Another trend is that evaluation is becoming more deployment-aware and more process-aware. MLPerf formalizes latency, throughput, and power measurements for systems, while recent SLM work adds on-device runtime and energy benchmarks; at the same time, ReTraceQA argues that answer-only metrics can overestimate capability because 14-24% of instances can have the right answer but flawed reasoning traces [12][10][20].

Open problems remain. Surveys still point to the difficulty of balancing accuracy against hardware constraints, and the system literature suggests that no single technique dominates across all hardware and workloads [2][3]. In practice, the most promising direction appears to be hybrid: compress the model, train it on better data, and serve it with decoding and hardware optimizations tuned to the target device [7][9][10].

## References
[1] A Survey on Model Compression and Acceleration for Pretrained Language Models. arxiv. https://arxiv.org/abs/2202.07105 (2022-02-01)
[2] A Survey on Efficient Inference for Large Language Models. arxiv. https://arxiv.org/abs/2404.14294 (n.d.)
[3] LLM Inference Unveiled: Survey and Roofline Model Insights. arxiv. https://arxiv.org/abs/2402.16363 (n.d.)
[4] ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. arxiv. https://arxiv.org/abs/1909.11942 (2020-02-09)
[5] I-BERT: Integer-only BERT Quantization. arxiv. https://arxiv.org/abs/2101.01321 (2021-01-05)
[6] Model Compression and Efficient Inference for Large Language Models: A Survey. arxiv. https://arxiv.org/abs/2402.09748 (2024-02-15)
[7] Compact Language Models via Pruning and Knowledge Distillation. arxiv. https://arxiv.org/abs/2407.14679 (2024-11-04)
[8] Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding. web. https://aclanthology.org/2024.findings-acl.456/ (n.d.)
[9] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. arxiv. https://arxiv.org/abs/2312.12456 (2024-12-12)
[10] Small Language Models:Survey, Measurements, and Insights. arxiv. https://arxiv.org/abs/2409.15790 (2025-02-26)
[11] Benchmark MLPerf Inference: Datacenter | MLCommons V3.1. web. https://mlcommons.org/benchmarks/inference-datacenter/ (2024-10-15)
[12] MLPerf Inference Benchmarks - MLPerf Inference Documentation. web. https://docs.mlcommons.org/inference/ (n.d.)
[13] Speculative Decoding - vLLM Documentation. web. https://docs.vllm.ai/en/stable/features/speculative_decoding/ (n.d.)
[14] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles. web. https://dl.acm.org/doi/10.1145/3694715.3695964 (2024-11-15)
[15] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[16] SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. hf-search. https://huggingface.co/papers/2502.02737 (2025-02-04)
[17] Shakti: A 2.5 Billion Parameter Small Language Model Optimized for Edge AI and Low-Resource Environments. hf-search. https://huggingface.co/papers/2410.11331 (2024-10-15)
[18] Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. arxiv. https://arxiv.org/abs/2308.02019 (2023-08-03)
[19] Domain-Adaptive Continued Pre-Training of Small Language Models. hf-search. https://huggingface.co/papers/2504.09687 (2025-04-13)
[20] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering. hf-search. https://huggingface.co/papers/2510.09351 (2025-10-13)
