# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR
- RLHF established the basic post-training recipe for language models through demonstrations and reward modeling, and later surveys describe it as the dominant alignment pipeline [1][2].
- The major shift since 2024 is toward reasoning-first RL: DeepSeek-R1 shows that pure RL with GRPO can improve reasoning, while OpenAI o1 uses large-scale RL to reason with chain-of-thought and spend more time thinking before answering [3][4][5].
- Recent methods increasingly replace coarse outcome-only rewards with process or verifier signals, including process reward models and automated process verifiers, to improve exploration and test-time search [6][7][8].
- Evaluation is now dominated by math, code, and hard reasoning benchmarks such as AIME 2024, MATH-500, LiveCodeBench, GPQA Diamond, HumanEval+, and MBPP+, where RL often improves pass@1 but not always pass@k [9][10][11].
- Open problems include reward hacking, expensive feedback collection, compute-heavy training, and uncertainty about whether long-horizon reasoning gains generalize beyond verifiable domains [12][8][10].

## Background
Reinforcement learning for LLM reasoning is the use of RL-style post-training to make language models better at multi-step inference, especially in domains where answers can be checked by preferences, rules, unit tests, or final-answer verification. The foundational pipeline was crystallized by InstructGPT, which combines demonstrations, a learned reward model, and PPO to align model outputs with human preferences [1]. OpenAI’s instruction-following post and Anthropic’s Constitutional AI show the early alignment lineage: human or AI feedback is converted into reward signals, then optimized through RL or related self-improvement loops [13][14].

Why this matters now is that RL has moved from generic instruction following to reasoning-centric post-training. Recent surveys describe a shift from user-centric RLHF to reasoning-oriented RL with verifiable rewards, process supervision, and stronger emphasis on math and coding [15][11][8]. DeepSeek-R1 and OpenAI o1 made this shift visible in practice: both frame reasoning as a trainable capability that improves when models are given more time, stronger RL signals, and, in DeepSeek-R1’s case, a multi-stage pipeline around a pure-RL core [3][4][5].

## From RLHF to AI feedback and multi-stage post-training
The original RLHF recipe is still the clearest baseline. InstructGPT’s three-step pipeline—demonstrations, reward modeling from comparisons, and PPO optimization—became the canonical template for aligning LLMs [1]. OpenAI’s instruction-following post states the same structure and emphasizes that the 1.3B InstructGPT model was preferred to the 175B GPT-3 model by labelers, despite being far smaller [13]. Anthropic’s Constitutional AI extends this lineage by replacing human labels with AI feedback and adding a supervised self-critique/revision stage before the RL phase [14].

Surveys from 2024–2025 show that this family has diversified but not disappeared. One survey describes RLHF/PPO as the dominant alignment pipeline and positions RLAIF as the main cost-reduction variant, while another explicitly categorizes RL by reward model, feedback source, policy type, and optimization style [16][17]. These surveys also note that direct-preference approaches such as DPO bypass the reward model, but the core RLHF idea remains central wherever the goal is to shape behavior from preference data [17][2].

## Reasoning-oriented RL: pure RL, cold start, and large reasoning models
The biggest technical change is the rise of reasoning-first RL. DeepSeek-R1 reports that reasoning ability can be incentivized with pure RL, without supervised fine-tuning as a preliminary step, and that DeepSeek-R1-Zero uses GRPO with a reward signal based on correctness of final predictions against ground truth [3]. The same work says DeepSeek-R1 adds cold-start data, rejection sampling, supervised fine-tuning, and a second RL stage to better align behavior while preserving reasoning [3].

OpenAI’s o1 materials describe a similar strategic direction, though with less implementation detail in the public posts: the system card says the models are trained with large-scale RL to reason using chain-of-thought, and the preview post says the models spend more time thinking before responding [18][4]. The “learning to reason” post adds that performance improves with more train-time compute and more time spent thinking at test time, underscoring that reasoning gains are coupled to inference-time search as well as training-time RL [5].

Recent surveys frame these systems as “large reasoning models” and argue that RL has become foundational for transforming LLMs into reasoning systems, especially in mathematics and coding [8][11]. That same survey literature emphasizes verifiable rewards: answer correctness for math and compiler or unit-test signals for code [11].

## Outcome rewards, process rewards, and verifier-guided training
A central methodological debate is whether RL should use only final outcomes or also intermediate process signals. Outcome-based RL gives a sparse signal: a final answer is correct or incorrect. DeepSeek-R1-Zero is an example, using correctness of final predictions as the reward signal [3]. This works well in verifiable domains, but it gives little guidance on how to improve a failed reasoning trace.

Process reward models try to fill that gap. The “Rewarding Progress” paper says process reward models provide stepwise feedback, measure progress under a distinct prover policy, and improve exploration and efficiency in both test-time search and online reinforcement learning [6]. The process-reward survey similarly argues that PRMs evaluate reasoning at the step or trajectory level rather than only at the end, and that they are used for test-time scaling and RL across math, code, multimodal reasoning, robotics, and agents [7].

Verifier-guided RL is another strong trend. The survey on reinforced reasoning says RL with verifiable rewards is a major driver of recent large reasoning models, and that coding tasks often use compiler or unit-test pass/fail signals [11]. The 2025 survey of RL techniques also includes RLVR and verifier-guided training among the major families, reflecting a field-wide move toward automatic supervision where a task can be checked cheaply and repeatedly [2].

## Human-AI feedback, self-play, and specialized reasoning domains
Not all reasoning RL is limited to automatic verifiers. The physics problem-solving paper on RL with Human-AI Feedback reports a hybrid approach that combines reinforcement learning with human-AI feedback and uses PPO, DPO, and ReMAX-style optimization, with improved performance on physics questions, especially reasoning and accuracy [19]. This illustrates an intermediate point between pure human preference alignment and fully automatic verification.

Self-play is a further extension. The RLSP paper says reinforcement learning via self-play improves reasoning, decouples exploration and correctness signals during training, and produces emergent behaviors such as backtracking and verification [20]. Compared with vanilla RLHF, self-play aims to generate its own training data and training dynamics, reducing dependence on static preference datasets [20].

These method families are often most effective in narrow domains where answers can be checked. Surveys repeatedly highlight math and coding as the main verifiable settings, because the model can be rewarded for correctness or executable outputs rather than subjective quality [11][8].

## Evaluation and empirical results
Benchmark choice has become more important as reasoning RL has matured. DeepSeek-R1’s public evaluation uses AIME 2024, MATH-500, LiveCodeBench, Codeforces, and GPQA Diamond, and the Hugging Face model card reports 79.8 on AIME 2024 pass@1, 97.3 on MATH-500 pass@1, 65.9 on LiveCodeBench pass@1-COT, 96.3 on Codeforces percentile, and 71.5 on GPQA Diamond pass@1 [9]. The same card notes the evaluation setup for sampling-based benchmarks: temperature 0.6, top-p 0.95, and 64 responses per query to estimate pass@1 [9].

The Nature paper on DeepSeek-R1 reports that DeepSeek-R1-Zero’s AIME 2024 pass@1 rose from 15.6% to 77.9% during RL training, and that self-consistency decoding further improved it to 86.7% [21]. It also reports that DeepSeek-R1 outperforms DeepSeek-V3 on several reasoning benchmarks, while only modestly improving some general-purpose benchmarks [21]. Together these results suggest that RL can produce very large gains on reasoning tasks, but the gains are benchmark-sensitive.

More skeptical evidence appears in the RLVR boundary study, which says RLVR-trained models beat base models at small k such as pass@1, but base models can achieve higher pass@k when k is large [10]. That result implies RL may sharpen the first answer more than it expands the broader search space of correct reasoning paths. The same paper uses math benchmarks such as GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23 and code benchmarks such as LiveCodeBench v5, HumanEval+, and MBPP+ [10].

## Trends and open problems
The last two years have shifted the field from generic preference optimization to reasoning-centric RL pipelines. Publicly visible systems like OpenAI o1 and DeepSeek-R1 treat reasoning as a capability to be trained, not just elicited, and they rely on larger-scale RL, more thinking time, and more explicit reasoning traces [4][5][3]. Surveys now describe RL as foundational for large reasoning models, with verifiable rewards, PRMs, and RLVR as the main current directions [8][7][11].

The main unresolved issues are still substantial. Reward hacking remains a major concern because models can optimize proxy signals in ways that violate intent or exploit evaluation artifacts [12]. Sample efficiency and compute cost are persistent bottlenecks, especially when RL requires many rollouts, long chains of thought, or expensive human feedback [8][17]. Another open question is whether reasoning gains transfer beyond the narrow verifiable domains where rewards are cheap and precise; the RLVR boundary paper suggests that gains at pass@1 may not always imply better search at larger k, which complicates claims of deeper reasoning improvement [10].

A final challenge is oversight of long-horizon reasoning and agentic behavior. Process-reward and self-play methods improve stepwise guidance, but they also raise evaluation questions: what counts as progress, how to score partial traces, and how to compare systems across domains and hidden-chain-of-thought settings [6][7][5].

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] A Technical Survey of Reinforcement Learning Techniques for Large Language Models. arxiv. https://arxiv.org/abs/2507.04136 (2025-07-05)
[3] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[4] Introducing OpenAI o1 | OpenAI. web. https://openai.com/index/introducing-openai-o1-preview/ (2024-09-12)
[5] Learning to reason with LLMs - OpenAI. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[6] Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. hf-search. https://huggingface.co/papers/2410.08146 (2024-10-10)
[7] A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. web. https://arxiv.org/html/2510.08049v3 (n.d.)
[8] A Survey of Reinforcement Learning for Large Reasoning Models. arxiv. https://arxiv.org/abs/2505.02686 (2025-06-12)
[9] DeepSeek-R1. web. https://huggingface.co/deepseek-ai/DeepSeek-R1 (n.d.)
[10] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://papers.nips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf (n.d.)
[11] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. web. https://arxiv.org/html/2501.09686 (n.d.)
[12] A Survey of Reward Hacking in Large Language Models. web. https://openreview.net/pdf?id=ENQrLsePEa (n.d.)
[13] Aligning language models to follow instructions. web. https://openai.com/index/instruction-following/ (2022-01-27)
[14] Constitutional AI: Harmlessness from AI Feedback. web. https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback (2022-12-15)
[15] A Survey on Post-training of Large Language Models - arXiv. web. https://arxiv.org/html/2503.06072v3 (n.d.)
[16] Reinforcement Learning Enhanced LLMs: A Survey. web. https://arxiv.org/abs/2412.10400 (n.d.)
[17] Reinforcement Learning for LLM Post-Training: A Survey - arXiv. web. https://arxiv.org/abs/2407.16216 (n.d.)
[18] OpenAI o1 System Card. web. https://openai.com/index/openai-o1-system-card/ (n.d.)
[19] Enhancing LLMs for Physics Problem-Solving using Reinforcement Learning with Human-AI Feedback. hf-search. https://huggingface.co/papers/2412.06827 (2024-12-06)
[20] On the Emergence of Thinking in LLMs I: Searching for the Right Intuition. hf-search. https://huggingface.co/papers/2502.06773 (2025-02-10)
[21] DeepSeek-R1 incentivizes reasoning in LLMs through ... - Nature. web. https://www.nature.com/articles/s41586-025-09422-z.pdf (n.d.)
