# World Models: A Survey
## TL;DR
- World models are learned predictive models of an environment that compress observations into latent state and use imagined rollouts for planning or control; this is the core framing in the seminal World Models paper and in Dreamer-style latent imagination [1][2].
- The field has diversified into latent-dynamics control, pixel/video prediction, and non-physical LLM-centered planning formulations; a video-prediction survey contrasts pixel-level prediction with compact latent-state world models, while RAP treats language-model reasoning as planning with a world model [3][4].
- Modern latent world models increasingly replace raw reconstruction with objectives that preserve task-relevant information, helping keep the model compact and control-oriented [5].
- Evaluation remains fragmented: surveys and position pieces argue that visual realism and open-loop prediction are insufficient evidence of decision-making utility, and that closed-loop, action-conditioned evaluation is still rare [6][7][8].
- Recent surveys emphasize broad applications in robotics, simulation, games, embodied AI, and autonomous agents, but also recurring open problems: long-horizon fidelity, sim-to-real transfer, action conditioning, and uncertainty/safety [9][10].

## Background
A world model is a learned internal model of how the environment evolves under actions. In modern deep RL, the defining idea is to compress high-dimensional observations into a latent state, predict future latent states and rewards, and use those imagined futures for planning or policy learning. The seminal Ha and Schmidhuber paper popularized this formulation with a vision module, memory dynamics, and a controller trained inside the learned latent space [1][11]. Dreamer generalized the same basic recipe into a scalable latent-imagination framework: it learns a representation model, a transition model, and a reward model, then optimizes behavior by backpropagating through imagined trajectories [2].

The topic matters now because world models have expanded beyond classic reinforcement learning. Recent work applies the same predictive abstraction to embodied decision-making, robotics, multimodal language models, and agentic reasoning, while recent evaluation surveys argue that the field lacks a single accepted benchmark and that many claims outstrip the evidence provided by current tests [9][6][4].

## Latent-dynamics world models for control
The dominant control-oriented family uses compact latent state rather than raw pixels for prediction and planning. PlaNet learns a latent dynamics model from images and performs fast online planning in latent space; its formulation includes deterministic and stochastic transition components and a multi-step latent overshooting objective [12]. Dreamer follows a similar philosophy but pushes farther toward end-to-end behavior learning: it reconstructs observations, predicts future latent states from actions, and trains actor-critic behavior entirely in imagination instead of through direct environment interaction [2].

A more recent contrastive variant shows how this family is still being refined. Contrastive latent dynamics replaces Dreamer-style pixel reconstruction with a contrastive objective on future observation features, aiming to preserve information useful for planning while discarding irrelevant visual detail [5]. This reflects a broader pattern in the literature: the model is not valued for perfectly reproducing the world, but for retaining the variables that matter for downstream control [2][5].

## Video and pixel prediction as world modeling
A second family models the world directly at the level of pixels or videos. The video-prediction survey distinguishes these methods from latent world models by their output space: instead of predicting a compact latent state, they forecast future frames or pixel trajectories from visual observations [3]. This makes them closer to perceptual simulation, and the survey notes that such predictions can support planning, policy learning, and evaluation of control agents [3].

The distinction matters because video models can look more faithful visually, but they are not automatically better for decision-making. Benchmark surveys and position pieces repeatedly warn that open-loop reconstruction quality does not establish closed-loop utility, and that action-conditioned or task-conditioned metrics are needed to show whether the predicted future is actually usable for control [6][7][8]. In other words, pixel realism and control usefulness overlap but are not the same objective [3][6].

## World models for embodied decision-making and robotics
Recent surveys place world models inside a larger robotics and embodied-AI stack. The robot-learning survey says world models support policy learning, planning, simulation, evaluation, data generation, and robotic video world models; it also describes them as learned simulators for reinforcement learning and as evaluators that rank candidate actions at decision time [9]. The benchmark survey further organizes evaluation around functional utility, control fidelity, and feedback-driven interaction, showing that the field increasingly treats world models as decision-making tools rather than only as predictors [8].

At the same time, the surveys identify practical constraints. Long-horizon reliability, sim-to-real transfer, and the need for control-consistent futures remain unresolved, and many benchmarks still emphasize open-loop metrics that do not fully capture closed-loop behavior [9][7]. The more recent position paper argues that exploitability gaps are rarely measured and that policy improvement should not be inferred from generation quality alone [6].

## LLM-centered world models and planning
The term world model is now also used outside physics and robotics. RAP frames reasoning as planning with a world model, treating the language model as part of a planning stack rather than only as a next-token predictor [4]. This broadens the concept from latent dynamics over physical state to a more abstract model of search, reasoning, and action selection in symbolic or textual domains [4].

A still newer line pushes world modeling into multimodal LLMs. WOVEN describes visual transition reasoning as a shared primitive for spatial, embodied, physical, and temporal reasoning in multimodal language systems, suggesting that world modeling can be learned from different supervision sources and reused across tasks [13]. Taken together, these papers show a shift from “world model as simulator” toward “world model as reusable transition abstraction” in broader agent architectures [4][13].

## Trends and open problems
The last two years show two main trends. First, world models are being reframed as decision-making components with clearer evaluation criteria: recent position papers propose ladders of evidence from visual plausibility to policy optimization utility, and benchmark surveys attempt to standardize what counts as meaningful evaluation [6][8]. Second, LLM-centered work is broadening the term toward reasoning and multimodal agents [4][13].

Several problems remain open. Surveys repeatedly cite action conditioning, long-horizon prediction error, sim-to-real transfer, data scarcity, uncertainty, and safety as unresolved issues [9][10]. Evaluation is still the biggest methodological bottleneck: many benchmarks remain open-loop, and multiple sources argue that good generation scores do not prove that a model can improve decisions or planning [7][6][8]. The field therefore still lacks a universally accepted definition of a “good” world model beyond the task-specific criterion that it should be useful for the downstream decision problem it is meant to solve [6][9].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (n.d.)
[2] Dream to Control: Learning Behaviors by Latent Imagination. web. https://arxiv.org/abs/1912.01603 (n.d.)
[3] Video Prediction for Reinforcement Learning: A Survey. arxiv. https://arxiv.org/abs/2105.00064 (2021-05-03)
[4] Reasoning with Language Model is Planning with World Model. hf-search. https://huggingface.co/papers/2305.14992 (2023-05-24)
[5] Learning World Models with Contrastive Latent Dynamics. arxiv. https://arxiv.org/abs/2409.06739 (2024-09-10)
[6] How Should World Models Be Evaluated for Embodied Decision-Making?A Decision-Making-Centric Position. web. https://arxiv.org/abs/2606.15032 (2026-06-28)
[7] Do World Models Make Better Robots?. web. https://arxiv.org/abs/2609.29669 (n.d.)
[8] A Survey of World Model Benchmarks. web. https://world-model-benchmarks.github.io/World-Model-Benchmarks/ (n.d.)
[9] World Model for Robot Learning: A Comprehensive Survey. web. https://arxiv.org/abs/2605.00080 (n.d.)
[10] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (n.d.)
[11] World Models (Ha & Schmidhuber) | Seminal 2018 Paper |.... web. https://world-models.io/en/models/ha-world-model/ (n.d.)
[12] Learning Latent Dynamics for Planning from Pixels. web. https://proceedings.mlr.press/v97/hafner19a.html (2019-05-24)
[13] WOVEN: Weaving Visual World Modeling into Multimodal LLMs. arxiv. https://arxiv.org/abs/2610.12417 (2026-10-08)
