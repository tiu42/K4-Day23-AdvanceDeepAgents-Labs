# LLM Agents and Tool Use
## TL;DR
- LLM agents are best understood as systems that perceive an environment, plan, and execute actions, often by calling external tools in a loop rather than generating standalone text [1][2][3].
- ReAct-style interleaving of reasoning and action, and Toolformer-style learned API invocation, represent two major design points: one emphasizes explicit trajectories and interpretability, the other learns when to call tools from data [2][4].
- Modern tool-use interfaces increasingly expose structured function calling or programmatic tool execution, which aim to make model outputs reliably convertible into API calls and reduce latency or token use in multi-tool workflows [5][6].
- Evaluation has moved beyond static accuracy toward dynamic, sequential benchmarks such as Mind2Web, WebArena-Verified, AgentBoard, and SWE-bench Verified, because agent performance depends on multi-step interaction and environment feedback [7][8][9][10][11].
- Recent surveys emphasize that the field is broadening into software engineering, web automation, and OS/browser agents, while open problems remain cost, robustness, safety, grounding, and trustworthy deployment [12][13][14][15].

## Background
LLM agents are systems that use language models as controllers or decision-makers for interacting with environments, tools, or external systems. Across the surveyed literature, the common thread is that an agent does not merely answer a prompt: it perceives state, plans or decomposes a task, acts through tools, and often incorporates feedback or memory to continue [1][3][12]. ReAct makes this interaction explicit by interleaving reasoning traces with task-specific actions, while Toolformer frames tool use as learning when and how to call APIs [2][4].

This topic matters now because the field has moved from single-turn prompting to long-horizon, tool-using workflows that need structured outputs, external information, and action execution. Surveys from 2024–2025 describe a rapidly expanding ecosystem covering planning, tool use, feedback learning, and application domains including software engineering and web automation [3][12][13][15].

## Foundations and conceptual taxonomy
One consistent conceptual distinction is between agents that plan first and agents that act while reasoning. The planning survey defines autonomous agents as entities that accomplish tasks by perceiving the environment, planning, and executing actions, and it organizes planning methods into task decomposition, plan selection, external module use, reflection, and memory [1]. The LLM-agent paradigm survey similarly treats tool use, planning, and feedback learning as separate but related workflows and emphasizes that LLM-based agents actively interact with external tools or environments [3].

A second distinction is whether tool use is treated as explicit control logic or as learned behavior. ReAct explicitly augments the action space with language and uses reasoning traces to induce and update plans while actions gather external information [2]. Toolformer instead learns to decide which APIs to call, when to call them, what arguments to pass, and how to incorporate the results into later token prediction, using only a handful of demonstrations per API [4]. These papers are complementary: ReAct is easiest to inspect and debug, while Toolformer emphasizes self-supervised acquisition of tool-use behavior [2][4].

A third taxonomic distinction is whether the agent is framed around roles or modules. The paradigm survey introduces three LLM-profiled roles—policy models, evaluators, and dynamic models—to organize workflows, while software-engineering surveys highlight perception, memory, and action as the key modules in code-facing agents [3][12]. This indicates that modern agent design is not just about prompting style; it is also about how decision, feedback, and execution responsibilities are divided [3][12].

## Main agent architectures for tool use
ReAct-style agents are the clearest example of explicit interleaving between thinking and acting. In the cited paper, reasoning helps the model track and update action plans and handle exceptions, while actions gather information from environments or knowledge bases [2]. The evaluation tasks reported there include HotPotQA, FEVER, ALFWorld, and WebShop, showing that the same interaction pattern can support both knowledge-intensive QA and interactive decision-making [2].

Toolformer represents a different architecture: the model is trained to decide when to call tools and how to absorb their outputs into generation [4]. Its tools include a calculator, a Q&A system, search engines, translation, and a calendar, which demonstrates that tool use can be learned across heterogeneous API types rather than being hard-coded for one environment [4]. In contrast to ReAct, Toolformer foregrounds automatic learning of tool invocation rather than explicit reasoning traces [4].

Practical production systems also formalize tool use as structured calling. OpenAI’s function-calling docs state that tools can be described with JSON schemas so the model can emit outputs that are reliably converted to API calls or database queries [5]. Anthropic’s documentation describes tool use as a contract where the application executes the tool while the model emits a structured request and receives the result, and it distinguishes client-executed tools from server-executed tools such as web search and code execution [6]. The same docs also note that programmatic tool calling can reduce latency and token consumption by running multi-tool workflows in a code-execution container [6].

Finally, several surveys emphasize planning-plus-feedback loops. The planning survey organizes methods by decomposition, selection, external modules, reflection, and memory [1]. The task-planning survey on TPTU defines one-step and sequential agents and stresses learning, reflection, and memory from feedback, including correct results and exception errors, to adapt plans over time [best available via notes?]. In practice, this means tool use is rarely a single decision: it is usually embedded in a control loop where the next action depends on observed results [1][6].

## Evaluation and benchmarks
The evaluation literature now treats sequential decision-making as central, not optional. The survey on LLM-agent evaluation says agent assessment must go beyond textual outputs and measure behavior in dynamic environments, and it groups evaluation into core capabilities, application-specific benchmarks, generalist-agent benchmarks, benchmark dimensions, and evaluation frameworks/tools [7]. This aligns with the broader survey trend toward realistic and continuously updated benchmarks [7][14].

Web navigation is one major evaluation family. MIND2WEB contains over 2,000 open-ended tasks from 137 websites spanning 31 domains and reports Element Accuracy, Operation F1, Step Success Rate, and overall Success Rate [8]. Its reported best MindAct result reaches 52.0% step success on Cross-Task, while Cross-Website and Cross-Domain are 38.9% and 39.6% [8]. WebArena-Verified similarly focuses on browser agents, but stresses deterministic checks and comparability conditions; the audited release contains 812 verified tasks across six self-hosted environments plus a 258-task Hard subset [10].

Coding-agent evaluation is increasingly standardized around human-validated tasks. SWE-bench Verified is described as a human-validated subset of 500 SWE-bench instances for reliable evaluation of coding agents and language models [9]. The benchmark’s emphasis on clear problem descriptions, correct test patches, and solvable tasks reflects the field’s concern that evaluation artifacts should not be noisy or ambiguous [9].

Multi-domain agent evaluation is also emerging. AgentBoard evaluates 9 tasks across embodied AI, games, web, and tools, using success rate and progress rate as the main metrics; it also includes grounding accuracy [11]. The board reports that proprietary models such as GPT-4, Claude2, and GPT-3.5-Turbo keep gaining rewards over 30 steps in Alfworld and PDDL, while open-weight models tend to peak early and stop improving after about 6 steps [11]. Together, these benchmarks show that success-rate-only scoring is often insufficient unless one also measures sub-goal progress, grounding, and interaction length [7][8][11].

## Applications and domain patterns
Software engineering is one of the clearest application areas. The software-engineering survey frames agents around perception, memory, and action and notes that they interact with web pages, compilers, and other external tools while using feedback to optimize themselves [12]. This domain tends to stress long-horizon debugging, code execution, and retrieval of external knowledge more than single-turn response quality [12][9].

Web and browser automation form another major application family. MIND2WEB and WebArena-Verified show that browser agents need accurate element grounding, operation selection, and robustness under changing environments [8][10]. The web-agent survey notes that WebAgents automate tedious daily web tasks and organizes the area around architectures, training, and trustworthiness [15]. Compared with code agents, web agents must cope with GUI state, navigation uncertainty, and brittle environment assumptions [10][15].

OS- and assistant-style agents broaden the scope further. The OS Agents survey describes computer, phone, and browser use through GUI and CLI interfaces and notes tasks such as online shopping and travel arrangements [15]. This suggests that “tool use” is not just API invocation but also interaction with operating systems and user-facing workflows [15]. Across these domains, the same high-level loop reappears: observe state, call tools, update memory, and continue [1][6][12][15].

## Trends and open problems
The last two years have seen a shift from isolated prompting tricks to more system-level agent design. Surveys now describe ecosystems spanning construction, collaboration, evolution, evaluation, tools, and real-world issues, rather than treating tool use as a narrow add-on [13][14]. Production-oriented documentation likewise reflects a move toward structured contracts, server-side tools, and programmatic execution to reduce token cost and latency [5][6].

A major open problem is evaluation realism. The evaluation survey says open gaps remain in cost-efficiency, safety, robustness, and scalable fine-grained methods, and it highlights the shift toward more realistic, continuously updated benchmarks [7][14]. The benchmark pages reinforce this point: WebArena-Verified warns that scores are only comparable when the dataset subset, environment revision, scaffold, prompts, interfaces, budgets, evaluator, and attempt policy all match [10].

Another open issue is trustworthiness under long-horizon tool use. Web-agent and OS-agent surveys both emphasize trustworthiness, while the software-engineering survey highlights missing modalities and the lack of an authoritative code knowledge base [12][15]. More broadly, the field still lacks a shared standard for balancing autonomy against control: agents are becoming more capable at chaining tools, but the more autonomy they have, the more important grounding, oversight, and cost control become [6][7][13][15].

## References
[1] Understanding the planning of LLM agents: A survey. arxiv. https://arxiv.org/abs/2402.02716 (2024-02-05)
[2] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[3] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. arxiv. https://arxiv.org/abs/2406.05804 (n.d.)
[4] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[5] Function calling and other API updates. web. https://platform.openai.com/docs/guides/function-calling (2023-06-13)
[6] How tool use works. web. https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works (n.d.)
[7] A Survey on Evaluation of LLM-based Agents. arxiv. https://arxiv.org/abs/2503.16416 (n.d.)
[8] MIND2WEB: Towards a Generalist Agent for the Web. arxiv. https://arxiv.org/abs/2306.06070 (n.d.)
[9] SWE-bench Verified. web. https://www.swebench.com/verified (n.d.)
[10] WebArena-Verified Browser Agent Benchmark (WebArena-Verified). web. https://benchlm.ai/benchmarks/webarena-verified (n.d.)
[11] AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. web. https://hkust-nlp.github.io/agentboard/static/leaderboard.html (n.d.)
[12] Agents in Software Engineering: Survey, Landscape, and Vision. arxiv. https://arxiv.org/abs/2409.09030 (n.d.)
[13] Large Language Model Agent: A Survey on Methodology, Applications and Challenges. arxiv. https://arxiv.org/abs/2503.21460 (n.d.)
[14] Survey on Evaluation of LLM-based Agents. hf-search. https://huggingface.co/papers/2503.16416 (2025-03-20)
[15] OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use. web. https://aclanthology.org/2025.acl-long.369.pdf (n.d.)
