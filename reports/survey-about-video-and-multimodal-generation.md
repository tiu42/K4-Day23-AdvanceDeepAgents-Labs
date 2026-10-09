# Survey of Video and Multimodal Generation
## TL;DR
- Early multimodal/video generation systems moved from joint video-language representation learning in VideoBERT to discrete-latent autoregressive video generation in VideoGPT, setting up the token-based generation stack used by later systems [1][2].
- Modern video generation is now split across diffusion, autoregressive, and hybrid families; recent surveys emphasize that diffusion dominates high-quality video synthesis, while autoregressive vision models vary mainly by representation strategy from pixels to tokens to scale [3][4][5].
- CogVideo shows the text-to-video scaling trend: a 9B transformer trained on 5.4 million text-video pairs with multi-frame-rate hierarchical training and 480×480 generation [6].
- Evaluation has shifted away from single aggregate scores such as FVD toward multi-dimensional and human-aligned benchmarks like VBench and Video-Bench, because legacy metrics correlate poorly with human judgment and miss video-specific artifacts [7][8][9].
- Recent work highlights application pressure toward long-form, controllable, and safe generation for gaming, embodied AI, autonomous driving, and world-modeling, while open problems remain coherence, physics, causality, safety, and licensing [10][11][12].
## Background
Video and multimodal generation refers to models that synthesize or transform video conditioned on text, images, or other modalities, and to systems that jointly learn video-language or broader multimodal representations. The field matters now because recent models are being used for text-to-video, image-to-video, video editing, and longer-horizon interactive systems, while surveys note growing interest in world-model-style simulation and decision support [3][13][10].

Foundationally, VideoBERT learned bidirectional joint distributions over visual and linguistic token sequences, with visual tokens obtained by vector quantization and text tokens from speech recognition outputs [1]. VideoGPT then showed that raw video can be compressed into discrete latents with VQ-VAE and generated autoregressively with a GPT-like model over spatio-temporal positions, making token-based video generation a practical path forward [2].

Recent surveys place these earlier ideas into a broader evolution from GANs to diffusion and autoregressive methods, and then to multimodal and hybrid systems [14][15].
## Foundations and the rise of tokenized video generation
VideoBERT is important because it treated video and language as a shared token sequence and learned joint distributions over those tokens, using video vector quantization and speech-recognition text as inputs [1]. That design helped establish the general multimodal pretraining pattern of aligning video representations with language, rather than generating pixels directly.

VideoGPT translated that idea into a direct generation model: it learned discrete latent video codes with VQ-VAE and then autoregressively predicted them using a GPT-like transformer [2]. The paper reports competitive samples on BAIR Robot and high-fidelity natural videos on UCF-101 and TGIF, which is notable because it showed that compressed video tokens could support generation quality comparable to leading GANs [2].

CogVideo extended the token/transformer line to large-scale text-to-video pretraining. It is described as a 9B-parameter transformer trained by inheriting CogView2, using multi-frame-rate hierarchical training on 5.4 million text-video pairs and producing 480×480 videos [6].

A recent survey-level synthesis echoes this trajectory: video generation has moved from GANs toward diffusion-based and autoregressive methods, and now includes multimodal approaches [14][15].
## Main model families: diffusion, autoregressive, and hybrid systems
The clearest current taxonomy separates diffusion-based, autoregressive-based, and hybrid systems. A recent unified-model survey states that multimodal understanding has been dominated by autoregressive architectures, while image generation has centered on diffusion, and it explicitly categorizes unified models into diffusion-based, autoregressive-based, and hybrid approaches [5].

Within vision more broadly, autoregressive models differ mainly by representation strategy: pixel-based, token-based, or scale-based. That matters for video and multimodal generation because the choice of representation changes the sequence length, compute profile, and what the model learns to predict [4]. The survey also frames autoregressive modeling as an extension of NLP-style sequential prediction to image, video, 3D, and multimodal generation [4].

Hybrid systems are increasingly common because they try to combine the strengths of both families. The unified-model survey highlights hybrids that fuse autoregressive and diffusion mechanisms [5], and the video-diffusion survey identifies several hybrid families: autoregressive diffusion models, decoupled AR and diffusion stages, shared-parameter AR+diffusion models, and AR backbones with lightweight diffusion samplers [16].

A practical multimodal precursor to these hybrids is the frozen-encoder plus autoregressive-decoder setup. The multitasking decoder paper reports that a small decoder trained on top of a frozen pretrained encoder works surprisingly well, and it describes this as teaching a decoder to interact with a pretrained vision model via natural language [17].

By contrast, the video-diffusion survey says diffusion models have become the leading approach for high-quality and temporally consistent video generation, while GANs struggle with temporal consistency and autoregressive methods have high computational cost [16].
## Evaluation and benchmark design
Evaluation has been a major bottleneck because older metrics do not match human judgment well. VBench states that IS, FID, FVD, and CLIPSIM are inconsistent with human judgment, and that standard VQA methods miss artifacts specific to generated video [7]. The AI-generated-video evaluation survey similarly says the field has evolved from video-opinion pairs to video-instruction-opinion triplets in order to capture instruction following [8].

VBench is important because it decomposes quality into 16 dimensions under two broad categories, Video Quality and Video-Condition Consistency, and uses around 100 prompts per dimension across eight content types [7]. The authors report human preference annotations and say VBench evaluations correlate highly with those preferences [7]. But the benchmark also notes it did not assess safety and equality at publication time [7].

Newer benchmarks try to align more closely with human judgment and broader constraints. Video-Bench uses an automatic multimodal LLM evaluation policy with chain-of-query and few-shot scoring, and its protocol includes 10 human annotators, 4 human scores per generated video, and 35,196 total evaluations [9]. It reports that chain-of-query improves Video-Condition Alignment by 0.093 over single-round evaluation and that few-shot scoring increases correlation with human evaluations by 10.33% on average [9].

Metric research also targets the weaknesses of FVD itself. The JEDi paper says FVD relies on assumptions that do not hold well, proposes a JEPA-feature plus MMD alternative, and reports that JEDi needs only 16% of the samples to reach its steady value while increasing alignment with human evaluation by 34% on average [18]. The same paper reports human-survey agreement figures of 83.70% and 53.54% on UCF-101 and Sky Scene clips, underscoring that benchmark comparisons still depend heavily on task and dataset [18].
## Trends and open problems
The last two years show a shift from short, isolated clip generation toward longer and more interactive systems. The long-video survey says the field is hard because planning, narrative construction, and spatiotemporal continuity are all challenging, and it identifies divide-and-conquer and autoregressive methods as the two main paradigms for long videos [19]. It also calls out the need to preserve semantic consistency, smooth motion transitions, and entity consistency across frames [19].

Application pressure is expanding beyond entertainment. The interactive generative video survey names gaming, embodied AI, and autonomous driving as major application domains, and it says an ideal system needs Generation, Control, Memory, Dynamics, and Intelligence modules [10]. A world-model survey similarly lists autonomous driving, robotics, and social simulacra as key application areas, showing that generation is increasingly discussed as simulation rather than just media synthesis [12].

Safety has become a separate research axis. T2VSafetyBench defines 4 primary categories and 14 critical aspects of video-generation safety and emphasizes trade-offs between usability and safety [11]. It also highlights risks such as illegal or unethical content, synthetic identities, misinformation, copyright violations, privacy issues, and temporal risks that can emerge across frames [11].

Open technical problems remain long-term coherence, real-time controllability, accurate physics, and causal reasoning. The interactive generative video survey names real-time generation, open-domain control, long-term coherence, accurate physics, and causal reasoning as open problems [10]. Separately, the multimodal generation survey points to controllability in open-domain text-to-video generation and editing as a major focus because these systems are central to controllable AIGC and world models [13].

## References
[1] VideoBERT: A Joint Model for Video and Language Representation Learning. arxiv. https://arxiv.org/abs/1904.01766 (2019-09-11)
[2] VideoGPT: Video Generation using VQ-VAE and Transformers. arxiv. https://arxiv.org/abs/2104.10157 (2021-04-20)
[3] Video Diffusion Models: A Survey. arxiv. https://arxiv.org/abs/2405.03150 (2024-11-17)
[4] Autoregressive Models in Vision: A Survey. hf-search. https://huggingface.co/papers/2411.05902 (2024-11-08)
[5] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities. hf-search. https://huggingface.co/papers/2505.02567 (2025-05-05)
[6] CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers. arxiv. https://arxiv.org/abs/2205.15868 (2022-05-29)
[7] VBench: Comprehensive Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2311.17982 (2023-11-29)
[8] A Survey of AI-Generated Video Evaluation. arxiv. https://arxiv.org/abs/2410.19884 (n.d.)
[9] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.pdf (n.d.)
[10] A Survey of Interactive Generative Video. web. http://arxiv.org/abs/2504.21853 (2025-04-30)
[11] T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models. web. https://papers.nips.cc/paper_files/paper/2024/file/74eed5f568354c2e77dd9b018f38a9d4-Paper-Datasets_and_Benchmarks_Track.pdf (n.d.)
[12] Understanding World or Predicting Future? A Comprehensive Survey of World Models | ACM Computing Surveys. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[13] LLMs Meet Multimodal Generation and Editing: A Survey. web. https://arxiv.org/html/2405.19334 (n.d.)
[14] Video generation: a survey. web. https://dl.acm.org/doi/10.1145/3678239 (n.d.)
[15] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (n.d.)
[16] Survey of Video Diffusion Models:Foundations, Implementations, and Applications. arxiv. https://arxiv.org/abs/2504.16081 (2026-02-10)
[17] A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision. hf-search. https://huggingface.co/papers/2303.17376 (2023-03-30)
[18] Beyond FVD: Enhanced Evaluation Metrics for Video Generation Quality. arxiv. https://arxiv.org/abs/2410.05203 (2024-10-08)
[19] Video Is Worth a Thousand Images: Exploring the Latest Trends in Long Video Generation. web. https://arxiv.org/html/2412.18688 (n.d.)
