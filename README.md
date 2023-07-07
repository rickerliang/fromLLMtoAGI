# fromLLMtoAGI
## Leaderboard
- [Chatbot Arena Leaderboard](https://lmsys.org/blog/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chain-of-Thought Hub: Measuring LLMs' Reasoning Performance](https://github.com/FranxYao/chain-of-thought-hub), Existing results strongly suggest that if RLHF is done right on LLaMA, it may be close to ChatGPT-3.5
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
## Dataset Investigate
- [Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data](https://arxiv.org/abs/2306.13840)💡
  - We compute the Task2Vec diversity coefficient as the expected cosine distance d between pairs of Task2Vec embeddings of batches $\hat{d}iv(D)=E_{B_1,B_2}d(\overrightarrow{f}_B1, \overrightarrow{f}_B2)$, where $\overrightarrow{f}_Bi$ is the Task2Vec embedding of a batch $B_i$ using the diagonal of the FIM matrix $\hat{B}_i$
  - compute the FIM as follows: $\hat{F}_B = E\nabla{w}log\hat{p}_w()\nabla{w}log\hat{p}_w()^T$
  - $\overrightarrow{f}_B=Diag(F_B)$
## Transformer, In-context Learning
- [What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning](https://arxiv.org/abs/2305.09731), Task recognition (TR) captures the extent to which LLMs can recognize a task through demonstrations -- even without ground-truth labels -- and apply their pre-trained priors, whereas task learning (TL) is the ability to capture new input-label mappings unseen in pre-training.
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654), We propose two hypotheses. First, Transformers solve compositional tasks by reducing multi-step compositional reasoning into linearized path matching. Second, due to error propagation, Transformers may have inherent limitations on solving high-complexity compositional tasks that exhibit novel patterns. 🚀
- a recall, query and inference?🚀
  - [How does in-context learning work? A framework for understanding the differences from traditional supervised learning](https://ai.stanford.edu/blog/understanding-incontext/)
- or a true learning algorithm?🚀
  - [What learning algorithm is in-context learning? Investigations with linear models](https://arxiv.org/abs/2211.15661), transformer-based in-context learners implement standard learning algorithms implicitly, by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context
  - [Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/abs/2212.10559), Transformer attention has a dual form of gradient descent.
- [Understanding In-Context Learning via Supportive Pretraining Data](https://arxiv.org/abs/2306.15091)💡💡, we use the similarity between gradients $\nabla_\theta L_\theta^{PT}(w)$ and $\nabla_\theta L_\theta^{ICL}(x,y)$ iteratively to find supportive pretraining data. We then compare the supportive subset contrastively with random subsets of pretraining data and discover: (1) The supportive pretraining data to ICL do not have a higher domain relevance to downstream tasks. (2) The supportive pretraining data have a higher mass of rarely occurring, long-tail tokens. (3) The supportive pretraining data are challenging examples where the information gain from long-range context is below average, indicating learning to incorporate difficult long-range context encourages ICL.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)💡, We find that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts.
## Prompt, Agent, Methodology
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), use LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information.
- [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910), Automatic Prompt Engineer (APE), 1)Use LLM to sample instruction proposals, 2)evaluate score on the subset of dataset, 3)filter the top k of instructions with high scores, 4)update instruction, 5)->2).
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761), a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.
- [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366), an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities.
- [Progressive-Hint Prompting Improves Reasoning in Large Language Models](https://arxiv.org/abs/2304.09797), a new prompting method, named Progressive-Hint Prompting (PHP), that enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers.
- [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495), textual gradient descent
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128), SELF-DEBUGGING with Simple Feedback, with Unit Tests and via Code Explanation.
- [Natural Language to Code Translation with Execution](https://arxiv.org/abs/2204.11454), we introduce execution result--based minimum Bayes risk decoding (MBR-EXEC) for program selection, Bayes risk of a
program is defined by the sum of the loss between itself and other examples.
- [LETI: Learning to Generate from Textual Interactions](https://arxiv.org/abs/2305.10314), LMs' potential to learn from textual interactions (LeTI) that not only check their correctness with binary labels, but also pinpoint and explain errors in their outputs through textual feedback. 
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738), introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools.
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)🚀
- [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291)
- [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323), Given a question, Planner composes a comprehensive blueprint of interlinked plans prior to tool response. The blueprint instructs Worker to use external tools and collect evidence. Finally, plans and evidence are paired and fed to Solver for the answer.
- [Introspective Tips: Large Language Model for In-Context Decision Making](https://arxiv.org/abs/2305.11598)
- [Grammar Prompting for Domain-Specific Language Generation with Large Language Models](https://arxiv.org/abs/2305.19234)🚀
- [Deliberate then Generate: Enhanced Prompting Framework for Text Generation](https://arxiv.org/abs/2305.19835), the already [DOSOMETHING] is [INCORRECT CONTENT], Please detect the error type firstly, and provide the refined informal sentence then.
- [Deductive Verification of Chain-of-Thought Reasoning](https://arxiv.org/abs/2306.03872), natural program format allows individual reasoning steps (an example in purple) and their corresponding minimal
set of premises (an example in orange) to be easily extracted, natural Program-based deductive reasoning verification approach, we identify and eliminate reasoning chains that contain errors in reasoning and grounding.🚀
- [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528), Given an LLM $f_{\theta}$ , a dataset D, and a clean prompt P, the objective of a prompt attack can be formulated as follows: $argmax_{\delta \in C}E_{(x;y) \in D}\mathcal{L}[f_{\theta}([P+\delta,x],y)]$, attack level: character, word, sentence, semantic.🚀
- [Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting](https://arxiv.org/abs/2305.07004), XLT is a generic template prompt that stimulates cross-lingual and logical reasoning skills to enhance task performance across languages.
- [Demystifying GPT Self-Repair for Code Generation](https://arxiv.org/abs//2306.09896),
- [Language models are weak learners](https://arxiv.org/abs/2306.14101)💡, summaries of a collection of examples(hypothesis, prompt) + LLM = weak model(s), hierarchical agglomerative clustering, AdaBoost.
- [Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/abs/2306.14892)🚀🚀🚀
- [LARGE LANGUAGE MODELS ARE EFFECTIVE TEXT RANKERS WITH PAIRWISE RANKING PROMPTING](https://arxiv.org/abs/2306.17563)
## AGI, Application
- [Ghost in the Minecraft: Generally Capable Agents for Open-World Enviroments via Large Language Models with Text-based Knowledge and Memory](https://github.com/OpenGVLab/GITM)🚀
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)🚀
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126), a dispatcher, a tool maker and a tool user
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334), GPT-4 SFT LlaMa, massive apis
- [OlaGPT: Empowering LLMs With Human-like Problem-Solving Abilities](https://arxiv.org/abs/2305.16334), The framework involves approximating different cognitive modules, including attention, memory, reasoning, learning, and corresponding scheduling and decision-making mechanisms.
- [Responsible Task Automation: Empowering Large Language Models as Responsible Task Automators](https://arxiv.org/abs/2306.01242), we enhance LLMs with three core capabilities, i.e., feasibility prediction, completeness verification and security protection.
- [Natural Language Commanding via Program Synthesis](https://arxiv.org/abs/2306.03460), Office Domain Specific Language, an analysis-retrieval prompt engineering framework.🚀
- [SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models](https://arxiv.org/abs/2305.19308), a set of virtual APIs, a state machine-based planner——Observing, Proposing, Revising and Acting.
- [Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow](https://arxiv.org/abs/2306.07209), explore data by self-request; interface definition, merging, implementation; intent analysis, planning workflow, multi-form output
- [WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences](https://arxiv.org/abs/2306.07906), 1)coarse-grained web search and fine-grained LLM-augmented dense retrieval(fine-tuned contrievers); 2)bootstrapped generator, WebGLM-QA dataset(LLM icl, bootstrapped, correction and filtering) finetune GLM; 3)a human preference-aware scorer, build a human preference-aware scorer based on massive user feedback(e.g., thumb-ups) from online QA forums(High quality feedback, Length-bias mitigation, Contrast augmentation), SFT, Comparison training, 6B GLM.
- [From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought](https://arxiv.org/abs/2306.12672)💡
## Language Model
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), despite of the small size of the models, we still observe an emergence of reasoning capabilities, knowledge of general facts and ability to follow certain instructions.
- [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. 
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), kpi vs okr
- [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561), employ N different LLMs to get output candidates then pair all candidates and concatenate them with the input before feeding them to PAIRRANKER, rank all candidates and take the top K of them for generative fusion.
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644), textbook quality training data
- [Bring Your Own Data! Self-Supervised Evaluation for Large Language Models](https://arxiv.org/abs/2306.13651), Knowledge Probing via Negations: is->is not; Toxicity: F-bombing; Context (Long Range) Sensitivity: replacing the first two sentences with two random sentences from the corpus; Word Order: two random words are swapped in each sentence; Tokenization Sensitivity: randomly chop strings of raw input text.
- [AN EFFICIENT SPARSE INFERENCE SOFTWARE ACCELERATOR FOR TRANSFORMER-BASED LANGUAGE MODELS ON CPUS](https://arxiv.org/abs/2306.16601)
- [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2306.17194)💡, content injection: prepend “Answer the following question and include [a key phrase] in your answer:”; over-refusal: “Tell me why you cannot answer the following question:”
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)🚀, an LLM uses prompts to evolve instructions,
with two types: in-depth evolving and in-breadth evolving, use the same LLM for evolving to generate the corresponding responses for the evolved instructions.
- [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)
## Training Technique
- [Training Transformers with 4-bit Integers](https://arxiv.org/abs/2306.11987)
- [GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models](https://arxiv.org/abs/2306.13649)💡🚀
  - $L_{GKD}(\theta):=$
    - $(1-\lambda)E_{(x,y)\sim(X,Y)}[\mathcal{D}(p_T\parallel p^{\theta}_S(y|x)]+$
    - $\lambda E_{x \sim X} \big[E_{y\sim ps(\cdot|x)}[\mathcal{D}(p_T\parallel p^{\theta}_S(y|x)]\big]$
  - when approximating $P(\mathcal{C})$ using a parameterized distribution $Q_\theta(\mathcal{C})$, minimizing the reverse and forward KL under model under-specification results in mean and mode-seeking behavior.
    - $D_{KL}(P\parallel Q)=\displaystyle \sum_{c \in C}P(c)\log \frac{P(c)}{Q(c)}$
    - $D_{RKL}(P\parallel Q):=D_{KL}(Q\parallel P)$
    - $D_{JSD[\beta]}(P\parallel Q)=\beta D_{KL}(P\parallel \beta P + (1-\beta)Q)+(1-\beta)D_{KL}(Q\parallel \beta P + (1-\beta)Q)$.
  - Supervised FT; Supervised KD; On-policy KD
- [Improving Language Plasticity via Pretraining with Active Forgetting](https://arxiv.org/abs/2307.01163), we introduce a simple active forgetting mechanism, that resets the token embeddings at regular intervals, while leaving all other parameters untouched throughout pretraining.
## Implementation, Code Repo

### Flow
- [LangChain](https://github.com/hwchase17/langchain), in-context learning, prompt template, chain of thought, toolformer, ReAct, ToT
- [LangFlow](https://github.com/logspace-ai/langflow)
- [Flowise](https://github.com/FlowiseAI/Flowise)
- [Chat UI](https://github.com/huggingface/chat-ui/), A chat interface using open source models, eg OpenAssistant.
### Inference of LLaMA model in pure C/C++
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
### Model Weights
- [MOSS](https://github.com/OpenLMLab/MOSS), An open-source tool-augmented conversational language model from Fudan University
- [LlaMA](https://github.com/shawwn/llama-dl)
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [lit-LlaMA](https://github.com/Lightning-AI/lit-llama)
- [OpenLlaMA](https://github.com/openlm-research/open_llama)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm), MLC LLM is a universal solution that allows any language models to be deployed natively on a diverse set of hardware backends and native applications, plus a productive framework for everyone to further optimize model performance for their own use cases.
- [GPT4ALL](https://github.com/nomic-ai/gpt4all), Open-source assistant-style large language models that run locally on your CPU.
- [Falcon](https://huggingface.co/tiiuae/falcon-40b)
- [Robin](https://huggingface.co/OptimalScale/robin-65b-v2-delta)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
- [中文法律大模型](https://github.com/PKU-YuanGroup/ChatLaw)
- [CodeGen](https://github.com/salesforce/CodeGen)
#### LLMs-In-China
- [中国大模型列表](https://github.com/wgwang/LLMs-In-China)
## Prompt Engineer, Guide
- [Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Learn Prompt](https://github.com/LearnPrompt/LearnPrompt)
- [Mr.-Ranedeer-AI-Tutor](https://github.com/JushBJJ/Mr.-Ranedeer-AI-Tutor)
- [GPT best practices](https://platform.openai.com/docs/guides/gpt-best-practices)
## Datasets
- [🤗🤗🤗](https://huggingface.co/datasets)
