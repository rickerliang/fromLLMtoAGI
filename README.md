# fromLLMtoAGI
## Leaderboard
- [Chatbot Arena Leaderboard](https://lmsys.org/blog/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
## Transformer, In-context Learning
- [What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning](https://arxiv.org/abs/2305.09731), Task recognition (TR) captures the extent to which LLMs can recognize a task through demonstrations -- even without ground-truth labels -- and apply their pre-trained priors, whereas task learning (TL) is the ability to capture new input-label mappings unseen in pre-training.
- [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/abs/2305.18654), We propose two hypotheses. First, Transformers solve compositional tasks by reducing multi-step compositional reasoning into linearized path matching. Second, due to error propagation, Transformers may have inherent limitations on solving high-complexity compositional tasks that exhibit novel patterns. 🚀
### a recall, query and inference?🚀
- [How does in-context learning work? A framework for understanding the differences from traditional supervised learning](https://ai.stanford.edu/blog/understanding-incontext/)
### or a true learning algorithm?🚀
- [What learning algorithm is in-context learning? Investigations with linear models](https://arxiv.org/abs/2211.15661), transformer-based in-context learners implement standard learning algorithms implicitly, by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context
- [Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/abs/2212.10559), Transformer attention has a dual form of gradient descent. 
## Prompt, Agent, Methodology
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), use LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761), a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.
- [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366), an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities.
- [Progressive-Hint Prompting Improves Reasoning in Large Language Models](https://arxiv.org/abs/2304.09797), a new prompting method, named Progressive-Hint Prompting (PHP), that enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers.
- [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495), textual gradient descent
- [LETI: Learning to Generate from Textual Interactions](https://arxiv.org/abs/2305.10314), LMs' potential to learn from textual interactions (LeTI) that not only check their correctness with binary labels, but also pinpoint and explain errors in their outputs through textual feedback. 
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738), introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools.
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)🚀
- [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291)
- [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323), Given a question, Planner composes a comprehensive blueprint of interlinked plans prior to tool response. The blueprint instructs Worker to use external tools and collect evidence. Finally, plans and evidence are paired and fed to Solver for the answer.
- [Introspective Tips: Large Language Model for In-Context Decision Making](https://arxiv.org/abs/2305.11598)
- [Grammar Prompting for Domain-Specific Language Generation with Large Language Models](https://arxiv.org/abs/2305.19234)🚀
- [Deliberate then Generate: Enhanced Prompting Framework for Text Generation](https://arxiv.org/abs/2305.19835), the already [DOSOMETHING] is [INCORRECT CONTENT], Please detect the error type firstly, and provide the refined informal sentence then.
## AGI, Application
- [Ghost in the Minecraft: Generally Capable Agents for Open-World Enviroments via Large Language Models with Text-based Knowledge and Memory](https://github.com/OpenGVLab/GITM)🚀
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)🚀
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126), a dispatcher, a tool maker and a tool user
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334), GPT-4 SFT LlaMa, massive apis
## Language Model
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), despite of the small size of the models, we still observe an emergence of reasoning capabilities, knowledge of general facts and ability to follow certain instructions.
- [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. 
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), kpi vs okr
## Implementation
- [LangChain](https://github.com/hwchase17/langchain), in-context learning, prompt template, chain of thought, toolformer, ReAct, ToT
- [LangFlow](https://github.com/logspace-ai/langflow)
- [MOSS](https://github.com/OpenLMLab/MOSS), An open-source tool-augmented conversational language model from Fudan University
- [LlaMA](https://github.com/facebookresearch/llama)
- [lit-LlaMA](https://github.com/Lightning-AI/lit-llama)
## Prompt Engineer
- [Learn Prompt](https://github.com/LearnPrompt/LearnPrompt)
- [Mr.-Ranedeer-AI-Tutor](https://github.com/JushBJJ/Mr.-Ranedeer-AI-Tutor)
