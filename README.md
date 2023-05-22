# fromLLMtoAGI
## In Context Learning
- [What In-Context Learning "Learns" In-Context: DisentanglingTask Recognition and Task Learning](https://arxiv.org/abs/2305.09731), Task recognition (TR) captures the extent to which LLMs can recognize a task through demonstrations -- even without ground-truth labels -- and apply their pre-trained priors, whereas task learning (TL) is the ability to capture new input-label mappings unseen in pre-training.
### a recall, query and inference?
- [How does in-context learning work? A framework for understanding the differences from traditional supervised learning](https://ai.stanford.edu/blog/understanding-incontext/)
### or a true learning algorithm?
- [What learning algorithm is in-context learning? Investigations with linear models](https://arxiv.org/abs/2211.15661), We investigate the hypothesis that transformer-based in-context learners implement standard learning algorithms implicitly, by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context
- [Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/abs/2212.10559), Transformer attention has a dual form of gradient descent. 
## Prompt, Agent, System
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information.
- [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366), an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities.
- [Progressive-Hint Prompting Improves Reasoning in Large Language Models](https://arxiv.org/abs/2304.09797), This paper proposes a new prompting method, named Progressive-Hint Prompting (PHP), that enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers.
- [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495), textual gradient descent
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291)
- [Introspective Tips: Large Language Model for In-Context Decision Making](https://arxiv.org/abs/2305.11598)
## Language Model
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), despite of the small size of the models, we still observe an emergence of reasoning capabilities, knowledge of general facts and ability to follow certain instructions.
