# Learning Resources

A curated list per project. **Free unless marked 💲.** Links can move over time; if one is dead, search for its title.

How to use this list: don't read everything up front. Read the ⭐ item for a milestone, build, and come back when you get stuck.

---

## Big picture (read alongside everything)

- ⭐ **Chip Huyen, *AI Engineering* (O'Reilly, 2025)** 💲: the closest thing to a textbook for this whole repo
- ⭐ [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/): practical lessons from practitioners
- [Eugene Yan, Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): when to use workflows vs. agents
- [Andrej Karpathy, Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): how LLMs work under the hood (optional, but it builds intuition)
- [3Blue1Brown, Neural Networks series](https://www.3blue1brown.com/topics/neural-networks): visual intuition for transformers and attention
- [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/): 1–2 hour focused courses on RAG, evals, agents, etc.
- [Hugging Face Learn](https://huggingface.co/learn): free courses on NLP, LLMs, agents, audio

---

## 00 · Foundations

**Python & tooling**
- ⭐ [The Python Tutorial](https://docs.python.org/3/tutorial/)
- [uv docs](https://docs.astral.sh/uv/): package/env manager used in this repo
- [Ruff](https://docs.astral.sh/ruff/): linter + formatter
- [pytest: Get Started](https://docs.pytest.org/en/stable/getting-started.html)
- [Pydantic docs](https://docs.pydantic.dev/): validation and structured output
- [The Missing Semester of Your CS Education (MIT)](https://missing.csail.mit.edu/): shell, git, editors, debugging

**LLM APIs**
- ⭐ [Anthropic docs](https://docs.anthropic.com/): messages, streaming, tool use, prompt caching
- [Anthropic courses (GitHub)](https://github.com/anthropics/courses): API fundamentals, prompt engineering, tool use
- [Anthropic interactive prompt engineering tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)
- [OpenAI Cookbook](https://cookbook.openai.com/): many patterns apply to any provider
- [Ollama](https://ollama.com/): run open models locally for free

**APIs, databases, deployment**
- ⭐ [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [PostgreSQL tutorial](https://www.postgresql.org/docs/current/tutorial.html)
- [Docker: Get started](https://docs.docker.com/get-started/)
- [GitHub Actions quickstart](https://docs.github.com/en/actions/writing-workflows/quickstart)

**Git & GitHub**
- ⭐ [Pro Git book](https://git-scm.com/book/en/v2): chapters 1–3 are enough to start
- [GitHub Skills](https://skills.github.com/): interactive GitHub courses
- [First Contributions](https://github.com/firstcontributions/first-contributions): practise your first open-source PR
- [goodfirstissue.dev](https://goodfirstissue.dev/): beginner-friendly issues in real projects

---

## 01 · Hybrid RAG

- ⭐ [Anthropic, Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval): hybrid search + reranking explained with numbers
- [pgvector](https://github.com/pgvector/pgvector): vector search inside Postgres
- [Qdrant docs](https://qdrant.tech/documentation/): good explanations of hybrid search even if you use pgvector
- [Pinecone Learn](https://www.pinecone.io/learn/): free articles on chunking, embeddings, reranking
- [Sentence-Transformers docs](https://www.sbert.net/): local embeddings and cross-encoder rerankers
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard): compare embedding models
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): BM25 in a few lines of Python
- [Postgres full-text search](https://www.postgresql.org/docs/current/textsearch.html)
- Reciprocal Rank Fusion: Cormack, Clarke & Büttcher (2009), *"Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods"* (search the title; it's a 2-page paper)
- [Postgres Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html): for tenant isolation (M8)
- [Ragas docs](https://docs.ragas.io/): RAG evaluation metrics (use after you've written your own)

---

## 02 · LLM Tracing

- ⭐ [OpenTelemetry: Observability primer](https://opentelemetry.io/docs/concepts/observability-primer/): traces, spans, attributes
- ⭐ [Langfuse docs](https://langfuse.com/docs): open-source LLM tracing, self-hostable
- [Arize Phoenix](https://github.com/Arize-ai/phoenix): open-source alternative with good eval tooling
- [OpenLLMetry](https://github.com/traceloop/openllmetry): OpenTelemetry instrumentation for LLM libraries
- [OpenTelemetry semantic conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/): standard attribute names for LLM spans
- [Microsoft Presidio](https://github.com/microsoft/presidio): PII detection and redaction (M7)

---

## 03 · LLM-as-a-Judge & Evals

- ⭐ [Hamel Husain, Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- ⭐ [Hamel Husain, Creating a LLM-as-a-Judge That Drives Business Results](https://hamel.dev/blog/posts/llm-judge/): the "label by hand first, critique, then align the judge" method this project follows
- [Eugene Yan, Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- [Zheng et al. 2023, Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685): the paper on judge biases
- [Anthropic docs: Define success criteria and build evaluations](https://docs.anthropic.com/en/docs/test-and-evaluate/define-success)
- [promptfoo](https://github.com/promptfoo/promptfoo): eval runner that works well in CI (M7)

---

## 04 · Model Routing & Cost

- ⭐ [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM): open-source learned routers + paper
- [FrugalGPT paper (Chen et al. 2023)](https://arxiv.org/abs/2305.05176): cascades and cost reduction
- [LiteLLM docs](https://docs.litellm.ai/): gateway, fallbacks, budgets, spend tracking
- [OpenRouter](https://openrouter.ai/): one API for many models; handy for cheap comparisons
- [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [GPTCache](https://github.com/zilliztech/GPTCache): semantic caching ideas (study it, then build a small version)
- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression): enough for a learned router

---

## 05 · Voice Agents

- ⭐ [LiveKit Agents docs](https://docs.livekit.io/agents/): real-time voice agents over WebRTC
- ⭐ [Pipecat](https://github.com/pipecat-ai/pipecat): open-source voice agent framework (good examples folder)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): fast local speech-to-text
- [OpenAI Whisper](https://github.com/openai/whisper): the original STT model
- [Silero VAD](https://github.com/snakers4/silero-vad): voice activity detection
- [Deepgram docs](https://developers.deepgram.com/): streaming STT/TTS API with free credits
- [Piper](https://github.com/rhasspy/piper): fast local text-to-speech
- [WebRTC for the Curious](https://webrtcforthecurious.com/): how real-time audio over the web actually works

---

## Communities (where to ask when stuck)

- Discord servers of the tools you use (LiveKit, Pipecat, Langfuse, Hugging Face, Ollama): maintainers often answer
- [Hugging Face Forums](https://discuss.huggingface.co/)
- GitHub Issues/Discussions of the library: search closed issues first
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/): local models, hardware, cheap inference
