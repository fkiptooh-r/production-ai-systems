# Production AI Systems

[![CI](https://github.com/fkiptooh-r/production-ai-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/fkiptooh-r/production-ai-systems/actions/workflows/ci.yml)

Five production-grade AI engineering projects, one core skill each, built in public while learning to ship real LLM systems: **retrieval, observability, evaluation, cost control, and real-time voice.**

> 🚧 **Status:** learning in public. Each project folder has its own roadmap with milestones; progress is tracked in the table below.

## Why this repo

Calling an LLM API is easy. Making the result **reliable, measurable, affordable and fast** is the real job. Each project here targets one of those problems, and they are designed to plug together into one system rather than being five disconnected demos.

## Projects

| # | Project | Core skill | Status |
|---|---------|------------|--------|
| 00 | [Foundations](00-foundations/) | Python, LLM APIs, FastAPI, Docker, deployment | 🟨 In progress |
| 01 | [Hybrid RAG search](01-hybrid-rag/) | Hybrid retrieval, reranking, citations, tenant isolation | ⬜ Not started |
| 02 | [Failure forensics](02-llm-tracing/) | LLM tracing and observability | ⬜ Not started |
| 03 | [Output arbitration](03-llm-judge/) | LLM-as-a-judge, eval pipelines | ⬜ Not started |
| 04 | [Cost autopilot](04-model-routing/) | Model routing, caching, budgets | ⬜ Not started |
| 05 | [Real-time voice agent](05-voice-agent/) | Streaming STT → LLM → TTS, turn-taking | ⬜ Not started |

Legend: ⬜ Not started · 🟨 In progress · ✅ Done (with eval results in the project README)

## How they fit together

```
  user (text / voice)
        │
        ▼
  05 voice agent ──► 04 router ──► cheap model / strong model
                         │
                         └──► 01 hybrid RAG (tenant-aware retrieval)

  every step traced by 02   ·   every output scored by 03
```

## Learning order

`00 → 01 → 02 → 03 → 04 → 05`

Tracing (02) and evals (03) come **before** routing (04) on purpose: you can't claim "60% cheaper" without measuring that quality held.

## Repo layout

```
production-ai-systems/
├── 00-foundations/     # Python, APIs, Docker, deploy
├── 01-hybrid-rag/
├── 02-llm-tracing/
├── 03-llm-judge/
├── 04-model-routing/
├── 05-voice-agent/
├── shared/             # reusable code: LLM client, config, tracing setup
└── docs/
    ├── RESOURCES.md    # curated learning resources per project
    └── WORKFLOW.md     # git workflow, conventions, security checklist
```

## Principles

1. **Concept before framework.** Build the basic version by hand once, then adopt the library.
2. **Every claim is backed by numbers.** Each project ships with an eval set and a results table.
3. **Multi-tenant from day one.** A `tenant_id` travels through retrieval, tracing, costs and budgets.
4. **No secrets, no real user data in git.** Synthetic or public datasets only.

## Running a project

Each project is self-contained with its own README, dependencies and instructions. Secrets live in a local `.env` file (copy [`.env.example`](.env.example)), which is never committed.

## Resources

See [docs/RESOURCES.md](docs/RESOURCES.md) for the curated reading/watching list behind each project.

## License

[MIT](LICENSE)
