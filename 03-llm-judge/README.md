# 03 · Output Arbitration (LLM-as-a-Judge)

**Goal:** automatically grade LLM outputs in a way you can **trust** (because it agrees with your own labels), then use it to pick the best answer among candidates and to gate changes in CI.
**Time:** 2–3 weeks.

## What you'll learn
- Why human labels come first, and how to write useful critiques
- Writing rubrics with binary pass/fail criteria (more reliable than 1–10 scores)
- Measuring judge quality: agreement, precision/recall on failures
- Judge biases: position, length, self-preference
- Pairwise comparison vs. direct scoring
- Running evals in CI

## Milestones

- [ ] **M1 · Label by hand first.** Take 100 outputs from project 01. Mark each pass/fail and write *one sentence why*. This is your ground truth. Don't skip it.
- [ ] **M2 · Rubric.** Turn your "why" sentences into 3–5 binary criteria, e.g. *grounded in context? answers the actual question? cites a source? correct refusal?*
- [ ] **M3 · Judge v1.** A prompt with the rubric + structured output: `{criterion: pass/fail, reason}`. Run it on the 100 labelled examples.
- [ ] **M4 · Validate.** Measure agreement with your labels and precision/recall on the *fail* class. Iterate the prompt on 70 examples, then report results on the 30 held-out examples.
- [ ] **M5 · Bias checks.** Swap A/B order in pairwise mode; test whether longer answers win unfairly; try a judge from a different model family than the generator.
- [ ] **M6 · Arbitration system.** Generate N candidates (different prompts, models or temperatures), judge them, return the winner. Log close calls to a human review queue.
- [ ] **M7 · CI gate.** A GitHub Action runs the eval suite when prompts change and fails if the pass rate drops more than X%.

## Results to report

| Metric | Value |
|--------|-------|
| Judge agreement with my labels (held-out) | |
| Precision on "fail" | |
| Recall on "fail" | |
| Position bias (A/B swap flip rate) | |
| Cost per 100 judgements | |

## Done when
- You can state "my judge agrees with me X% of the time on held-out data"
- The CI gate works on a deliberate bad prompt change (show it failing in a PR)

## Ethics note
For grading anything that affects people (student marks, listing removals), keep a human in the loop and let people appeal. Use the judge to triage and assist, not as the final say.

## Where this goes in your products
- **E-commerce:** check generated product descriptions for accuracy and tone before publishing
- **House hunting:** flag suspicious or low-quality listings for review
- **EdTech:** rubric-based feedback on short answers and essays, with teacher review

## Resources
See [docs/RESOURCES.md → LLM-as-a-Judge](../docs/RESOURCES.md#03--llm-as-a-judge--evals).
