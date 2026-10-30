# Workflow & Conventions

## Planning on GitHub

Work is planned and tracked in the open, the way a team would:

| GitHub feature | Used for |
|----------------|----------|
| [**Milestones**](https://github.com/fkiptooh-r/production-ai-systems/milestones) | One per project (`00 · Foundations` … `05 · Voice agent`); progress = closed issues + merged PRs |
| [**Issues**](https://github.com/fkiptooh-r/production-ai-systems/issues) | One per roadmap milestone or task, with a checklist and a "done when". Titles: `NN · MX · name` |
| **Labels** | `milestone` (roadmap item), `advanced` (beyond the core roadmap), `eval` (measurement), `infra` (CI, Docker, tooling), `blocked` (waiting on something outside the repo), plus `bug` / `documentation` |
| [**Project board**](https://github.com/users/fkiptooh-r/projects/2) | Todo → In Progress → Done across all projects |
| **Pull requests** | Every PR closes its issue (`Closes #N`) and fills in the [template](../.github/pull_request_template.md): what changed, results, what I learned, verification |

## Daily loop: issue → branch → PR

```bash
gh issue list --milestone "01 · Hybrid RAG"            # pick the next issue
gh issue develop 25 --name 01/a1-query-rewriting --checkout   # branch linked to issue #25
# ... work, test ...
git add -p                                             # review each change before staging
git commit -m "feat(01): add RAG-Fusion and HyDE query rewriting"
git push -u origin 01/a1-query-rewriting
gh pr create --fill                                    # body starts with "Closes #25"
gh pr checks --watch                                   # CI must be green
gh pr merge --squash --delete-branch                   # merging closes #25 and moves it to Done
```

Move the issue to **In Progress** on the board when you start. Something unexpected (a bug, an
idea for later) gets its own issue instead of growing the current PR.

Working solo, this still matters: each issue + PR pair is a readable record of what was planned,
what was built, how it was verified and what was learned, and it's the same workflow teams use.

**Stacked PRs:** when one piece of work depends on another unmerged branch, branch from it and
open the PR against it. Say "Stacked on #N" in the description. When #N merges, GitHub deletes
its branch and retargets the next PR to `main`.

## Branch names

`<project>/<milestone>-<short-name>`, e.g. `00/m5-fastapi`, `02/m4-failure-taxonomy`, or `docs/resources-update`

## Commit messages ([Conventional Commits](https://www.conventionalcommits.org/))

```
feat(01): add RRF fusion of BM25 and vector results
fix(02): redact phone numbers before storing traces
test(03): add held-out judge agreement test
docs(04): add cost vs quality results table
chore: bump dependencies
```

Write the *why* in the body when it's not obvious.

## Milestone → tag

When a project milestone ships, tag it so the history is easy to browse:

```bash
git tag 01-m5 -m "Hybrid search: Recall@5 0.62 -> 0.81"
git push --tags
```

When a whole project is done, update its status in the root README table.

## Security checklist (every commit)

- [ ] No API keys, tokens or passwords in code, notebooks or outputs
- [ ] `.env` is not staged (`git status` should never show it)
- [ ] No real user/customer data, only synthetic or public datasets
- [ ] Notebook outputs cleared if they print secrets or personal data

### If a key leaks

1. **Revoke/rotate the key immediately** in the provider dashboard. This is the only real fix.
2. Then remove it from the code. Deleting it in a new commit does **not** remove it from history; bots have already scanned it.
3. Check the provider's usage page for unexpected charges.

## Python project setup (per project)

```bash
cd 01-hybrid-rag
uv init --package .        # creates pyproject.toml
uv add fastapi pydantic    # runtime deps
uv add --dev pytest ruff   # dev deps
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Definition of done (per project)

- [ ] All milestones ticked in the project README
- [ ] Results table filled with real numbers
- [ ] Runs from a clean clone following the README
- [ ] Tests + lint pass in CI
- [ ] 2-minute demo video or GIF linked
- [ ] Short write-up posted (blog / LinkedIn): problem, approach, numbers, what broke
