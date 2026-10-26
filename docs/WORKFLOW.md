# Workflow & Conventions

## Daily git loop

```bash
git checkout main && git pull            # start from the latest main
git checkout -b 01/m3-vector-search      # one branch per milestone
# ... work, test ...
git add -p                               # review each change before staging
git commit -m "feat(01): add pgvector search and Recall@5 eval"
git push -u origin 01/m3-vector-search
gh pr create --fill                      # open a PR (even working solo)
gh pr merge --squash --delete-branch     # merge when checks pass
```

Working solo, PRs still matter: each one is a readable record of what you built and why, and it's the same workflow teams use.

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
