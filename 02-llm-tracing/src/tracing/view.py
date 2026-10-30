"""Read a JSONL trace file and print traces as trees.

uv run trace-view              # the most recent trace
uv run trace-view --list       # one line per trace: id, root span, duration, errors
uv run trace-view 7f3a         # a trace by id (a unique prefix is enough)
uv run trace-view --stats      # p50/p95 latency per step, tokens and cost per tenant
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tracing.tracer import trace_file


def load(path: Path) -> dict[str, list[dict]]:
    """Spans grouped by trace id, in file order (the order the traces finished)."""
    traces: dict[str, list[dict]] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                traces[s["trace_id"]].append(s)
    return traces


def render(spans: list[dict]) -> list[str]:
    """Tree lines for one trace, children sorted by start time."""
    children: dict[str | None, list[dict]] = defaultdict(list)
    for s in spans:
        children[s["parent_id"]].append(s)
    for kids in children.values():
        kids.sort(key=lambda s: s["start"])

    lines: list[str] = []

    def walk(s: dict, prefix: str, last: bool, root: bool) -> None:
        branch = "" if root else ("└── " if last else "├── ")
        label = f"{prefix}{branch}{s['name']}"
        mark = f"  ✗ {s['error']}" if s["status"] == "error" else ""
        attrs = "  " + " ".join(f"{k}={v}" for k, v in s["attributes"].items())
        lines.append(f"{label:<44}{s['duration_ms']:>10.1f} ms{mark}{attrs.rstrip()}")
        kids = children.get(s["span_id"], [])
        for i, kid in enumerate(kids):
            child_prefix = prefix if root else prefix + ("    " if last else "│   ")
            walk(kid, child_prefix, i == len(kids) - 1, False)

    for root in children.get(None, []):
        walk(root, "", True, True)
    return lines


def summary(trace_id: str, spans: list[dict]) -> str:
    roots = [s for s in spans if s["parent_id"] is None]
    root = roots[0] if roots else spans[-1]
    errors = sum(s["status"] == "error" for s in spans)
    return (
        f"{trace_id[:12]}  {root['start']}  {root['name']:<20} "
        f"{root['duration_ms']:>9.1f} ms  {len(spans)} spans  {errors} errors"
    )


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: p50 is the median, p95 the value 95% of calls stay under."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round(p / 100 * len(ordered)) - 1))]


def stats(traces: dict[str, list[dict]]) -> list[str]:
    """Per step: calls, errors, p50/p95 latency. Then tokens and cost per tenant."""
    durations: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    tenants: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for spans in traces.values():
        tenant = next(
            (s["attributes"]["tenant_id"] for s in spans if "tenant_id" in s["attributes"]), "-"
        )
        for s in spans:
            durations[s["name"]].append(s["duration_ms"])
            errors[s["name"]] += s["status"] == "error"
            for key in ("tokens_in", "tokens_out", "cost_usd", "cost_usd_hosted"):
                tenants[tenant][key] += s["attributes"].get(key, 0)
    lines = [f"{'step':<16}{'calls':>7}{'errors':>8}{'p50 ms':>10}{'p95 ms':>10}"]
    for name, values in sorted(durations.items(), key=lambda kv: -percentile(kv[1], 50)):
        p50, p95 = percentile(values, 50), percentile(values, 95)
        lines.append(f"{name:<16}{len(values):>7}{errors[name]:>8}{p50:>10.1f}{p95:>10.1f}")
    lines += ["", f"{'tenant':<16}{'tokens in':>11}{'tokens out':>12}{'cost':>8}{'if hosted':>12}"]
    for tenant, t in sorted(tenants.items()):
        if t["tokens_in"] or t["tokens_out"]:
            lines.append(
                f"{tenant:<16}{t['tokens_in']:>11.0f}{t['tokens_out']:>12.0f}"
                f"{t['cost_usd']:>8.2f}{t['cost_usd_hosted']:>12.4f}"
            )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show traces from a JSONL trace file.")
    parser.add_argument("trace_id", nargs="?", help="trace id or unique prefix (default: latest)")
    parser.add_argument("--list", action="store_true", help="list all traces")
    parser.add_argument("--stats", action="store_true", help="latency per step, cost per tenant")
    parser.add_argument("--file", type=Path, default=None, help="trace file (default: TRACE_FILE)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252: no tree glyphs

    traces = load(args.file or trace_file())
    if not traces:
        sys.exit("No traces yet. Run `uv run trace-demo` first.")

    if args.list:
        for tid, spans in traces.items():
            print(summary(tid, spans))
        return

    if args.stats:
        print(f"{len(traces)} traces\n")
        print("\n".join(stats(traces)))
        return

    if args.trace_id:
        matches = [t for t in traces if t.startswith(args.trace_id)]
        if len(matches) != 1:
            sys.exit(f"{len(matches)} traces match {args.trace_id!r}; give a longer prefix.")
        tid = matches[0]
    else:
        tid = list(traces)[-1]

    print(f"trace {tid}")
    print("\n".join(render(traces[tid])))
