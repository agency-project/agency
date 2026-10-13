"""Break down an oolong_pilot.py results.jsonl by arm and question type.

    python benchmarks/context_tools/analyze.py <out>/results.jsonl [--whole-doc ID,ID,...]

Runs that errored (an API failure, say) are reported and left out of every mean.
Differences between arms come with a bootstrap 95% interval over runs.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st

METRICS = ("relative_error", "score", "cost_usd", "compactions", "peak_prompt_tokens")


def edits(record: dict) -> int:
    return (record.get("tool_calls") or {}).get("edit_context", 0)


def line(rows: "list[dict]") -> str:
    if not rows:
        return "n= 0"

    def mean(key):
        return st.mean(r.get(key) or 0 for r in rows)

    return (
        f"n={len(rows):2d}  rel.err {mean('relative_error'):.4f}  score {mean('score'):.3f}  "
        f"cost ${mean('cost_usd'):.3f}  peak {mean('peak_prompt_tokens'):>7,.0f}  "
        f"compactions {mean('compactions'):.2f}  edits {st.mean(edits(r) for r in rows):.1f}  "
        f"runs that edited {sum(edits(r) > 0 for r in rows)}"
    )


def bootstrap(
    a: "list[dict]", b: "list[dict]", key: str, n: int = 4000
) -> "tuple[float, float, float]":
    rng = random.Random(0)
    diffs = sorted(
        st.mean(rng.choice(b)[key] for _ in b) - st.mean(rng.choice(a)[key] for _ in a)
        for _ in range(n)
    )
    point = st.mean(r[key] for r in b) - st.mean(r[key] for r in a)
    return point, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument(
        "--whole-doc",
        default="117010206,117010207,117010231,117010232",
        help="ids of the questions that count labels over the whole document",
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.results)]
    whole = {int(i) for i in args.whole_doc.split(",") if i}
    arms = sorted({r["arm"] for r in records})
    failed = [r for r in records if "error" in r]
    ok = [r for r in records if "error" not in r]

    print(f"{len(records)} runs, total cost ${sum(r.get('cost_usd') or 0 for r in records):.2f}")
    for r in failed:
        print(f"  left out: {r['arm']} q{r['id']} r{r.get('repeat', 0)}: {r['error'][:100]}")

    groups = {
        f"whole document ({len(whole)} questions)": lambda r: r["id"] in whole,
        "subsets": lambda r: r["id"] not in whole,
        "all": lambda r: True,
    }
    for name, keep in groups.items():
        print(f"\n== {name}")
        for arm in arms:
            print(f"  {arm}  {line([r for r in ok if r['arm'] == arm and keep(r)])}")

    print("\n== whole-document questions, per question (mean over repeats)")
    for q in sorted(whole):
        parts = []
        for arm in arms:
            rows = [r for r in ok if r["id"] == q and r["arm"] == arm]
            if rows:
                parts.append(
                    f"{arm}: err {st.mean(r['relative_error'] for r in rows):.3f} "
                    f"${st.mean(r['cost_usd'] for r in rows):.2f} "
                    f"c{st.mean(r['compactions'] for r in rows):.1f} "
                    f"e{st.mean(edits(r) for r in rows):.1f} (n={len(rows)})"
                )
        print(f"  q{q}  " + "  |  ".join(parts))

    print("\n== differences on whole-document questions (bootstrap 95% CI)")
    by_arm = {arm: [r for r in ok if r["arm"] == arm and r["id"] in whole] for arm in arms}
    pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1 :]]
    for key in ("relative_error", "cost_usd", "compactions", "peak_prompt_tokens"):
        for x, y in pairs:
            point, lo, hi = bootstrap(by_arm[x], by_arm[y], key)
            print(f"  {y} - {x}  {key:<18} {point:+10.4f}   [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
