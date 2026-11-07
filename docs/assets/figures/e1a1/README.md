# E1A.1 trajectory variation and runtime figure

The updated figure compares contention-related trajectory variation with normal baseline-to-baseline variation. It uses only the saved E1A.1 campaign. The observed similarities cluster around the `y = x` line while all 15 contended runs take longer. This is a **descriptive** result: the five-task sample does not prove that CPU contention has no trajectory effect.

## Saved inputs

- `../../../../artifacts/e1a1/trajectory-catalog/trajectory-index.csv` maps all 30 runs to task, condition, repetition, and event stream. A matched pair shares task and repetition.
- `../../../../artifacts/e1a1/trajectory-catalog/trajectory-events/slot-*.jsonl` contains the ordered event-level `tool_call` records. The script verifies each stream's count against the index.
- `../../../../artifacts/e1a1/imported/campaign-e1bf2bbb354147149fb7015de83eb217/integrity-report.json` supplies terminal worker wall times and official resolved outcomes. Its reported median paired wall-time ratio is **1.7470206889777513×**.
- `../../../../artifacts/e1a1/imported/campaign-e1bf2bbb354147149fb7015de83eb217/campaign-all-epochs.tar.gz` supplies each saved `worker/final.patch` for final-patch file-set and byte-identity checks.

## Constructing Panel A

A **trajectory** is the ordered sequence of saved tool calls after each call is mapped to a broad category: `edit`, `test`, `review`, `search`, `read`, `probe`, or `other`. Each shell call receives one category, even when it contains multiple operations. The script's fixed classification order is explicit in `action_label`. Timing, exact command text, LLM text, and tool results are excluded from similarity.

**Similarity** is the length of the longest common subsequence of those category sequences divided by the length of the longer sequence. It preserves action order and yields 1.0 for identical category sequences.

For each task and repetition `i`:

- **x** = median of `similarity(B_i, B_j)` over the other two baseline repetitions `j` for that same task. With two comparisons, this is their arithmetic mean. It estimates normal baseline run-to-run similarity anchored to the exact baseline run used on the y-axis.
- **y** = `similarity(B_i, C_i)`, where `C_i` is the CPU-contended run with the same task and repetition.

Thus every one of the 15 matched pairs has one point, with no selected baseline comparator. The dashed line is `y = x`. A point below it is less similar under contention than that baseline run was to its two baseline peers; a point above it is more similar. `pair_metrics.csv` records all x-, y-, and difference values with the source attempt IDs.

## Comparison and limitations

The median `y − x` is **−0.003289**; the mean is **+0.001411**. Eight points are below the diagonal and seven above it. As an exploratory check that respects the five task clusters, the script averages the three differences within each task, then enumerates all `5^5` bootstrap resamples of those five task means. The 95% percentile interval for the mean difference is **[−0.041812, +0.047877]**. These values show no directional shift in this coarse action metric. Because there are only five tasks and no prespecified equivalence margin, the interval is **not proof of equivalence** and cannot rule out smaller or task-specific effects.

Panel B uses the terminal integrity report's worker wall times: **15/15** matched contended runs are slower, and the median paired ratio is **1.747×**. All **30/30** runs resolved. The sets of files in the final patches match for **15/15** pairs, but **0/15** final patches are byte-identical. Matching file sets and outcomes do not imply identical patches or action paths.

To regenerate the PNG, SVG, and per-pair CSV from saved data:

```sh
python -m pip install matplotlib
python docs/assets/figures/e1a1/make_trajectory_figure.py
```

The script reads the archived records and does not invoke agents, models, graders, containers, or experiment runners.
