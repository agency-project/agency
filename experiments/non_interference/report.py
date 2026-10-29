"""Write the final research report directly from recorded observations."""

from pathlib import Path
from collections import Counter
import csv
import json
import sys


def read(path):
    return json.loads(path.read_text())


def main(root):
    aggregate = read(root / "analysis_results.json")["aggregate"]
    tasks = read(root / "analysis_results.json")["task_summary"]
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    config = read(root / "configuration.json")
    environment = read(root / "ec2_environment.json")
    freeze = read(root / "freeze_metadata.json")
    planning = read(root / "sample_size_planning.json")
    outcomes = Counter(r["status"] for r in rows)
    resolved = Counter(r["resolved"] for r in rows)
    missing = sum(r["usage_complete"] != "True" for r in rows)
    records = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    measured_window = (
        min(r["start_unix_s"] for r in records),
        max(r["end_unix_s"] for r in records),
    )
    samples = [
        json.loads(line)
        for line in (root / "resource_samples.jsonl").read_text().splitlines()
        if measured_window[0] <= json.loads(line)["unix_s"] <= measured_window[1]
    ]
    calls = [c for r in records for c in r.get("provider", {}).get("calls", [])]
    events = [
        json.loads(line) for line in (root / "scheduling_events.jsonl").read_text().splitlines()
    ]
    measured_starts = [
        event
        for event in events
        if event.get("event") == "start" and event.get("diagnostic") is False
    ]
    throttle = sum(c.get("status") == 429 for c in calls)
    transport = sum(bool(c.get("error")) for c in calls)
    http_errors = Counter(str(c.get("status")) for c in calls if c.get("status", 0) >= 400)
    cost_ci = aggregate["difference_95ci"]
    variance_ci = aggregate["variance_ratio_95ci"]
    lines = [
        "# Agency non-interference pilot — SWE-bench Verified",
        "",
        "Completed execution is distinct from solving the benchmark issue. All measured attempts, failures, and timeouts are retained.",
        "",
        "## Main result",
        "",
        "The primary endpoint is **standardized cache-neutral reference cost in USD**, calculated from observed provider token usage and the frozen account-specific LiteLLM pricing schedule. It is not an audited invoice.",
        "",
        f"Equal-task mean Agency − standalone cost: **${aggregate['difference_usd']:.6f}** (95% CI ${cost_ci[0]:.6f} to ${cost_ci[1]:.6f}). Equal-task relative difference: **{aggregate['equal_task_relative_difference_pct']:+.2f}%** (95% CI {aggregate['relative_difference_pct_95ci'][0]:+.2f}% to {aggregate['relative_difference_pct_95ci'][1]:+.2f}%).",
        f"Geometric mean task-level variance ratio: **{aggregate['geometric_variance_ratio']:.3f}** (95% CI {variance_ci[0]:.3f} to {variance_ci[1]:.3f}).",
        "",
        f"Recorded unique measured executions: {len(rows)}/100. Execution outcomes: {dict(outcomes)}. Official resolved outcomes: {dict(resolved)}. Executions with incomplete usage: {missing}.",
        f"Raw worker attempt records retained: {len(records)}; attempt statuses: {dict(Counter(r['status'] for r in records))}. The normalized table uses the latest allowed attempt for each logical execution.",
        "",
    ]
    if missing:
        missing_ids = [r["run_id"] for r in rows if r["usage_complete"] != "True"]
        lines += [
            f"**Measurement limitation:** the numerical comparisons above use only {len(rows) - missing} executions with complete observed usage. Incomplete run(s): {', '.join('`' + x + '`' for x in missing_ids)}. One or more forwarded requests lack authoritative usage; full costs for those runs remain unknown. The normalizer checks raw per-request records rather than trusting the recorder's optimistic completeness flag. Missing usage was not imputed. This is an exploratory complete-case comparison, not an unbiased estimate for all 100 executions; provider failures can cause informative missingness. The planned full-sample cost endpoint is unavailable.",
            "",
        ]
    if cost_ci[0] <= 0 <= cost_ci[1]:
        lines += [
            "The pilot does not clearly separate the mean cost effect from zero for these five tasks. A confidence interval containing zero does not establish equivalence.",
            "",
        ]
    else:
        lines += [
            "The pilot suggests a mean cost difference for these five fixed tasks under this configuration. It does not establish a population-wide framework effect.",
            "",
        ]
    if variance_ci[0] <= 1 <= variance_ci[1]:
        lines += [
            "The variance interval includes one. Ten repetitions per arm yield noisy variance estimates and do not establish equal variability.",
            "",
        ]
    else:
        lines += [
            "The observed variance interval excludes one for this fixed task set. This is preliminary, and variance estimates at ten repetitions remain sensitive to outliers and provider failures.",
            "",
        ]
    lines += [
        "## Per-task reference cost",
        "",
        "| Frozen task | n S/A | Mean standalone $ | Mean Agency $ | Difference $ | Relative % | SD S/A $ | CV S/A | Variance ratio [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in tasks:
        lines.append(
            f"| {t['instance_id']} | {t['n_standalone']}/{t['n_agency']} | {t['mean_standalone_usd']:.6f} | {t['mean_agency_usd']:.6f} | {t['difference_usd']:+.6f} | {t['relative_difference_pct']:+.2f} | {t['sd_standalone_usd']:.6f}/{t['sd_agency_usd']:.6f} | {t['cv_standalone']:.3f}/{t['cv_agency']:.3f} | {t['variance_ratio']:.3f} [{t['variance_ratio_ci_low']:.3f}, {t['variance_ratio_ci_high']:.3f}] |"
        )
    lines += [
        "",
        "Task order above is the approved order. No task was replaced or selected using outcomes.",
        "",
        "## Execution and benchmark outcomes",
        "",
        "Every cell contains ten measured executions. A completed execution means the harness finished; resolution is determined independently by the official SWE-bench evaluator.",
        "",
        "| Task | Completed S/A | Timed out S/A | Resolved S/A | Unresolved S/A | Grading unknown S/A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        cells = [
            [r for r in rows if r["instance_id"] == task["instance_id"] and r["arm"] == arm]
            for arm in ("standalone", "agency")
        ]
        paired = lambda predicate: "/".join(str(sum(predicate(r) for r in cell)) for cell in cells)
        lines.append(
            f"| {task['instance_id']} | {paired(lambda r: r['status'] == 'completed')} | {paired(lambda r: r['timeout'] == 'True')} | {paired(lambda r: r['resolved'] == 'True')} | {paired(lambda r: r['resolved'] == 'False')} | {paired(lambda r: r['resolved'] not in ('True', 'False'))} |"
        )
    lines += [
        "",
        "| Arm | Mean reference cost $ | Mean estimated cache-aware cost $ | Mean tool calls | Mean LLM requests | Mean wall time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("standalone", "agency"):
        selected = [r for r in rows if r["arm"] == arm]

        def observed_mean(key):
            task_means = []
            for task in tasks:
                values = [
                    float(r[key])
                    for r in selected
                    if r["instance_id"] == task["instance_id"] and r[key]
                ]
                if values:
                    task_means.append(sum(values) / len(values))
            return sum(task_means) / len(task_means) if task_means else float("nan")

        lines.append(
            f"| {arm} | {observed_mean('reference_cost_usd'):.6f} | {observed_mean('estimated_cached_cost_usd'):.6f} | {observed_mean('tool_calls'):.2f} | {observed_mean('llm_requests'):.2f} | {observed_mean('wall_s'):.2f} |"
        )
    lines += [
        "",
        "These means give each task equal weight. Cost means use complete usage records; call counts and wall times retain all 100 executions.",
        "",
        "## Reproducibility pins",
        "",
        f"- Freeze commit: `{freeze['freeze_commit']}`.",
        f"- Canonical manifest SHA-256: `{freeze['manifest_sha256']}`.",
        "- Dataset: `princeton-nlp/SWE-bench_Verified`, test, revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`; raw Parquet and dataset metadata are preserved under `evidence/`.",
        f"- Agency source: `{environment['agency_commit']}`; package version `{environment['agency_version']}`.",
        f"- Claude Code: `{environment['claude_code_version']}`; binary SHA-256 `{environment['claude_binary_sha256']}`.",
        "- Official evaluator: exact Python sources preserved in `evidence/evaluator_source/`, with file hashes in `grader_metadata.json`. The existing source copy exposes no Git revision; file hashes pin the actual code used.",
        f"- Model: `{config['model']}`, upstream `{environment['router_upstream_model']}`, through the same existing LiteLLM endpoint for both arms. The provider exposes no independently verifiable immutable weight snapshot.",
        f"- Execution host: `{environment.get('execution_host', environment.get('ec2'))}`; {environment['cpu_count']} vCPUs. OS, kernel, memory, Docker, packages, and image IDs are in `ec2_environment.json` and `configuration.json`.",
        f"- Scheduling seed `{config['schedule_seed']}`. Bootstrap seed `{config['bootstrap_seed']}`; {config['bootstrap_resamples']:,} resamples.",
        "",
        "## Controls, concurrency and caches",
        "",
        "Each execution uses a fresh immutable official task image, repository at its approved base commit, independent container/HOME/temp/session/logs, and 2-vCPU/4-GiB limits. Required compiled dependencies in the fresh image are retained. Both arms receive the same rendered benchmark instructions, system settings, permissions, model, output cap, effort, native compaction setting, and disabled subagents. Fresh session identifiers and session-directory paths differ in this setup. The paths are model-visible and can affect token counts and cache prefixes. Initial-request audits normalize these fields to check other settings; the cost analysis retains all actual tokens, including path-related tokens.",
        "",
        "Agency uses the baseline annotation arm, no tool/syscall policy callbacks, no context interventions, and disabled schema/tool offloading. Its study launcher matches the standalone native toolset by using an empty external MCP configuration. Native request bodies/headers and response control envelopes are audited in `diagnostic_gate.json`; no hidden provider retry layer is added. Claude Code owns native provider retries in both arms. Agency's tracing, lifecycle management, hooks and completion checkpoint remain framework overhead, with unavoidable differences documented rather than erased.",
        "",
        f"Maximum sampled study containers: {max(len(s['containers']) for s in samples)}. Maximum sampled total execution CPU quota: {max(s['cpu_quota_total'] for s in samples):g} vCPUs. Minimum sampled host available memory: {min(s['mem_available_kb'] for s in samples) / 1024**2:.2f} GiB. `scheduling_events.jsonl` records exact starts, finishes, active arm counts and any concurrency reduction. Ten-second samples cannot exclude unsampled host contention.",
        "",
        f"The exact start journal records at most {max(e['active_total'] for e in measured_starts)} active executions, {max(e['active_standalone'] for e in measured_starts)} standalone and {max(e['active_agency'] for e in measured_starts)} Agency. Maximum recorded active API admissions: {max(c.get('admission', {}).get('active_requests_at_admission', 0) for c in calls)}. Dedicated queues maintain the per-arm cap through the tail; the standalone queue finished {(max(r['end_unix_s'] for r in records if r['arm'] == 'agency') - max(r['end_unix_s'] for r in records if r['arm'] == 'standalone')) / 60:.1f} minutes before Agency, leaving an Agency-only tail. Consequently provider time and cache exposure are not perfectly matched even with randomized balanced scheduling.",
        "",
        f"Global admission limits: {config['admission']['requests_per_minute']} requests/minute, {config['admission']['tokens_per_minute']:,} conservatively reserved tokens/minute, at most four active API streams. Native auxiliary requests share the same limiter. Recorded HTTP 429s: {throttle}; recorder exceptions (including the local pre-admission error described below): {transport}; HTTP errors: {dict(http_errors)}. All attempt records include errors and limiter waits.",
        "",
        "Native provider cache directives are preserved in both arms; provider caches can be shared across parallel runs. The recorder requests HTTP Cache-Control:no-cache from the router, but administrative confirmation of response-cache policy is unavailable. Cache-hit response headers are preserved where supplied. Cache-neutral cost charges each uncached/read/written input token once at the same pinned input rate, with the >100k prompt tier applied per request. Cache-read/write tokens and estimated cache-aware costs are reported separately. Router response-cost headers may be sparse for streaming calls. Exact billing availability must be established separately through the account ledger. Do not interpret missing router costs as zero.",
        "",
        "`tool_calls` counts provider-generated tool_use blocks, including calls in failed executions; it does not assert that every proposed call finished successfully. `llm_requests` includes auxiliary native requests and retries. Execution/controller exit status is distinct from official benchmark resolution.",
        "",
        "| Arm | Observed cache-read tokens | Observed cache-write tokens | Cache-read share of observed input | Mean admission wait per execution (s) | HTTP 429 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("standalone", "agency"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        total = lambda key: sum(float(r[key]) for r in arm_rows if r[key])
        inputs = total("total_input_including_cache")
        fraction = total("cache_read_input_tokens") / inputs if inputs else 0
        lines.append(
            f"| {arm} | {total('cache_read_input_tokens'):,.0f} | {total('cache_creation_input_tokens'):,.0f} | {fraction:.1%} | {total('admission_wait_s') / len(arm_rows):.2f} | {total('http_429'):.0f} |"
        )
    lines += [
        "",
        "These are descriptive observed totals; incomplete streams can omit additional tokens. The cache table is not a billing statement.",
        "",
        "A local admission failure before opening the upstream HTTP request consumes no provider tokens. Such events are retained as `native_api_attempts` and `local_pre_inference_errors`; `llm_requests` counts actual upstream attempts. Server/transport errors after forwarding without authoritative usage remain unknown rather than zero.",
        "",
    ]
    amendments = root / "runtime_amendments.json"
    if amendments.exists():
        lines += ["### Recorded infrastructure repair", ""]
        for change in read(amendments):
            lines += [
                f"After {change['completed_executions_at_deployment']} completed executions, the common recorder's SQLite admission locking was repaired following a verified pre-provider database-lock event. Short transactions are serialized, connections are closed, and transient database locks are waited out internally. Active workers retained their loaded version; subsequent workers in both conditions used the repaired recorder. Source hashes and deployment time are preserved in `runtime_amendments.json`. No completed execution was rerun and no model, prompt, resource, rate, sample-size or task setting changed.",
                "",
            ]
        local_events = sum(int(r.get("local_pre_inference_errors") or 0) for r in rows)
        lines += [
            f"Observed local pre-inference events: {local_events}. All affected measured runs remain in the primary dataset. This shared infrastructure incident and mid-study reliability repair are limitations when interpreting small pilot effects. The launch and repaired recorder sources are preserved for audit. Offline accounting corrections and their failing/passing fixture evidence are documented in `analysis_amendments.json`.",
            "",
        ]
        versions = Counter(
            (
                r["arm"],
                "repaired"
                if any(
                    c.get("recorder_source_sha256") for c in r.get("provider", {}).get("calls", [])
                )
                else "launch",
            )
            for r in records
        )
        lines += [
            f"Runs using launch/repaired recorder: standalone {versions['standalone', 'launch']}/{versions['standalone', 'repaired']}; Agency {versions['agency', 'launch']}/{versions['agency', 'repaired']}. This imbalance reflects different execution durations and is another reason not to interpret the pilot as a definitive causal non-interference result.",
            "",
        ]
    audit_path = root / "fidelity_audit.json"
    if audit_path.exists():
        audit = read(audit_path)
        cache = audit.get("response_cache_evidence", {})
        lines += [
            "### Full-run protocol audit",
            "",
            f"Request-preservation mismatch runs: {len(audit['request_mismatch_runs'])}; response-control mismatch runs: {len(audit['response_mismatch_runs'])}; starting-state/resource mismatch runs: {len(audit['resource_or_starting_state_mismatch_runs'])}. Detailed outcomes and scope are in `fidelity_audit.json`; failed/cancelled streams can have incomplete response envelopes. No result is dropped based on this audit.",
            f"Observed streamed response IDs: {cache.get('response_messages')}; unique IDs: {cache.get('unique_message_ids')}; repeated IDs: {len(cache.get('repeated_message_ids', {}))}; router cache-header records: {len(cache.get('router_cache_headers', []))}. These are empirical cache checks and do not replace administrative confirmation of router settings.",
            "",
        ]
    lines += [
        "## Statistical method and limitations",
        "",
        "The bootstrap independently resamples observed runs within each fixed task and arm, preserving the repeated-run structure and equal task weights. It never treats 100 executions as 100 benchmark tasks. Mean intervals are percentile bootstrap intervals. Variance intervals resample within task/arm; degenerate variance resamples are counted in the analysis JSON. Aggregation averages task log-variance ratios before exponentiation. These intervals are conditional on the five frozen tasks and do not justify generalization to all SWE-bench tasks. Stochastic harness behavior, shared-host load, rate-admission waiting, shared caches and provider faults may induce temporal dependence; timestamps and resource traces support follow-up audits. No trajectory identity or statistical equivalence is claimed.",
        "",
        "## Larger-study planning",
        "",
        planning["method"] + ".",
        "",
        planning["assumptions"],
        "",
        "| Equivalence margin | Repetitions per arm/task | Approx. tasks required | Additional tasks | Extra repetitions per existing task/arm |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in planning["grid"]:
        lines.append(
            f"| ±{r['margin_pct']:g}% | {r['repetitions_per_arm']} | {r['required_tasks']} | {r['additional_tasks']} | {r['additional_repetitions_per_existing_task_arm']} |"
        )
    lines += [
        "",
        "For inference limited to the same five tasks, a separate normal approximation removes between-task sampling variance:",
        "",
        "| Equivalence margin | Repetitions per existing task/arm | Additional repetitions per task/arm |",
        "|---|---:|---:|",
    ]
    for r in planning["fixed_task_only"]:
        lines.append(
            f"| ±{r['margin_pct']:g}% | {r['repetitions_per_arm_task']} | {r['additional_repetitions_per_arm_task']} |"
        )
    lines += [
        "",
        "The TOST interpretation follows [Lakens (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/); the task/run variance decomposition and power approximation are specific to this pilot's repeated-run design.",
    ]
    lines += [
        "",
        "The companion JSON also evaluates power at the observed pilot mean effect. Requirements assume zero true systematic effect for equivalence planning; more data cannot establish a too-narrow margin when the true effect lies outside it. Variance components estimated from only five tasks are highly uncertain. Treat these counts as planning approximations, not guarantees.",
        "",
        "## Reproduce",
        "",
        "In the prepared remote study directory, run `python run_experiment.py measured` to resume; completed attempts are skipped. An unfinished attempt without a final record requires an infrastructure audit rather than an automatic rerun. Only one retry is permitted for verified zero-inference setup failures. The manifest is never rewritten.",
        "",
        "After all executions, run `python grade.py batch`. Install `requirements-analysis.txt` in an isolated environment. Run `python analysis.py /path/to/non_interference`, `python audit_results.py /path/to/non_interference`, `python report.py /path/to/non_interference`, then `python analysis_secondary.py /path/to/non_interference`. The raw provider records, attempt summaries, official grader reports and frozen prices are the analysis inputs. Diagnostic attempts in `diagnostics/` are excluded. The five figure families are exported as PDF, SVG and 300-dpi PNG under `figures/`.",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent)
