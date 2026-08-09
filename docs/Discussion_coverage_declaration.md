# Discussion: what a profiler artifact is allowed to claim

**Status:** for lab discussion. Blocks M10 in
[Design_profiler_harness_integration.md](Design_profiler_harness_integration.md).
Does not block M0–M9.

**Ask:** a decision on four questions in §4. The engineering is small; the
semantics are not ours to pick unilaterally, because the artifact is what an
external hardware comparison gets read against.

---

## 1. The problem in one example

`agency` runs agents through five interchangeable engines: `native`,
`claude_code`, `codex`, `opencode`, `grok`. The profiler emits one
`summary.json` per run, with sections for runs, sandbox ops, LLM calls, turns,
tools, and resource samples.

The profiler cannot see all five engines equally. `native` runs our own ReAct
loop, so turn and tool boundaries are emitted precisely from inside the
container. The other four are third-party CLIs; their loop is opaque to us. We
recover turns and tools for them by diffing the conversation the model provider
sees on consecutive dispatches — exact for turn counts and tool *names*,
inferred for tool *durations*.

So a `summary.json` from a `codex` run and one from a `native` run have the
same keys. One of them has a `tools` section built from direct measurement. The
other has one built from inference — and before M3 lands, from nothing at all.

An empty `tools` section is indistinguishable from an agent that made no tool
calls. A reader comparing two machines sees a number, not a provenance. The
failure mode is not a crash or a wrong-looking chart — it is a plausible
number, quoted confidently, that measures our instrumentation rather than the
hardware.

## 2. Why this is live now

The Samsung engagement asks for hardware comparisons, with one explicit
methodological constraint:

> "Mocked or recorded model end points must be used so that hardware
> comparisons are not confounded by network/model variance."

That constraint is about removing a *known* confound. The coverage question is
the same class of problem one level down: it is about disclosing the confound
we cannot remove. We will not have Tier-3 instrumentation for `codex`,
`opencode`, and `grok` by the time numbers are wanted, and there is no
schedule in which we do. The question is whether the artifact says so.

Once a number is in a partner-facing document, the disclosure cannot be
retrofitted onto the reader's memory.

## 3. What is proposed (§5.1 of the design doc)

`summary.json` gains a `coverage` block declaring, per section and per engine,
one of:

| State | Meaning |
|---|---|
| `complete` | Directly instrumented. The number is a measurement. |
| `derived` | Reconstructed host-side from provider-dispatch diffs. Turn counts and tool identities are exact; tool durations are inferred from the gap between dispatches — exact when a turn has one tool call, an **upper bound** when it has several in parallel. |
| `unavailable(reason)` | Not measurable for this engine, with the reason stated inline. |

The governing rule: **an absent section must never be readable as zero.**

Expected state matrix at the end of the current roadmap:

| Section | native | claude_code | codex / opencode / grok |
|---|---|---|---|
| runs, sandbox, GPU leases, resource samples | complete | complete | complete |
| LLM calls (latency, TTFT, tokens) | complete | complete | complete |
| turns | complete | complete | derived (count exact) |
| tools | complete | complete (M7) | derived |
| per-span CPU / runqueue time | unavailable (container-originated spans have no host thread) | unavailable | unavailable |
| process spawn/exit | complete (M6) | complete (M6) | complete (M6) |

The rows that are `complete` everywhere are not an accident: every engine's
model traffic passes through one credentialed host-side choke point, and
resource sampling is host-observed. **The headline hardware numbers —
wall-clock, CPU, memory, GPU utilisation, model latency — are Tier 1 and are
equally trustworthy across all five engines.** The asymmetry is confined to
the semantic breakdown: which tool ran, and for how long.

That is worth stating plainly, because it means the coverage block is not an
admission that the benchmark is weak. It is what lets us say the strong part is
strong.

## 4. The four questions

### Q1 — Is a three-state vocabulary the right granularity?

Options:

- **(a) Three states, per section, as proposed.** Simplest. Loses the fact that
  within a `derived` tools section, turn counts are exact while durations are
  not.
- **(b) Add a fourth state** (`partial`, `upper_bound`) to separate "inferred
  but bounded" from "inferred, unbounded".
- **(c) Per-field provenance** rather than per-section — every number carries
  its own state. Most honest, most verbose, and it changes the artifact's shape
  for every consumer.

*Our lean: (a), with Q2's error bound attached as data rather than as a fourth
state name.* Consumers can branch on three states; they can read a bound as a
number.

### Q2 — Should `derived` carry a quantified error bound?

We know, per run, how many turns had exactly one tool call (duration exact) and
how many had several in parallel (duration is an upper bound, and the
attribution across them is a heuristic). Emitting that ratio costs a counter.

Without it, `derived` is a warning label. With it, `derived` is a measurement
with stated uncertainty — which is the difference between a number a reader
must discard and one they can use with a caveat.

*Our lean: yes.* It is nearly free and it materially changes what the artifact
supports.

### Q3 — What is the publication rule?

The decision that actually matters. Options:

- **(a) Nothing `derived` leaves the team.** Safest, and it discards usable
  data — the tool *identities* and turn counts are exact even when durations
  are not.
- **(b) `derived` may be published with its state and bound shown adjacent to
  the number.** Requires the report format to carry the annotation, not just
  the JSON.
- **(c) `derived` may be published for within-engine comparisons (machine A vs
  machine B, same engine), but a cross-engine comparison requires both engines
  to be in the same state for the section being compared.**

*Our lean: (c).* The confound is asymmetry between engines, not inference
itself. Comparing `codex`-on-machine-A to `codex`-on-machine-B has the same
inference error on both sides and it largely cancels; comparing `codex` to
`native` does not.

### Q4 — Who signs off that a comparison is admissible?

The coverage block makes admissibility *checkable*. It does not make it
*checked*. Someone has to own the step between "the artifact says `derived`"
and "this number goes in the deck" — either a named reviewer, or a mechanical
gate in the report generator that refuses to render a comparison violating Q3.

*Our lean: mechanical gate, with a documented override.* A rule enforced only
by attention fails on the week someone is busy.

## 5. Cost and sequencing

Small: a static per-engine table, a serializer, and a regeneration of M8's
golden fixture. Essentially all of the cost is in this discussion, which is
why it is broken out rather than buried in a milestone.

It is numbered M10 because it depends on M3 (which determines what `derived`
can actually deliver) and M8 (whose fixture it invalidates). **It is late in
the numbering, not late in priority: it must land before any profiler-derived
number leaves the team.**

## 6. Interim position

Until M10 lands, profiler output is internal and provisional. Any number pulled
from a `summary.json` before then carries no disclosure of what was measured
versus inferred, and should not be quoted externally.

---

## Appendix — an adjacent confound, same conversation

Samsung's requirement covers *model* endpoints. If the benchmark suite includes
SWE-bench, each task's `git clone` and `pip install` reintroduce network
variance into wall-clock — the same confound the requirement was written to
eliminate, entering through a door the sentence does not cover. The fix is
pre-baked task images. Worth raising in the same discussion, since it affects
whether the resulting numbers mean what the requirement intends them to mean.
