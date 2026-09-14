# Host inventory and historical stress evidence

`inspect_host.py` collects host topology; `placement.py` derives and validates
CPU placement from that inventory. These utilities do not execute agent work.

```sh
python3 benchmarks/agent_stress/inspect_host.py > inventory.json
python3 benchmarks/agent_stress/placement.py --inventory inventory.json --out cpu_layout.json
python -m pytest -q benchmarks/agent_stress/test_placement.py
```

`results/` contains historical measurements and environment preparation records.
Their commands, API names, and validation outcomes describe those recorded
checkouts, not the current release. The former execution runner depended on
removed invocation state and `agent.destroy()` APIs and has been removed.

For current lifecycle acceptance coverage, use `tests/test_golden_execution.py`
on Linux with the documented container prerequisites. The numbered tutorials
and `examples/run_all.py` cover the current public API.
