"""Execution counters maintained from authoritative, identity-keyed projections."""

import math


class ExecutionCounters:
    def __init__(self):
        self.calls = {}
        # The same counters keyed by recorded actor, so a client holding only a
        # bounded window of calls can still scope execution totals to agents.
        self.by_agent = {}
        self.totals = dict(
            tools=0,
            models=0,
            completed_tools=0,
            tool_errors=0,
            input=0,
            output=0,
            input_samples=0,
            output_samples=0,
        )

    def update(self, actions):
        for action in actions:
            model = action["kind"] == "model"
            # Queue/wait/lifecycle intervals are not tool invocations.
            if action["kind"] in ("queue", "dependency", "unknown", "setup"):
                continue
            values = {key: 0 for key in self.totals}
            values["models" if model else "tools"] = 1
            values["completed_tools"] = int(
                not model and action["outcome"] in ("success", "failed", "cancelled", "timeout")
            )
            values["tool_errors"] = int(not model and action["outcome"] == "failed")
            if model:
                for source, target in [("tokens", "input"), ("output_tokens", "output")]:
                    value = action.get(source)
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        values[target] = value
                        values[target + "_samples"] = 1
            previous, owner = self.calls.get(action["id"], ({}, None))
            agent = action.get("agent")
            for key, value in values.items():
                self.totals[key] += value - previous.get(key, 0)
                # A call keeps its identity if its recorded actor is corrected.
                if owner is not None:
                    self.by_agent[owner][key] -= previous.get(key, 0)
                if agent is not None:
                    self.by_agent.setdefault(agent, dict.fromkeys(self.totals, 0))[key] += value
            self.calls[action["id"]] = (values, agent)
        return dict(self.totals)
