"""Lesson 1: one agent, one typed skill, and one pending result."""

from __future__ import annotations

from agency import AgError, Agent, agcanceled, agdata, agent, agerror, agschema, agskill

from _common import close_sandboxes, example_wait_timeout, run_example, tutorial_config


def main() -> None:
    assert Agent is agent
    # agdata is the common payload, schema, and pending-result shape.
    payload = agdata(topic="sandboxed agents", audience="Python developers")
    assert agdata.from_json(payload.to_json()) == payload

    schema = agschema(agdata(topic=str, audience=str))
    assert schema.validate_input(payload) is None
    assert schema.check(agdata(topic=3, audience="Python developers"))

    failure = agerror("illustrative failure")
    cancellation = agcanceled()
    assert failure.error == "illustrative failure"
    assert cancellation.error == "agent invocation cancelled"
    try:
        _ = failure.answer
    except AgError:
        pass

    cfg, run_dir = tutorial_config("01_basic_agent")
    summarize = agskill(
        name="summarize",
        prompt="Summarize the topic for the requested audience in one sentence.",
        input_schema=agdata(topic=str, audience=str),
        output_schema=agdata(summary=str),
    )
    learner = Agent("learner", agconfig=cfg)

    result = learner.run(summarize, payload)
    print(f"submitted: pending={result.is_pending()}")
    result.wait(timeout=example_wait_timeout())
    print(f"summary: {result.summary}")
    print(f"result JSON: {result.to_json()}")
    print(f"history messages: {len(learner.history.messages)}")
    print(f"artifacts: {run_dir}")

    close_sandboxes([learner])


if __name__ == "__main__":
    run_example(main)
