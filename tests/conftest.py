"""Reset mutable agent class-level config between tests so they don't bleed into each other."""
import pytest
from src.agent import agent


@pytest.fixture(autouse=True)
def reset_agent_class_config():
    saved = {
        "ping_interval_s": agent.ping_interval_s,
        "poll_interval_s": agent.poll_interval_s,
        "max_outer_iters": agent.max_outer_iters,
        "log_dir":         agent.log_dir,
        "output_dir":      agent.output_dir,
    }
    yield
    agent.ping_interval_s = saved["ping_interval_s"]
    agent.poll_interval_s = saved["poll_interval_s"]
    agent.max_outer_iters = saved["max_outer_iters"]
    agent.log_dir         = saved["log_dir"]
    agent.output_dir      = saved["output_dir"]
