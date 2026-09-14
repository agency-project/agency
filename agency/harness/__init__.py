"""Implementation details behind agency's `agharness` engine seam --
`agproxy_ptrace` (the syscall-level supervisor) and `agharness_backends/`
(one concrete off-the-shelf harness CLI per file). The per-agent
host/harness-side managers each backend actually dispatches through live
in `agency/manager/` (`agmanager_host`/`agmanager_harness`), not here.
Nothing in here is part of the public API; `agharness.py` (this package's
own thin, engine-agnostic glue module) and `agent`/`agskill`'s `harness=`
seam are the intended entry points.
"""
