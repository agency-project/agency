"""Benchmark-only lifecycle: collect failed episodes before destroying them."""

from agency import agSandbox


class EpisodeSandbox(agSandbox):
    def rm_container(self):
        # AgentEngine normally rolls failed skills back. A benchmark instead
        # needs the attempted patch and trace, including when the budget expires.
        # Explicit destroy() still uses the backend's ordinary final cleanup.
        with self._lock:
            try:
                self.commit()
            finally:
                self.stop()
