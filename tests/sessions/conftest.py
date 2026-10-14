import pytest

from agency.configs.agconfig import agconfig, dataloggerconfig
from agency.observability.agdatalogger import agDataLogger
from agency.sessions.claude_code import ClaudeCodeSession

from .helpers import FIXTURES


@pytest.fixture
def make_session(tmp_path):
    """Start a session whose CLI is a scripted fake; close it afterwards."""
    created = []

    def make(
        session_class=ClaudeCodeSession,
        binary=FIXTURES / "fake_claude_session.py",
        use_scope=False,
    ):
        run_dir = tmp_path / f"run{len(created)}" / "logs"
        run_dir.mkdir(parents=True)
        global_logger = agDataLogger(
            agconfig(dataloggerconfig(db_path=str(run_dir / "global_data.sqlite3")))
        )
        global_logger.start()
        session = session_class(
            cwd=tmp_path,
            binary=str(binary),
            cli_args=[],
            run_dir=run_dir,
            ui_port=0,
            use_scope=use_scope,
        )
        session.start(global_logger=global_logger)
        created.append((session, global_logger))
        return session

    yield make
    for session, global_logger in created:
        session.close()
        global_logger.stop()
