from contextvars import ContextVar

# Set to the active agteam instance while its run() method is executing.
# Used by agent.__init__ to auto-register fork agents with the enclosing team.
_active_team: ContextVar = ContextVar("_active_team", default=None)
