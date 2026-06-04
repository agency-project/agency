from __future__ import annotations


def agsync(targets) -> None:
    """Block until every in-flight agent task in *targets* has finished.

    *targets* may be any of:

    * a single ``agent``
    * a single ``agteam``
    * a list containing any mix of ``agent`` and ``agteam`` objects

    For an ``agteam``, every agent tracked by the team is included — both
    agents created in ``setup()`` and any fork agents (``agent(parent)``)
    created dynamically during ``run()``.  The team's background thread
    (started by ``run()``) is also joined.

    Example::

        from agency import agsync

        # Single agent or team
        agsync(my_agent)
        agsync(my_team)

        # List — any mix of agents and teams
        teams   = [ResearchTeam(topic=t) for t in topics]
        pending = [t.run() for t in teams]
        agsync(teams)                   # barrier: wait for all to finish
        for r in pending:
            print(r.report_path)        # all resolved, no blocking

        # Mix agents and teams
        agsync([my_agent, my_team])
    """
    from .agent import agent as _agent_cls
    from .agteam import agteam as _agteam_cls

    if not isinstance(targets, list):
        targets = [targets]

    solo_agents: list = []
    teams: list = []

    for t in targets:
        if isinstance(t, _agent_cls):
            solo_agents.append(t)
        elif isinstance(t, _agteam_cls):
            teams.append(t)
        else:
            raise TypeError(
                f"agsync: expected agent or agteam, got {type(t).__name__!r}"
            )

    # Join submit() threads first — fork agents registered by the background
    # thread must be fully visible in the WeakSet before we snapshot it.
    for team in teams:
        if team._run_future is not None:
            team._run_future.result()

    # Snapshot WeakSet now; dead entries are skipped automatically.
    team_agents = [ag for team in teams for ag in team._agents]

    for ag in solo_agents + team_agents:
        ag._history._resolve()
