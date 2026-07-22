from __future__ import annotations


def _agsync_impl(*targets) -> None:
    """Block until every in-flight agent task in *targets* has finished.

    *targets* may be any combination of:

    * a single ``agent``
    * a single ``agteam``
    * a single ``agtask`` (a pending ``agmap`` result)
    * a list containing any mix of ``agent``, ``agteam``, and ``agtask`` objects
      (e.g. the list returned by ``agmap(..., is_asynchronous=True)``)

    For an ``agteam``, every agent tracked by the team is included — both
    agents created in ``setup()`` and any fork agents (``agent(parent)``)
    created dynamically during ``run()``.  The team's background thread
    (started by ``run()``) is also joined.

    Example::

        from agency import agsync

        # Single agent or team
        agsync(my_agent)
        agsync(my_team)

        # Variadic — any mix of agents and teams
        agsync(agent_a, team_b, agent_c)

        # List still works
        teams = [ResearchTeam(topic=t) for t in topics]
        agsync(teams)

        # Mix of variadic and lists
        agsync(my_agent, teams)
    """
    from .agent import agent as _agent_cls
    from .agteam import agteam as _agteam_cls
    from .agmap import agtask as _agtask_cls

    # Flatten: each positional arg may itself be a list
    flat: list = []
    for t in targets:
        if isinstance(t, list):
            flat.extend(t)
        else:
            flat.append(t)
    targets = flat

    solo_agents: list = []
    teams: list = []
    tasks: list = []

    for t in targets:
        if isinstance(t, _agent_cls):
            solo_agents.append(t)
        elif isinstance(t, _agteam_cls):
            teams.append(t)
        elif isinstance(t, _agtask_cls):
            tasks.append(t)
        else:
            raise TypeError(f"agsync: expected agent, agteam, or agtask, got {type(t).__name__!r}")

    # Join ALL team threads before raising any exception, so that no team is
    # abandoned mid-run. Collect exceptions and re-raise after everything joins.
    errors: list[BaseException] = []
    for team in teams:
        if team._run_future is not None:
            try:
                team._run_future.result()
            except Exception as exc:
                errors.append(exc)

    # Snapshot WeakSet now; dead entries are skipped automatically.
    team_agents = [ag for team in teams for ag in team._agents]

    for ag in solo_agents + team_agents:
        ag.ctx.resolve_prev_dependencies()

    # Join any agmap tasks passed as targets. Task errors resolve to agerror
    # (agmap's never-crash-siblings contract) rather than re-raising here.
    for task in tasks:
        task._resolve()

    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise ExceptionGroup(f"agsync: {len(errors)} team(s) failed", errors)


def agsync(*targets) -> None:
    from .profiler import agprof

    with agprof.span("agsync:join"):
        return _agsync_impl(*targets)


agsync.__doc__ = _agsync_impl.__doc__
