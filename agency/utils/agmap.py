"""agmap — run non-agent functions concurrently, the deterministic analog of
fanning out ``agent.run()``.

``agmap`` maps a function (or functions) over an item (or items), running each
call on its own background thread and returning results as ``agtask`` (an
``agdata`` subclass) — so plain Python work (e.g. forking a sandbox, applying a
patch, running a test) can be parallelised without a ``ThreadPoolExecutor`` or a
throwaway ``agteam``.

    # synchronous (default): block, return resolved results
    results = agmap(validate, candidates)          # [agtask, agtask, ...]

    # asynchronous: return pending agtasks immediately; join them explicitly
    pending = agmap(validate, candidates, is_asynchronous=True)
    ...                                            # do other work
    agsync(pending)                                # barrier — like agsync(team)

Pairing (either argument may be a single value or a list):
  - one fn,  many items -> fn(item) for each item        (the common "map" case)
  - many fns, one item  -> fn(item) for each fn
  - many fns, many items-> zip: fns[i](items[i])          (equal lengths)
  - one fn,  one item   -> a single fn(item)

Each call's return value is wrapped in an ``agtask`` (an ``agdata`` return is
used as its payload unchanged); a raised exception becomes ``agerror`` so one
failing task never crashes the others. The return shape mirrors the input: pass
a single fn+item and you get one ``agtask`` back; pass any list and you get a
list.

Joining: results resolve lazily like any agdata — access a field, call
``agdata.wait_all([...])``, or pass them to ``agsync`` (which recognises
``agtask`` targets exactly as it recognises teams).

Container creation inside the mapped function is throttled by ``agsandbox``'s
container semaphore, so mapping over a large list never starts unbounded
containers even though each task gets its own thread.

Profiling: each task runs on a traced thread under its own
``agmap:{fn}[{index}]`` span (see ``_spawn``), so a fan-out appears as children
of whatever ran the map instead of as N disconnected trace roots.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Callable

from ..agdata import agdata, agerror
from .agutil import format_exception
from ..observability.profiler import agprof


class agtask(agdata):
    """A (possibly pending) result of one ``agmap`` call.

    Subclass of ``agdata`` purely so ``agsync`` can recognise agmap tasks as
    joinable targets — it behaves identically to agdata otherwise (lazy field
    access, ``wait_all``, ``to_dict``, error inspection). Plain agdata objects
    are still rejected by ``agsync``, keeping its type checking strict.
    """


def _spawn(fn: "Callable[[object], object]", arg: object, index: int = 0) -> agtask:
    """Run ``fn(arg)`` on a traced daemon thread; return a pending agtask
    immediately.

    The thread comes from ``agprof.spawn_traced()`` rather than a bare
    ``threading.Thread`` so each task is a *child* of whatever ran the map,
    which is the truthful parentage: the map call is what caused it. A bare
    thread starts with empty ``contextvars``, which would make every mapped
    task a disconnected trace root — and silently, since a disconnected root
    is a valid trace, not an error. This is the highest-fan-out spawn site in
    the framework (one thread per mapped item, deliberately unbounded), so it
    is also where flat parentage would cost the most.

    ``agmap:{fn}[{index}]`` is deliberately a *task* label, not a
    ``run{N}:`` one: run-shaped labels are what ``run_metrics`` counts, and a
    mapped function is not an agent run. Laying these spans out as flat lanes
    in a timeline view stays a rendering concern (a synthetic ``tid`` per
    span), never a parentage one, so ``thread_name()`` here is naming only.

    Tasks of an ``is_asynchronous=True`` map may outlive the span that
    enclosed the ``agmap()`` call. That is legal — ``parent_span_id`` is a
    stored field, so the trace stays correct — and ``agsync`` is the natural
    join point.
    """
    future: "Future[agdata]" = Future()
    label = f"agmap:{getattr(fn, '__name__', type(fn).__name__)}[{index}]"

    def _run() -> None:
        agprof.thread_name(label)
        with agprof.span(label):
            try:
                out = fn(arg)
                result = out if isinstance(out, agdata) else agdata(result=out)
            except Exception as e:  # noqa: BLE001 — mirror skills: never propagate
                result = agerror(format_exception(e))
            error = result._data.get("error")
            agprof.annotate(
                outcome="failure" if error else "success",
                error_type="agmap_task_error" if error else None,
            )
        # Resolved only after the span above fully exits (and is persisted),
        # so a caller unblocked by this future can never race agprof.stop()
        # into reading the profile store before this task's span lands in it.
        future.set_result(result)

    agprof.spawn_traced(_run).start()
    return agtask(_future=future)


def _as_list(x: object) -> "tuple[list, bool]":
    """Return (list_form, was_scalar). Lists/tuples are treated as many items."""
    if isinstance(x, (list, tuple)):
        return list(x), False
    return [x], True


def agmap(
    fn: "Callable | list[Callable]",
    items: "object | list",
    *,
    is_asynchronous: bool = False,
) -> "agtask | list[agtask]":
    """Run ``fn`` over ``items`` concurrently — the non-agent analog of fanning
    out ``agent.run()``. See the module docstring for pairing and return-shape
    rules.

    is_asynchronous=False (default): block until all tasks finish; return the
        resolved result(s).
    is_asynchronous=True: return the pending result(s) immediately. Join them by
        passing them to ``agsync`` (e.g. ``agsync(pending)``), with
        ``agdata.wait_all``, or by accessing a field.
    """
    fns, fn_scalar = _as_list(fn)
    its, item_scalar = _as_list(items)

    if len(fns) == 1 and len(its) >= 1:
        pairs = [(fns[0], it) for it in its]
    elif len(its) == 1 and len(fns) > 1:
        pairs = [(f, its[0]) for f in fns]
    elif len(fns) == len(its):
        pairs = list(zip(fns, its))
    else:
        raise ValueError(
            f"agmap: cannot pair {len(fns)} function(s) with {len(its)} item(s) — "
            "provide one function, one item, or equal-length lists"
        )

    pending = [_spawn(f, it, index) for index, (f, it) in enumerate(pairs)]

    if not is_asynchronous:
        agdata.wait_all(pending)

    # Mirror the input shape: a lone fn+item yields a single agtask.
    if fn_scalar and item_scalar:
        return pending[0]
    return pending
