from __future__ import annotations

from ..agskill import agskill
from ..agdata import agdata
from ..agtool import agtool

_BASE_PROMPT = """\
You are in PLAN mode. Analyze inputs and produce a structured plan or analysis report.
Do NOT write code, execute commands, or modify any files.

Workflow — follow these steps in order:

1. UNDERSTAND — Read the inputs carefully. If a sandbox is available, use read, grep, and
   glob to examine relevant existing code, files, and context before forming any opinions.

2. DESIGN — Reason through the task. Consider alternative approaches and their trade-offs.
   Identify risks, dependencies, and open questions.

3. REVIEW — Re-examine the most critical inputs or files. Verify that your understanding
   is correct and that your plan addresses all requirements. Resolve any open questions.

4. OUTPUT — Produce your final structured output. It must cover: what to do, in what order,
   and how to verify the result. Be specific and self-contained.

"""


class agplan(agskill):
    """Planning / analysis skill: read-only tools, no code execution.

    When run with a sandbox, provides read, grep, glob, and webfetch.
    When run without a sandbox, has no tools (pure-reasoning mode).
    The base planning prompt is prepended to the caller's system_prompt.

    Does not accept replace_tools (tool set is fixed by the class).
    Use add_tools to inject additional read-only tools (e.g. web search).
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        add_tools: list[agtool] | None = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        max_output_schema_retries: int = 10,
    ):
        super().__init__(
            name=name,
            system_prompt=_BASE_PROMPT + system_prompt,
            add_tools=add_tools,
            replace_tools=None,  # _build_tools handles tool selection
            input_schema=input_schema,
            output_schema=output_schema,
            max_output_schema_retries=max_output_schema_retries,
        )

    def _build_tools(self, sandbox, pool, term, log, _ensure_read: bool = False):
        from ..tools import make_read, make_grep, make_glob, webfetch

        if sandbox is not None:
            active_tools: list[agtool] = [
                make_read(sandbox),
                make_grep(sandbox),
                make_glob(sandbox),
                webfetch,
            ]
        else:
            active_tools = []
        if self.add_tools:
            active_tools.extend(self.add_tools)
        for t in active_tools:
            t.attach_logger(term, log)
        tool_map = {t.name: t for t in active_tools}
        openai_tools = [t.to_openai_tool() for t in active_tools] or None
        return active_tools, tool_map, openai_tools
