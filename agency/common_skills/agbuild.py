from __future__ import annotations

from ..agskill import agskill
from ..agdata import agdata
from ..agtool import agtool

_BASE_PROMPT = """\
You are in BUILD mode. You have full access to execute code, write files, and run commands.

Workflow — follow these steps in order:

1. UNDERSTAND — Read your inputs and any relevant existing files carefully before writing
   any code. Use read and grep to examine what already exists if a sandbox is available.

2. PLAN — Break the task into concrete steps. Identify what to build, in what order, and
   what each step's success looks like. Consider failure modes.

3. IMPLEMENT — Execute the steps one at a time. Prefer editing existing files over creating
   new ones. After each meaningful change, verify it works before moving on.

4. VERIFY — Run the relevant test, script, or check to confirm correctness. If something
   fails, diagnose the root cause and fix it — do not work around the failure.

5. OUTPUT — Return your result only after you have verified it works.

"""


class agbuild(agskill):
    """Build / implementation skill: full sandboxed tool access.

    Inherits the complete tool set (bash, write, edit, read, grep, glob,
    webfetch, todowrite, gpu/cpu resources) when a sandbox is provided.
    The base build prompt is prepended to the caller's system_prompt.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        add_tools: list[agtool] | None = None,
        replace_tools: list[agtool] | None = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        max_output_schema_retries: int = 10,
    ):
        super().__init__(
            name=name,
            system_prompt=_BASE_PROMPT + system_prompt,
            add_tools=add_tools,
            replace_tools=replace_tools,
            input_schema=input_schema,
            output_schema=output_schema,
            max_output_schema_retries=max_output_schema_retries,
        )
