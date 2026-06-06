from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agsandbox import agSandbox


class agtype:
    """Base class for typed agdata field values.

    Subclass to add custom serialization, sandbox I/O, system prompts, or
    cleanup logic for a field type.  The default implementations pass plain
    Python values through unchanged.

    Class-level use
    ---------------
    An ``agtype`` subclass is used as a *type marker* in agskill schemas::

        skill = agskill(
            name="write",
            system_prompt="...",
            input_schema=agdata(theme=str, background=agfile),
            output_schema=agdata(report=agfile),
        )

    The class object itself (not an instance) is stored in the schema agdata.
    All interface methods are therefore ``@classmethod`` so they can be called
    on the class without instantiation.

    Interface methods
    -----------------
    schema_type() -> str
        Human-readable type label shown in the JSON format hint in the system
        prompt.  Default: ``"str"``.

    needs_sandbox() -> bool
        True if this type requires sandbox filesystem access during
        prepare/recover.  Default: ``False``.

    prepare(value, sandbox, skill_name, field_name) -> tuple[str, list[str]]
        Called before the skill's ReAct loop.  ``value`` is the raw Python
        value from agdata.  Returns (transformed_value, paths_to_cleanup).
        ``transformed_value`` replaces the field in the JSON sent to the LLM.
        Default: returns (value, []).

    recover(value, sandbox) -> tuple[str, list[str]]
        Called after the skill's ReAct loop on output fields.  ``value`` is
        whatever the LLM returned for this field.  Returns
        (recovered_value, paths_to_cleanup).  Default: returns (value, []).

    extra_input_prompt(field_name) -> str
        Additional instruction line added to the system prompt for this input
        field.  Return an empty string to add nothing.  Default: ``""``.

    extra_output_prompt(field_name, skill_name) -> str
        Additional instruction line added to the system prompt for this output
        field.  Return an empty string to add nothing.  Default: ``""``.
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[object, list[str]]:
        return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return ""

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return ""


class agfile(agtype):
    """File-backed agskill schema field.

    Input fields
    ------------
    The framework writes the Python string value to a temporary file inside
    the agent's sandbox before the skill runs.  The LLM receives the file
    path and uses the ``read`` tool to access the content.

    Output fields
    -------------
    The agent writes its output to a file and returns the path.  The
    framework reads the file content back and stores it as a plain string
    in the result agdata.  The agent-side file is deleted after the skill
    ends.

    Example::

        design_skill = agskill(
            name="design",
            system_prompt="Create a story design document.",
            input_schema=agdata(theme=str),
            output_schema=agdata(design_doc=agfile),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "file"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return True

    @classmethod
    def prepare(
        cls,
        value: object,
        sandbox: "agSandbox",
        skill_name: str,
        field_name: str,
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        path = f"/workspace/inputs/{skill_name}_{field_name}.txt"
        try:
            sandbox.write_file(path, value)
            return path, [path]
        except Exception:
            return value, []

    @classmethod
    def recover(
        cls,
        value: object,
        sandbox: "agSandbox",
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        try:
            content = sandbox.read_file(value)
            return content, [value]
        except Exception:
            return value, []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: the JSON value is a path to a temporary "
            f"file in your sandbox. Use the read tool to access the full content "
            f"before completing the task."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your output to a file (e.g. "
            f"/workspace/outputs/{skill_name}_{field_name}.txt) and return only "
            f"the file path as the field value. The framework will read the "
            f"content automatically."
        )
