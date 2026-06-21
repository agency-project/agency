from __future__ import annotations
import base64
import mimetypes
from pathlib import Path
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

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        """Description for the return_<field> tool itself."""
        return f"Register the '{field_name}' output field."

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        """Description for the `value` parameter of return_<field>."""
        return f"Value for '{field_name}'"


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
        path = f"/workspace/inputs/{field_name}.txt"
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
            f"before completing the task. This input file is temporary and will be cleaned up automatically "
            f"after the task completes."
        )

    @classmethod
    def extra_output_prompt(cls, field_name: str, skill_name: str) -> str:
        return (
            f"  - Output `{field_name}`: write your output to a file (e.g. "
            f"/workspace/outputs/{field_name}.txt) and return only "
            f"the file path as the field value. This output file is temporary and will be cleaned up automatically "
            f"after the task completes. "
            f"If the output is long, write it in multiple smaller tool calls (e.g. write the first portion, "
            f"then append subsequent portions) rather than generating it all in one response — "
            f"this avoids hitting generation length limits."
        )

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        return (
            f"Register the '{field_name}' output file. "
            f"Write your output to a sandbox file first (e.g. /workspace/outputs/{field_name}.txt), "
            f"then call this tool with that file path as the value."
        )

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        return (
            f"Absolute path to the file you wrote in the sandbox "
            f"(e.g. /workspace/outputs/{field_name}.txt). Do NOT pass the file content — pass the path."
        )


class agimage(agtype):
    """Image input field for agskill schemas.

    The caller supplies a local file path, an http/https URL, or an existing
    data URL.  Local files are base64-encoded by the framework before the skill
    runs.  The image is injected directly into the multimodal content array of
    the user message — the LLM sees it as a visual input, not as text.

    Single image::

        skill = agskill(
            name="describe",
            system_prompt="Describe the image.",
            input_schema=agdata(question=str, photo=agimage),
        )

    List of images::

        skill = agskill(
            name="compare",
            system_prompt="Compare the images.",
            input_schema=agdata(question=str, frames=list[agimage]),
        )
    """

    @classmethod
    def schema_type(cls) -> str:
        return "image"

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
    ) -> tuple[str, list[str]]:
        if not isinstance(value, str):
            return value, []
        # Already a URL or data URL — pass through unchanged.
        if value.startswith(("http://", "https://", "data:")):
            return value, []
        # Local file path — read and base64-encode.
        path = Path(value)
        raw = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}", []

    @classmethod
    def extra_input_prompt(cls, field_name: str) -> str:
        return (
            f"  - Input `{field_name}`: an image attached directly to this message "
            f"as a visual input. The JSON field value is a placeholder — the actual "
            f"image is visible to you in the message content."
        )

    @classmethod
    def return_tool_description(cls, field_name: str) -> str:
        return f"Register the '{field_name}' image output as a file path or data URL."

    @classmethod
    def return_value_description(cls, field_name: str) -> str:
        return f"File path, http/https URL, or data: URL of the image for '{field_name}'."


class agrawstring(agtype):
    """Raw string field — bypasses JSON input/output formatting entirely.

    Input
    -----
    When used as the sole input field, the string value is sent directly as
    the user message content.  No JSON wrapping, no schema hint.

    Output
    ------
    When used as the sole output field, the model's complete text response is
    captured as-is.  No JSON parsing, no retry loop.

    Constraint
    ----------
    Must be the only field in its input or output schema.

    Example::

        write_skill = agskill(
            name="write_chapter",
            system_prompt="You are a novelist. Write the chapter as requested.",
            input_schema=agdata(prompt=agrawstring),
            output_schema=agdata(chapter=agrawstring),
        )

        result = ag.run(write_skill, agdata(prompt="Write a dark opening scene."))
        print(result.chapter)   # the model's prose, unmodified
    """

    @classmethod
    def schema_type(cls) -> str:
        return "str"

    @classmethod
    def needs_sandbox(cls) -> bool:
        return False
