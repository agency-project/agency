# configs — component configuration

`agency/configs/` defines the settings passed through Agency's components. It gives an application one configuration object while preserving explicit ownership of model, sandbox, scheduling and tool settings.

## Structure

[agconfig.py](../../agency/configs/agconfig.py) contains `agconfig` and its namespace dataclasses. Namespaces such as `llmconfig`, `sandboxconfig`, `orchestratorconfig`, `skillconfig` and `ptraceconfig` collect fields for the component that consumes them. `agconfig` assembles these namespaces and accepts explicit replacements by their dataclass type.

The namespace boundary is useful when composing agents: changing a model endpoint belongs to `llm`, selecting Docker or Podman belongs to `sandbox`, and limiting active requests belongs to `orchestrator`. A harness choice and a provider choice are separate settings.

## How settings reach components

Agents and several runtime components clone the supplied configuration so later changes to the caller's object do not silently mutate their state. Their `change_config()` methods provide an explicit update path. `update()` merges named fields and rejects unknown names; it does not perform complete semantic validation of every value. Provider and runtime code apply their own checks.

`safe_snapshot()` produces JSON-safe namespace values and omits declared secrets and non-serializable objects. Logging and saved agent metadata can therefore describe configuration without copying the whole live credential-bearing object. The [host profile loader](host.md) applies selected sandbox defaults during configuration construction, before explicitly supplied namespaces replace them.

This package defines configuration data. Runtime allocation belongs to [orchestrator](orchestrator.md), provider behavior to [llm](llm.md), and container creation to [sandbox](sandbox.md). Full field definitions are in the [configuration reference](../api/configuration.md).
