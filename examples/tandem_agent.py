import os
from agency import agent, agskill, agdata
from agency.configs.agconfig import agconfig, agentconfig, harnessadapterconfig, llmconfig
from agency.agtype import agpath

# harness="tandem" (see agency/tandem_harness/tandem_loop.py's own docstring
# for the design): the FIRST config here is the supervisor -- it keeps the
# full task history and drives the task via natural-language orders, never
# touching a tool itself. The SECOND is the worker -- it becomes
# agconfig.llm (the same field every harness already uses for its one
# model), since it's the one that actually calls tools in the sandbox.
model_config_list = [
    llmconfig(
        provider="litellm",
        base_url="https://router.js-park.info/v1",
        model="gpt-5.6-luna",
        api_key=os.environ["LITELLM_API_KEY"],
    ),
    llmconfig(
        provider="vllm",
        base_url="http://127.0.0.1:18000/v1",
        model="Qwen/Qwen3.5-4B",
        api_key="OHNiopuHBipuYgpighp983yhoBUIB89329",
    ),
]
supervisor_llm, worker_llm = model_config_list


def main():
    file_skill = agskill(
        name="file_manager",
        prompt=(
            "You are a file management assistant. "
            "Use the write and read tools to complete the task. "
            "Always confirm what you wrote by reading the file back. "
            "Write files to /workspace."
        ),
        input_schema=agdata(
            task=str,
            file_path=agpath,  # datatype for passing path in the sandbox
        ),
        output_schema=agdata(
            path=agpath,  # datatype for passing path in the sandbox
            content=str,
        ),
    )

    qa_skill = agskill(
        name="qa",
        prompt=(
            "Answer the user's question directly and concisely. "
            "You have access to prior conversation context."
        ),
        input_schema=agdata(question=str),
        output_schema=agdata(answer=str),
    )

    ag = agent(
        agconfig=agconfig(
            worker_llm,
            agentconfig(harness="tandem"),
            harnessadapterconfig(
                supervisor_model=supervisor_llm.model,
                supervisor_base_url=supervisor_llm.base_url,
                supervisor_api_key=supervisor_llm.api_key,
            ),
        )
    )

    print(">> [file_manager] write and verify a note")
    r1 = ag.run(
        file_skill,
        agdata(
            task="Write 'Hello from the agent!' to the given file and verify it.",
            file_path="/workspace/note.txt",
        ),
    )
    print(f"   path    : {r1.path!r}")
    print(f"   content : {r1.content!r}")
    print()

    print(">> [qa] ask about the note using shared history")
    r2 = ag.run(
        qa_skill,
        agdata(question="What was written to the note file, and where is it?"),
    )
    print(f"   answer : {r2.answer!r}")
    print()

    print(f"Shared history : {len(ag.history.messages)} messages total")


if __name__ == "__main__":
    from agency.observability.agwebui import agwebui

    agwebui.run(main, port=8009)
