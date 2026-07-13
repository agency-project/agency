"""
Image processing example.

Demonstrates agimage — the multimodal image input field type.

Three patterns are shown:

  1. SingleImageTeam   — describe a single image passed as a local file path.

  2. MultiImageTeam    — compare two images passed as a list[agimage].

  3. UrlImageTeam      — analyse an image passed as an http URL (no encoding needed).

The model in cfg.agllm_backend.model must support vision.  Set LLM_MODEL to a
multimodal model, e.g. Qwen/Qwen2.5-VL-7B-Instruct.

Run:
    uv run python examples/image_processing.py /path/to/image.jpg [/path/to/image2.jpg]
"""

import os
import sys
from pathlib import Path

from agency import agent, agdata, agimage, agskill, agteam, agsync
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig

# See ../README.md for OpenAI, Anthropic, or Bedrock agconfig examples.
cfg = agConfig(
    agVLLMBackendConfig(
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=0.7,
        top_p=0.95,
        top_k=20,
    )
)

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

describe_skill = agskill(
    name="describe_image",
    system_prompt=(
        "You are a visual analysis assistant. "
        "Describe the image in detail: content, colours, layout, and any text visible."
    ),
    input_schema=agdata(question=str, photo=agimage),
    output_schema=agdata(description=str),
)

compare_skill = agskill(
    name="compare_images",
    system_prompt=(
        "You are a visual comparison assistant. "
        "Compare the provided images and describe what is similar and what differs between them."
    ),
    input_schema=agdata(question=str, frames=list[agimage]),
    output_schema=agdata(comparison=str),
)

analyse_url_skill = agskill(
    name="analyse_url_image",
    system_prompt=(
        "You are a visual analysis assistant. Analyse the image and answer the user's question."
    ),
    input_schema=agdata(question=str, image_url=agimage),
    output_schema=agdata(answer=str),
)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


class SingleImageTeam(agteam):
    """Describe a single local image file."""

    agconfig = cfg

    def setup(self) -> None:
        self.ag = agent()

    def run(self) -> agdata:
        image_path = getattr(self, "image_path", "")
        print(f"\n[SingleImage] Describing: {image_path}")
        return self.ag.run(
            describe_skill,
            agdata(
                question="Please describe this image in detail.",
                photo=image_path,
            ),
        )


class MultiImageTeam(agteam):
    """Compare two local image files side by side."""

    agconfig = cfg

    def setup(self) -> None:
        self.ag = agent()

    def run(self) -> agdata:
        paths = getattr(self, "image_paths", [])
        print(f"\n[MultiImage] Comparing {len(paths)} images")
        return self.ag.run(
            compare_skill,
            agdata(
                question="What is similar and what differs between these images?",
                frames=paths,
            ),
        )


class UrlImageTeam(agteam):
    """Analyse an image from a public URL — no local file needed."""

    agconfig = cfg

    def setup(self) -> None:
        self.ag = agent()

    def run(self) -> agdata:
        url = getattr(self, "url", "")
        question = getattr(self, "question", "What does this image show?")
        print(f"\n[UrlImage] Analysing: {url}")
        return self.ag.run(
            analyse_url_skill,
            agdata(
                question=question,
                image_url=url,
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime
    from agency.agwebui import agwebui

    args = sys.argv[1:]

    run_dir = (
        Path(__file__).parent.parent
        / "runs"
        / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_image_processing"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    def _script() -> None:
        print(f"Model  : {cfg.agllm_backend.model or 'default'}")
        print(f"Rundir : {run_dir}\n")

        teams = []

        # Pattern 1 — single image from local file
        if args:
            t1 = SingleImageTeam(image_path=args[0])
            r1 = t1.run()
            teams.append((t1, r1, "description"))

        # Pattern 2 — compare two images (requires two paths)
        if len(args) >= 2:
            t2 = MultiImageTeam(image_paths=args[:2])
            r2 = t2.run()
            teams.append((t2, r2, "comparison"))

        # Pattern 3 — image from URL (uses a public sample if no URL provided)
        sample_url = "https://upload.wikimedia.org/wikipedia/commons/1/15/Cat_August_2010-4.jpg"
        t3 = UrlImageTeam(
            url=sample_url,
            question="What is this image of?",
        )
        r3 = t3.run()
        teams.append((t3, r3, "answer"))

        agsync(*[t for t, _, _ in teams])

        from agency import AgError

        for _, result, field in teams:
            try:
                val = getattr(result, field)
            except AgError as e:
                val = f"ERROR: {e}"
            print(f"\n[{field}]\n{val}")

    agwebui.run(_script, port=8005)
