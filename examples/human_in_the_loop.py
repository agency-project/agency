"""
human_in_the_loop.py — Human-in-the-loop collaborative scene writer.

The human acts as the creative director, approving or revising each step:

  1. Python asks human what scene to write next (guaranteed ask_human call).
  2. Planner generates a paragraph-by-paragraph scene plan.
  3. Python presents the plan and asks for approval (guaranteed ask_human call).
     Loops with revision feedback until approved.
  4. Writer generates the full scene prose from the approved plan.
  5. Python presents the scene and asks for approval (guaranteed ask_human call).
     Loops — re-plans then re-writes — until approved.
  6. On approval, plan and text are saved; the loop advances to the next scene.

Human interaction is driven entirely from Python, not delegated to the LLM.
This guarantees ask_human is always called — the LLM cannot shortcut approvals.

Output files written to runs/<timestamp>_human_in_the_loop/:
  plans.md   — one section per scene, overwritten if revised
  story.txt  — approved scene texts appended in order, overwritten if revised

Usage:
    uv run python examples/human_in_the_loop.py
    LLM_MODEL=gpt-4o uv run python examples/human_in_the_loop.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from agency import agent, agskill, agdata, agfile
from agency.agconfig import agConfig
from agency.agllm_backends import agVLLMBackendConfig
from agency.agwebui import agwebui
from agency.tools.human import make_ask_human

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
# Human interaction helpers
# ---------------------------------------------------------------------------


def _is_approved(reply: str) -> bool:
    """Strict approval check — only 'yes' or 'y' counts, nothing else."""
    return reply.strip().lower() in ("yes", "y")


def _ask(ask_human_tool, question: str) -> str:
    """Call ask_human directly from Python and return the reply string."""
    return ask_human_tool(agdata(question=question)).reply


# ---------------------------------------------------------------------------
# File helpers — overwrite a scene's section in place
# ---------------------------------------------------------------------------


def _update_section(path: Path, marker: str, content: str) -> None:
    """Insert or replace a marked section in a file."""
    block = f"{marker}\n\n{content.strip()}\n\n"
    pattern = rf"{re.escape(marker)}.*?(?={re.escape('## Scene ') if '##' in marker else re.escape('=== Scene ')}|\Z)"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        text = re.sub(pattern, block, text, flags=re.DOTALL)
    else:
        text += block
    path.write_text(text, encoding="utf-8")


def save_plan(path: Path, scene_num: int, plan: str) -> None:
    _update_section(path, f"## Scene {scene_num}", plan)


def save_scene(path: Path, scene_num: int, text: str) -> None:
    _update_section(path, f"=== Scene {scene_num} ===", text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_human_in_the_loop"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"

    plans_path = run_dir / "plans.md"
    story_path = run_dir / "story.txt"

    print(f"[story] Run dir : {run_dir}")
    print(f"[story] Plans   : {plans_path}")
    print(f"[story] Story   : {story_path}\n")

    # ── Agents ───────────────────────────────────────────────────────────────

    planner = agent(agconfig=cfg, agname="PlannerAgent")
    writer = agent(agconfig=cfg, agname="WriterAgent")

    # ask_human_tool runs in-process (run_in_subprocess=False) so it can reach
    # the live webUI singleton.
    ask_human_tool = make_ask_human(planner.agname, timeout_s=None)

    # ── Skills (pure generation — no human interaction) ───────────────────────

    plan_skill = agskill(
        name="plan_scene",
        system_prompt=(
            "You are a creative story planner. "
            "Given the scene description and the story so far, produce a scene plan "
            "with two sections:\n\n"
            "GLOBAL NOTES (optional): Any concerns that apply to the whole scene — "
            "tone, style, voice, pacing, mood, or other cross-cutting directives. "
            "Omit this section entirely if there is nothing to say at this level.\n\n"
            "PARAGRAPHS: A breakdown of what happens in each paragraph of the scene. "
            "Include whatever is relevant for each paragraph — you decide.\n\n"
            "If revision feedback is provided, address every point. "
            "Output only the plan — no prose, no meta-commentary."
        ),
        input_schema=agdata(
            scene_description=str,
            scene_number=int,
            previous_story=agfile,
            feedback=str,
        ),
        output_schema=agdata(plan=str),
    )

    write_skill = agskill(
        name="write_scene",
        system_prompt=(
            "You are a skilled fiction writer. "
            "You will receive a scene plan that may contain two sections: "
            "GLOBAL NOTES (cross-cutting directives for the whole scene) and "
            "PARAGRAPHS (what happens in each paragraph). "
            "Apply the global notes uniformly across the entire scene. "
            "Expand each planned paragraph into polished prose. "
            "Maintain continuity with previous scenes. "
            "Output only the scene text — no headings, no plan references, "
            "no meta-commentary."
        ),
        input_schema=agdata(plan=str, scene_number=int, previous_story=agfile),
        output_schema=agdata(scene_text=str),
    )

    # ── Main loop ─────────────────────────────────────────────────────────────

    scene_num = 1
    previous_story = ""

    while True:
        print(f"\n{'─' * 60}")
        print(f"  Scene {scene_num}")
        print(f"{'─' * 60}")

        # ── Step 1: Ask human for scene description (Python-level) ───────────
        print(
            f"[Scene {scene_num}] Waiting for scene description from human... (Please use the webUI to interact with the agent.)"
        )
        scene_description = _ask(
            ask_human_tool,
            f"What scene would you like written next? "
            f"(Scene {scene_num}"
            + (f" — story so far: {len(previous_story)} chars" if previous_story else "")
            + ") — or type 'stop' to finish.",
        )

        if scene_description.strip().lower() in ("stop", "quit", "exit"):
            print("\n[story] Human requested stop. Exiting.")
            break

        print(f"[Scene {scene_num}] Description: {scene_description!r}")

        # ── Step 2: Plan → human review loop (Python-level approval) ─────────
        plan_feedback = ""
        plan = ""
        while True:
            print(
                f"[Scene {scene_num}] Generating plan"
                + (" (with revision feedback)" if plan_feedback else "")
                + "..."
            )
            plan_result = planner.run(
                plan_skill,
                agdata(
                    scene_description=scene_description,
                    scene_number=scene_num,
                    previous_story=previous_story,
                    feedback=plan_feedback,
                ),
            )
            plan = plan_result.plan
            save_plan(plans_path, scene_num, plan)
            print(f"[Scene {scene_num}] Plan saved → {plans_path.name}")

            print(f"[Scene {scene_num}] Asking human to review plan...")
            reply = _ask(
                ask_human_tool,
                f"Scene {scene_num} Plan:\n\n{plan}\n\n"
                "Type YES to approve, or describe the changes you want.",
            )

            if _is_approved(reply):
                print(f"[Scene {scene_num}] Plan approved.")
                break

            plan_feedback = reply
            print(f"[Scene {scene_num}] Plan revision requested: {plan_feedback!r}")

        # ── Step 3: Write scene → human review loop (Python-level approval) ───
        text_feedback = ""
        scene_text = ""
        while True:
            if text_feedback:
                # Human rejected the prose — re-plan, then get plan approval before writing
                plan_feedback = (
                    f"The written scene was rejected. Human feedback: {text_feedback}\n"
                    "Revise the plan to address these concerns."
                )
                while True:
                    print(f"[Scene {scene_num}] Re-planning from text feedback...")
                    plan_result = planner.run(
                        plan_skill,
                        agdata(
                            scene_description=scene_description,
                            scene_number=scene_num,
                            previous_story=previous_story,
                            feedback=plan_feedback,
                        ),
                    )
                    plan = plan_result.plan
                    save_plan(plans_path, scene_num, plan)
                    print(f"[Scene {scene_num}] Revised plan saved → {plans_path.name}")

                    print(f"[Scene {scene_num}] Asking human to review revised plan...")
                    reply = _ask(
                        ask_human_tool,
                        f"Revised Scene {scene_num} Plan:\n\n{plan}\n\n"
                        "Type YES to approve, or describe further changes.",
                    )

                    if _is_approved(reply):
                        print(f"[Scene {scene_num}] Revised plan approved.")
                        break

                    plan_feedback = reply
                    print(f"[Scene {scene_num}] Plan revision requested: {plan_feedback!r}")

            print(f"[Scene {scene_num}] Writing scene...")
            write_result = writer.run(
                write_skill,
                agdata(plan=plan, scene_number=scene_num, previous_story=previous_story),
            )
            scene_text = write_result.scene_text
            save_scene(story_path, scene_num, scene_text)
            print(f"[Scene {scene_num}] Scene saved → {story_path.name}")

            print(f"[Scene {scene_num}] Asking human to review scene text...")
            reply = _ask(
                ask_human_tool,
                f"Scene {scene_num}:\n\n{scene_text}\n\n"
                "Type YES to approve, or describe what you would like changed.",
            )

            if _is_approved(reply):
                print(f"[Scene {scene_num}] Scene approved.")
                break

            text_feedback = reply
            print(f"[Scene {scene_num}] Scene revision requested: {text_feedback!r}")

        # ── Step 4: Commit and advance ────────────────────────────────────────
        previous_story += f"\n\n=== Scene {scene_num} ===\n\n{scene_text}"
        print(f"[Scene {scene_num}] Complete. Advancing to scene {scene_num + 1}.")
        scene_num += 1


if __name__ == "__main__":
    agwebui.run(main, port=8004)
