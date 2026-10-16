"""One harness x model x context-limit run: where does the harness compact?

    sudo bash -c 'set -a; . ./.env; set +a; .venv/bin/python benchmarks/context_window/run.py \
        --harness codex --model claude-haiku-5-5 --provider anthropic --limit 120000 \
        --out runs/context_window/results.jsonl'

See benchmarks/context_window/README.md.

The agent follows a chain of files, each ~KB kilobytes of random words whose
first and last lines name the next file, so its context grows by one file per
step. Agency's own log of every LLM call then shows the prompt size per call
and where it dropped (a compaction). `context_limit` overrides the model's
window so that thresholds are reachable cheaply; it is what every adapter
configures its harness with.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agency import Agent, agdata, agrawstring, agskill
from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    harnessadapterconfig,
    llmconfig,
    orchestratorconfig,
    resourcesconfig,
    sandboxconfig,
)
from agency.native_harness.usage import prompt_tokens


BINARIES = {
    "claude_code": "/usr/local/bin/claude",
    "codex": "/usr/local/bin/codex",
    "grok": "/usr/local/bin/grok",
    "opencode": "/usr/local/bin/opencode",
    "kimi": "/usr/local/bin/kimi",
}


def task(files: int, kb: int) -> str:
    lines = kb * 1024 // 66
    return f"""You are testing how much text you can hold in context.

1. Run this one shell command to create a chain of files:
   python3 -c "import random,os; r=random.Random(7); w=[''.join(r.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(r.randint(3,9))) for _ in range(5000)]; n=['/tmp/chain/'+''.join(r.choice('abcdefghij') for _ in range(8))+'.txt' for _ in range({files})]; os.makedirs('/tmp/chain',exist_ok=True)
for i,p in enumerate(n):
  nxt=n[i+1] if i+1<len(n) else 'END'; c=f'CODE-{{i+1:02d}}-'+r.choice(w).upper(); f=open(p,'w'); f.write(f'{{c}} NEXT: {{nxt}}'+chr(10)); [f.write(' '.join(r.choice(w) for _ in range(10))+chr(10)) for _ in range({lines})]; f.write(f'{{c}} NEXT: {{nxt}}'+chr(10)); f.close()
print('START:', n[0])"
2. Print the START file IN FULL into this conversation (for example with `cat`, or your file-reading
   tool without offsets or limits). Never use head, tail, grep, sed, wc or any other way of looking at
   only part of a file: the point of the test is to load every file completely into the conversation.
   Its first and last lines say `CODE-.. NEXT: <path>`. Print that next file in full the same way, and
   keep following NEXT until a file says NEXT: END. You can only learn each path by printing the
   previous file, so print exactly one file per step: one `cat <path>` (or one read) per tool call.
   Do not write scripts, loops or pipelines that follow the chain for you; that defeats the test.
3. Your harness may summarize older parts of the conversation when it gets long. If that happens,
   do not re-read files you have already printed: continue with the next file in the chain, and keep
   the CODE values you have seen in your notes.
4. Answer with the list of all CODE values in order.
"""


def calls(db: Path) -> "list[dict]":
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    order = {cl: ts for cl, ts in c.execute("select call_label, timestamp from exchanges")}
    rows = []
    for cl, payload in c.execute(
        "select call_label, payload from message_block_lookup where payload like '%prompt_tokens%'"
    ):
        block = json.loads(payload)
        usage = block.get("usage") or {}
        p = prompt_tokens(usage)
        rows.append(
            {
                "ts": order.get(cl, 0),
                "prompt": p.total,
                "reported_prompt_tokens": usage.get("prompt_tokens"),
                "cache_read": p.cache_read,
                "cache_write": p.cache_write,
                "out": usage.get("completion_tokens") or 0,
                "stop": block.get("stop_reason"),
            }
        )
    rows.sort(key=lambda r: str(r["ts"]))
    return rows


def summarize(rows, limit):
    prompts = [r["prompt"] for r in rows]
    compactions = [i for i in range(1, len(prompts)) if prompts[i] < 0.9 * prompts[i - 1]]
    first = compactions[0] if compactions else None
    return {
        "calls": len(rows),
        "max_prompt": max(prompts, default=0),
        "max_prompt_share": round(max(prompts, default=0) / limit, 3),
        "compactions": len(compactions),
        # The largest prompt sent before the first drop: where it compacted.
        "compacted_at": max(prompts[:first]) if first else None,
        "compacted_at_share": round(max(prompts[:first]) / limit, 3) if first else None,
        "prompts": prompts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="litellm")
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--files", type=int, default=18)
    ap.add_argument("--kb", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default="localhost/agency-harnesses:pinned")
    ap.add_argument("--max-completion-tokens", type=int, default=8192)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    name = f"{args.tag}_{args.harness}_{args.model}_{args.provider}_{args.limit}_{stamp}"
    run_dir = Path(args.out).resolve().parent / name
    run_dir.mkdir(parents=True)
    llm = {
        "provider": args.provider,
        "model": args.model,
        "context_limit": args.limit,
        "max_completion_tokens": args.max_completion_tokens,
    }
    if args.provider == "litellm":
        llm.update(api_key=os.environ["LITELLM_API_KEY"], base_url=os.environ["LITELLM_BASE_URL"])
    else:
        llm["api_key"] = os.environ["ANTHROPIC_API_KEY"]
    cfg = agconfig(
        llmconfig(**llm),
        agentconfig(
            harness=args.harness,
            log_dir=str(run_dir / "logs"),
            output_dir=str(run_dir / "agent_output"),
        ),
        sandboxconfig(base_image=args.image, gpu_passthrough=False),
        resourcesconfig(idle_cpus=1),
        harnessadapterconfig(binary_path=BINARIES.get(args.harness)),
        orchestratorconfig(db_path=str(run_dir / "agency.sqlite3")),
    )
    skill = agskill(
        name="follow_chain",
        prompt=task(args.files, args.kb),
        input_schema=agdata(request=agrawstring),
        output_schema=agdata(answer=agrawstring),
    )
    agent = Agent("ctx", agconfig=cfg)
    started = time.time()
    status, answer = "ok", None
    try:
        result = agent.run(skill, agdata(request="Start."))
        result.wait(timeout=float(os.environ.get("WAIT_S", "3000")))
        answer = str(result.answer)
    except Exception as exc:
        status = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if agent.sandbox is not None:
            agent.sandbox.destroy()
    dbs = list((run_dir / "logs").glob("agent_*_data.sqlite3"))
    rows = calls(dbs[0]) if dbs else []
    fits = 0
    if dbs:
        db = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
        fits = db.execute("select count(*) from events where type = 'context_fit'").fetchone()[0]
    record = {
        "run": name,
        "tag": args.tag,
        "harness": args.harness,
        "model": args.model,
        "provider": args.provider,
        "limit": args.limit,
        "status": status,
        "codes_found": answer.count("CODE-") if answer else 0,
        "files": args.files,
        "seconds": round(time.time() - started),
        "context_fits": fits,
        **summarize(rows, args.limit),
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "prompts"}))


if __name__ == "__main__":
    sys.exit(main())
