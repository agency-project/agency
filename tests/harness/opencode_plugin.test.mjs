// agency/harness/adapters/pty/_opencode_pty_plugin.js: the stop event carries
// the turn's final answer, also when OpenCode compacted in the middle of it.
import { test } from "node:test"
import assert from "node:assert/strict"
import { copyFile, mkdtemp, readdir, readFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

// OpenCode loads the plugin as an ES module; without a package.json that says
// so, Node would not, so it is imported from a .mjs copy.
const source = join(dirname(fileURLToPath(import.meta.url)), "../../agency/harness/adapters/pty/_opencode_pty_plugin.js")
const copy = join(await mkdtemp(join(tmpdir(), "agency-plugin-")), "plugin.mjs")
await copyFile(source, copy)
const { AgencyPlugin } = await import(pathToFileURL(copy).href)

function assistant(id, parentID, text, { finish = "stop", summary = false } = {}) {
  return {
    info: { id, role: "assistant", parentID, summary, finish, time: { completed: 1 }, tokens: { input: 10, output: 1 } },
    parts: text ? [{ type: "text", text }] : [{ type: "tool", tool: "read" }],
  }
}

function user(id, text) {
  return { info: { id, role: "user" }, parts: [{ type: "text", text }] }
}

async function stopEvent(messages) {
  process.env.AGENCY_PTY_STATE = await mkdtemp(join(tmpdir(), "agency-opencode-"))
  const client = { session: { messages: async () => ({ data: messages }) } }
  const hooks = await AgencyPlugin({ client })
  await hooks["chat.message"]({ sessionID: "s" }, { message: { id: "u1" }, parts: [{ type: "text", text: "task" }] })
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "s" } } })
  const dir = join(process.env.AGENCY_PTY_STATE, "events")
  const events = await Promise.all((await readdir(dir)).map(async f => JSON.parse(await readFile(join(dir, f), "utf8"))))
  return events.find(e => e.kind === "stop")
}

test("the final answer of a turn without compaction", async () => {
  const stop = await stopEvent([
    user("u1", "task"),
    assistant("a1", "u1", null, { finish: "tool-calls" }),
    assistant("a2", "u1", "CODE-01 CODE-02"),
  ])
  assert.equal(stop.turn_id, "u1")
  assert.equal(stop.text, "CODE-01 CODE-02")
})

test("the final answer after OpenCode compacted mid-turn", async () => {
  const stop = await stopEvent([
    user("u1", "task"),
    assistant("a1", "u1", null, { finish: "tool-calls" }),
    user("c1", ""),
    assistant("s1", "c1", "summary of the work so far", { summary: true }),
    user("c2", "Continue if you have next steps"),
    assistant("a2", "c2", null, { finish: "tool-calls" }),
    assistant("a3", "c2", "CODE-01 CODE-02"),
  ])
  assert.equal(stop.turn_id, "u1")
  assert.equal(stop.text, "CODE-01 CODE-02")
})
