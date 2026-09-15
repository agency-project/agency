import { mkdir, writeFile, rename } from "node:fs/promises"
import { randomUUID } from "node:crypto"
import { join } from "node:path"

export const AgencyPlugin = async ({ client }) => {
  const root = process.env.AGENCY_PTY_STATE
  const pending = new Map()
  const turns = new Map()
  async function emit(event) {
    const dir = join(root, "events")
    await mkdir(dir, { recursive: true })
    const path = join(dir, `${Date.now()}-${randomUUID()}`)
    await writeFile(path + ".tmp", JSON.stringify(event))
    await rename(path + ".tmp", path + ".json")
  }
  async function post(path, body) {
    const response = await fetch(process.env.AGPOLICY_BASE_URL.replace(/\/$/, "") + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + process.env.AGPOLICY_TOKEN },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    })
    if (!response.ok) throw new Error(`agpolicy HTTP ${response.status}`)
    return response.json()
  }
  return {
    "chat.message": async (input, output) => {
      const turn = { session_id: input.sessionID, turn_id: output.message.id,
        prompt: output.parts.filter(p => p.type === "text").map(p => p.text).join("\n") }
      turns.set(input.sessionID, turn)
    },
    event: async ({ event }) => {
      if (event.type === "message.part.updated") {
        const part = event.properties.part
        const turn = turns.get(part.sessionID)
        // chat.message runs before persistence and other plugins can reject it.
        // A committed, matching user text part is the acceptance boundary.
        if (turn && !turn.acknowledged && part.messageID === turn.turn_id && part.type === "text" && part.text === turn.prompt) {
          turn.acknowledged = true
          await emit({ session_id: turn.session_id, turn_id: turn.turn_id, kind: "submit", prompt: turn.prompt })
        }
        return
      }
      if (event.type !== "session.idle" && event.type !== "session.error") return
      const sessionID = event.properties.sessionID
      const turn = turns.get(sessionID)
      if (!turn) return
      // Read native committed messages. An idle notification by itself is not
      // success: cancellation and errors also make the session idle. Session
      // errors have no parent ID, so never stamp them with the newest turn;
      // recover the error from a committed message with the matching parent.
      const response = await client.session.messages({ path: { id: sessionID } })
      if (response.error) {
        await emit({ ...turn, kind: "error", error: JSON.stringify(response.error) })
        return
      }
      const messages = response.data || []
      const replies = messages.filter(m => m.info.role === "assistant" && m.info.parentID === turn.turn_id)
      const reply = replies.at(-1)
      if (!reply) return
      if (reply.info.error) {
        await emit({ ...turn, kind: reply.info.error.name === "MessageAbortedError" ? "interrupt" : "error", error: JSON.stringify(reply.info.error) })
        return
      }
      if (!reply.info.time.completed || !reply.info.finish || reply.info.finish === "tool-calls") return
      const text = reply.parts.filter(p => p.type === "text" && !p.synthetic).map(p => p.text).join("\n")
      await emit({ ...turn, kind: "stop", text,
        input_tokens: replies.reduce((sum, m) => sum + (m.info.tokens?.input || 0), 0),
        output_tokens: replies.reduce((sum, m) => sum + (m.info.tokens?.output || 0), 0) })
    },
    "tool.execute.before": async (input, output) => {
      const decision = await post("/agpolicy/check_tool", { tool_name: input.tool, tool_input: output.args })
      if (decision.decision !== "allow") throw new Error(decision.reason || "denied by agpolicy")
      if (decision.call_id) pending.set(input.callID, decision.call_id)
    },
    "tool.execute.after": async (input, output) => {
      const callId = pending.get(input.callID)
      pending.delete(input.callID)
      if (callId) {
        try {
          await post("/agpolicy/complete_tool", { call_id: callId, result: output.output, error: null })
        } catch (error) {
          console.error("[Agency] tool completion telemetry failed:", error)
        }
      }
    },
  }
}
