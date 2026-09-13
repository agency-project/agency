// Installed into the pinned Perfetto build by build_perfetto.py.
import type {PerfettoPlugin} from '../../public/plugin';
import type {Trace} from '../../public/trace';
import {TrackNode} from '../../public/workspace';
import {LONG, NUM, STR} from '../../trace_processor/query_result';
import {SourceDataset} from '../../trace_processor/dataset';
import {SliceTrack} from '../../components/tracks/slice_track';
import TraceProcessorTrack from '../dev.perfetto.TraceProcessorTrack';
import m from 'mithril';
import {Button} from '../../widgets/button';

interface Presentation {
  version: number;
  path: [string, string][];
  label?: string;
  laneLabel?: string;
  shared_with?: string[];
}

function decode(raw: string): Presentation | undefined {
  try {
    const p = JSON.parse(raw);
    if (p.version !== 1 || !Array.isArray(p.path) || p.path.length === 0 ||
        p.path.length > 16 || !p.path.every((v: unknown) => Array.isArray(v) &&
          v.length === 2 && v.every((s) => typeof s === 'string')) ||
        (p.label !== undefined && typeof p.label !== 'string') ||
        (p.shared_with !== undefined && (!Array.isArray(p.shared_with) ||
          !p.shared_with.every((s: unknown) => typeof s === 'string')))) return;
    return p;
  } catch { return; }
}

export default class AgencyPresentation implements PerfettoPlugin {
  static readonly id = 'dev.agency.Presentation';
  static readonly dependencies = [TraceProcessorTrack];

  async onTraceLoad(trace: Trace): Promise<void> {
    const result = await trace.engine.query(`
      SELECT s.track_id AS id, extract_arg(s.arg_set_id, 'args.agency_presentation') AS spec,
        coalesce(t.name, 'Execution') AS label
      FROM slice s LEFT JOIN thread_track tt ON tt.id = s.track_id
      LEFT JOIN thread t USING (utid)
      WHERE spec IS NOT NULL GROUP BY s.track_id
      UNION ALL
      SELECT id, substr(name, 19) AS spec, '' AS label FROM counter_track
      WHERE name GLOB 'agency-counter-v1:*'
    `);
    const specs = new Map<number, Presentation>();
    for (const it = result.iter({id: NUM, spec: STR, label: STR}); it.valid(); it.next()) {
      // Chrome counter import appends its single numeric argument name.
      const p = decode(it.spec.replace(/ value$/, ''));
      if (p) {
        p.laneLabel = it.label;
        specs.set(it.id, p);
      }
    }
    if (!specs.size) return;
    trace.onTraceReady.addListener(async () => {
      const workspace = trace.workspaces.currentWorkspace;
      const original = [...workspace.flatTracks];
      const emptied = new Set<TrackNode>();
      const groups = new Map<string, {node: TrackNode; sliceIds: Set<number>}>();
      const rank: Record<string, number> = {workflow: 0, orchestrator: 1, shared: 3, unattributed: 4};
      for (const node of original) {
        const track = node.uri ? trace.tracks.getTrack(node.uri) : undefined;
        const ids = track?.tags?.trackIds ?? [];
        const spec = ids.map((id) => specs.get(id)).find((p) => p !== undefined);
        if (!spec) continue;
        let parent: TrackNode | undefined;
        const path: string[] = [];
        for (const [id, name] of spec.path) {
          path.push(id);
          const key = JSON.stringify(path);
          let group = groups.get(key);
          if (!group) {
            const uri = `/agency/group/${groups.size}`;
            const node = new TrackNode({name, uri, collapsed: true,
              sortOrder: path.length === 1 ? rank[id] ?? 2 : id === 'host' ? 0 : 1});
            group = {node, sliceIds: new Set()};
            groups.set(key, group);
            if (parent) parent.addChildInOrder(node);
            else workspace.addChildInOrder(node);
          }
          if (!spec.label) ids.forEach((id) => group.sliceIds.add(id));
          parent = group.node;
        }
        node.name = spec.label ?? spec.laneLabel ?? node.name;
        for (let old = node.parent; old instanceof TrackNode; old = old.parent) {
          emptied.add(old);
        }
        parent?.addChildLast(node);
      }
      // Remove the now-empty default process/thread containers. Their summary
      // renderers describe physical processes, so they do not belong here.
      for (const node of original.reverse()) {
        if (!node.hasChildren && emptied.has(node)) {
          node.remove();
        }
      }
      // Shortcuts have no dataset, slices, counters or summary contribution.
      // Every agent navigates to the same canonical shared-sandbox group.
      const shortcuts = new Set<string>();
      for (const spec of specs.values()) {
        if (spec.path[0][0] !== 'shared' || spec.path.length < 2 || !spec.shared_with) continue;
        const sandboxPath = spec.path.slice(0, 2);
        const targetKey = JSON.stringify(sandboxPath.map(([id]) => id));
        const target = groups.get(targetKey)?.node;
        if (!target?.uri) continue;
        for (const agent of spec.shared_with) {
          const key = JSON.stringify([agent, targetKey]);
          if (shortcuts.has(key)) continue;
          shortcuts.add(key);
          const agentKey = JSON.stringify([`agent:${agent}`]);
          let group = groups.get(agentKey);
          if (!group) {
            group = {node: new TrackNode({name: `Agent ${agent}`, sortOrder: 2}), sliceIds: new Set()};
            groups.set(agentKey, group);
            workspace.addChildInOrder(group.node);
          }
          const uri = `/agency/reference/${shortcuts.size}`;
          const open = () => {
            target.reveal();
            target.expand();
            trace.scrollTo({track: {uri: target.uri!, expandGroup: true}});
          };
          trace.tracks.registerTrack({uri, renderer: {
            render: () => {},
            getHeight: () => 24,
            getTrackShellButtons: () => m(Button, {icon: 'open_in_new',
              title: `Open ${sandboxPath[1][1]} from Agent ${agent}`, compact: true, onclick: open}),
            onMouseClick: () => { open(); return true; },
          }});
          group.node.addChildLast(new TrackNode({uri, name: `↗ ${sandboxPath[1][1]} (shared)`,
            subtitle: 'Open the shared sandbox timeline'}));
        }
      }
      for (const {node, sliceIds} of groups.values()) {
        if (!sliceIds.size || !node.uri) continue;
        // Union the recorded intervals: gaps stay visible and nested calls
        // count once. This is recorded span activity, not CPU utilization.
        const sql = `(WITH ordered AS (
          SELECT ts, ts + max(dur, 0) AS end_ts,
            max(ts + max(dur, 0)) OVER (ORDER BY ts, dur
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
          FROM slice WHERE track_id IN (${[...sliceIds].join(',')})
        ), marked AS (
          SELECT *, sum(CASE WHEN prev_end IS NULL OR ts > prev_end THEN 1 ELSE 0 END)
            OVER (ORDER BY ts, end_ts) AS island FROM ordered
        ) SELECT row_number() OVER () AS id, min(ts) AS ts,
          max(end_ts) - min(ts) AS dur, 0 AS depth, 'Recorded activity' AS name
          FROM marked GROUP BY island)`;
        trace.tracks.registerTrack({uri: node.uri,
          renderer: await SliceTrack.create({trace, uri: node.uri,
            dataset: new SourceDataset({src: sql,
              schema: {id: NUM, ts: LONG, dur: LONG, depth: NUM, name: STR}}),
          }),
        });
        node.isSummary = true;
      }
    });
  }
}
